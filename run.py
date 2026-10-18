#!/usr/bin/env python
"""一个入口跑所有工具。

大部分子命令只是转发到 scripts/ 下的脚本，参数保持一致；
train / cover / doctor 是本文件自己实现的编排，一条命令跑完多步。

    python run.py list
    python run.py doctor
    python run.py train --repo tools/DDSP-SVC-6.3 --name myvoice --dry work/vocal
    python run.py cover --repo tools/DDSP-SVC-6.3 --model exp/myvoice/model_30000.pt \
                        --vocal song.wav --backing inst.wav --out final.wav
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

# 转发类：子命令 -> (脚本, 说明)
PASSTHROUGH = {
    "slice":   ("extract_vocals.py",      "把干声切成训练片段并质检"),
    "filter":  ("rvc_filter_clean.py",    "按静音段电平筛掉带伴奏残留的片段"),
    "config":  ("make_train_config.py",   "生成 DDSP 训练配置"),
    "patch":   ("patch_train_windows.py", "修 Windows 下的多进程 pickle 崩溃"),
    "status":  ("train_status.py",        "查看训练进度与 loss 曲线"),
    "watch":   ("watch_train.py",         "训练健康检查（NaN / 发散 / 检查点）"),
    "infer":   ("run_ddsp.py",            "用模型做音色转换（单步）"),
    "mix":     ("mix.py",                 "转换结果与伴奏混音（单步）"),
    "check":   ("check_voicepack.py",     "校验模型文件能否被推理加载"),
    "follow":  ("check_vocal.py",         "包络相关：转换是否跟随原唱时间轴"),
    "speaker": ("speaker_id.py",          "说话人相似度：像不像目标"),
    "eval":    ("eval_model.py",          "一条命令跑完上述评估"),
}

# 编排类：本文件实现
NATIVE = {
    "doctor":  "检查环境、上游仓库、底模是否就位",
    "train":   "干声目录 → 切片 → 生成配置 → 预处理 → 训练（全自动）",
    "cover":   "模型 + 干声 + 伴奏 → 转换 → 混音（全自动）",
    "cleanup": "收尾：结束残留的推理/训练/分离进程，释放显存和内存",
}

# 收尾时要清掉的进程特征
PIPELINE_MARKERS = (
    "infer.cli", "train.train", "inference.py", "msst_gui",
    "preprocess.py", "train_reflow", "extract_f0", "extract_hubert",
)


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run(cmd: list[str], cwd: str | None = None) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def cmd_cleanup(_args) -> int:
    """结束流水线残留进程。跑完不关会一直占着显存和内存，机器会变卡。"""
    try:
        import psutil
    except ImportError:
        print("  需要 psutil：pip install psutil", file=sys.stderr)
        return 1

    me = os.getpid()
    killed, freed = [], 0.0
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            info = proc.info
            if info["pid"] == me or "python" not in (info["name"] or "").lower():
                continue
            cmdline = " ".join(info["cmdline"] or [])
            if not any(marker in cmdline for marker in PIPELINE_MARKERS):
                continue
            mem = info["memory_info"].rss / 1e6 if info["memory_info"] else 0.0
            proc.kill()
            proc.wait(timeout=5)
            killed.append((info["pid"], mem))
            freed += mem
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

    if killed:
        for pid, mem in killed:
            say(f"已结束 PID {pid}（{mem:.0f} MB）")
        say(f"共 {len(killed)} 个进程，释放约 {freed / 1000:.2f} GB 内存")
    else:
        say("没有残留的流水线进程")

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
        total = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9
        say(f"显存 {used:.2f} / {total:.2f} GB 占用")
        pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        say(f"可用内存 {vm.available / 1e9:.1f} GB / {vm.total / 1e9:.1f} GB")
    except Exception:
        pass
    return 0


def resolve_repo(arg: str) -> str:
    """允许只写 6.3 这种简称，自动补成 tools/DDSP-SVC-6.3"""
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    cand = os.path.join(ROOT, "tools", arg)
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(ROOT, "tools", f"DDSP-SVC-{arg}")
    if os.path.isdir(cand):
        return cand
    raise SystemExit(f"找不到上游仓库: {arg}（先跑 python run.py doctor 看看）")


# ---------------- doctor ----------------
def cmd_doctor(_args) -> int:
    py = sys.executable
    ok = True

    print("\n=== Python 与 torch ===")
    say(f"解释器 {py}")
    code = "import torch,sys;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no-gpu')"
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        say(f"torch {r.stdout.strip()}")
        if "+cu128" not in r.stdout:
            say("! 不是 cu128，50 系显卡会报 no kernel image")
            ok = False
    else:
        say(f"! torch 导入失败: {r.stderr.strip()[:160]}")
        ok = False

    print("\n=== 外部工具 ===")
    for tool in ("ffmpeg", "ffprobe", "git"):
        p = shutil.which(tool)
        say(f"{tool:<8} {p if p else '缺失'}")
        if not p:
            ok = False

    print("\n=== 上游仓库 ===")
    for rel in ("tools/DDSP-SVC-6.3", "rvc"):
        p = os.path.join(ROOT, rel)
        say(f"{rel:<22} {'在' if os.path.isdir(p) else '缺失'}")
        if not os.path.isdir(p):
            ok = False

    print("\n=== 底模 ===")
    for rel in (
        "tools/DDSP-SVC-6.3/pretrain/contentvec/pytorch_model.bin",
        "tools/DDSP-SVC-6.3/pretrain/nsf_hifigan/model",
        "rvc/assets/hubert_base/pytorch_model.bin",
        "rvc/assets/rmvpe/rmvpe.pt",
        "rvc/assets/pretrained_v2/f0G48k.pth",
    ):
        p = os.path.join(ROOT, rel)
        size = os.path.getsize(p) / 1e6 if os.path.isfile(p) else 0
        say(f"{rel:<58} {'%.0f MB' % size if size else '缺失'}")
        if not size:
            ok = False

    print(f"\n结论: {'可以开工' if ok else '还有缺项，先跑 setup.ps1 或补下文件'}")
    return 0 if ok else 1


# ---------------- train ----------------
def cmd_train(args) -> int:
    repo = resolve_repo(args.repo)
    name = args.name
    dataset = os.path.join(ROOT, "work", f"dataset-{name}")

    print(f"\n=== 1/5 切片（{args.dry} → {dataset}） ===")
    cmd = [sys.executable, os.path.join(SCRIPTS, "extract_vocals.py"),
           "--use-separated", args.dry, "--out", dataset,
           "--min-sec", str(args.min_sec), "--max-sec", str(args.max_sec),
           "--val-clips", str(args.val_clips)]
    if run(cmd) != 0:
        return 1

    print("\n=== 2/5 生成训练配置 ===")
    cmd = [sys.executable, os.path.join(SCRIPTS, "make_train_config.py"),
           "--name", name,
           "--train", os.path.join(dataset, "train"),
           "--val", os.path.join(dataset, "val"),
           "--repo", repo]
    if run(cmd) != 0:
        return 1

    cfg_path = os.path.join(repo, "configs", f"train-{name}.yaml")

    print("\n=== 3/5 修配置（Windows 必需）===")
    try:
        import yaml
    except ImportError:
        say("! 缺 pyyaml，无法自动修配置，请手动改 num_workers=0 / cache_device=cuda")
        return 1
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section in ("data", "train"):
        if isinstance(cfg.get(section), dict):
            cfg[section]["num_workers"] = 0
            cfg[section]["cache_device"] = "cuda"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    say(f"已把 num_workers 设为 0、cache_device 设为 cuda（{cfg_path}）")

    print("\n=== 4/5 预处理（提特征） ===")
    cmd = [sys.executable, "preprocess.py", "-c", f"configs/train-{name}.yaml", "-j", "2"]
    if run(cmd, cwd=repo) != 0:
        return 1

    print("\n=== 5/5 训练 ===")
    say("按 Ctrl+C 可随时中断，检查点会保留，重跑本命令会继续")
    cmd = [sys.executable, "train_reflow.py", "-c", f"configs/train-{name}.yaml"]
    rc = run(cmd, cwd=repo)
    print(f"\n训练结束（exit={rc}）。检查点目录: {os.path.join(repo, 'exp', name)}")
    return rc


# ---------------- cover ----------------
def cmd_cover(args) -> int:
    repo = resolve_repo(args.repo)
    tmp_dir = os.path.join(ROOT, "work", "cover-tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    converted = os.path.join(tmp_dir, "converted.wav")

    print(f"\n=== 1/2 音色转换（{os.path.basename(args.vocal)}） ===")
    cmd = [sys.executable, os.path.join(SCRIPTS, "run_ddsp.py"),
           "--repo", repo, "--model", args.model,
           "--input", args.vocal, "--output", converted,
           "--key", str(args.key), "--step", str(args.step)]
    if args.device:
        cmd += ["--device", args.device]
    if run(cmd) != 0:
        return 1

    print("\n=== 2/2 混音 ===")
    cmd = [sys.executable, os.path.join(SCRIPTS, "mix.py"),
           "--vocal", converted, "--backing", args.backing, "--out", args.out,
           "--vocal-db", str(args.vocal_db), "--limit",
           "--limit-range-db", str(args.limit_range_db)]
    if run(cmd) != 0:
        return 1

    print(f"\n成品: {args.out}")
    return 0


# ---------------- 入口 ----------------
def usage() -> int:
    print("用法: python run.py <命令> [参数...]\n")
    print("一键流程:")
    for name, desc in NATIVE.items():
        print(f"  {name:<8} {desc}")
    print("\n单步工具:")
    for name, (script, desc) in PASSTHROUGH.items():
        print(f"  {name:<8} {desc}")
    print("\n加 --help 看某个命令的参数，例如: python run.py train --help")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "list", "help"):
        return usage()

    cmd, rest = argv[0], argv[1:]

    if cmd == "doctor":
        return cmd_doctor(rest)

    if cmd == "cleanup":
        return cmd_cleanup(rest)

    if cmd == "train":
        ap = argparse.ArgumentParser(prog="run.py train", description=NATIVE["train"])
        ap.add_argument("--repo", default="6.3", help="DDSP 仓库路径（默认 tools/DDSP-SVC-6.3）")
        ap.add_argument("--name", required=True, help="实验名")
        ap.add_argument("--dry", required=True, help="干声目录")
        ap.add_argument("--min-sec", type=float, default=2.5)
        ap.add_argument("--max-sec", type=float, default=10.0)
        ap.add_argument("--val-clips", type=int, default=30)
        return cmd_train(ap.parse_args(rest))

    if cmd == "cover":
        ap = argparse.ArgumentParser(prog="run.py cover", description=NATIVE["cover"])
        ap.add_argument("--repo", default="6.3", help="DDSP 仓库路径")
        ap.add_argument("--model", required=True, help="训练好的 .pt")
        ap.add_argument("--vocal", required=True, help="要转换的干声")
        ap.add_argument("--backing", required=True, help="伴奏")
        ap.add_argument("--out", required=True, help="成品路径")
        ap.add_argument("--key", type=float, default=0, help="升降调（半音）")
        ap.add_argument("--step", type=int, default=100, help="折返步数，建议 >=100")
        ap.add_argument("--device", default="", help="cuda / cpu，默认自动")
        ap.add_argument("--vocal-db", type=float, default=2.0)
        ap.add_argument("--limit-range-db", type=float, default=8.0)
        return cmd_cover(ap.parse_args(rest))

    if cmd in PASSTHROUGH:
        script = PASSTHROUGH[cmd][0]
        path = os.path.join(SCRIPTS, script)
        if not os.path.isfile(path):
            print(f"找不到 {path}", file=sys.stderr)
            return 1
        return subprocess.run([sys.executable, path, *rest], cwd=ROOT).returncode

    print(f"未知命令: {cmd}\n", file=sys.stderr)
    usage()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
