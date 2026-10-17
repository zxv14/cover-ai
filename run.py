#!/usr/bin/env python
"""一个入口跑所有工具。

不用记 12 个脚本的名字，用子命令调用：

    python run.py list

    python run.py slice  --use-separated work/vocal --out work/dataset
    python run.py filter --exp rvc/logs/sayue --threshold -45
    python run.py config --name myvoice --train work/dataset/train --val work/dataset/val
    python run.py infer  --repo tools/DDSP-SVC-6.3 --model exp/x/model_30000.pt \
                         --input dry.wav --output converted.wav --step 100
    python run.py mix    --vocal converted.wav --backing inst.wav --out final.wav
    python run.py eval   --exp tools/DDSP-SVC-6.3/exp/x --tag v1

每个子命令的参数与对应脚本完全一致，加 --help 可以看说明：

    python run.py mix --help
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

# 子命令 -> (脚本, 一句话说明)
COMMANDS = {
    "slice":  ("extract_vocals.py",      "把干声切成训练片段并质检"),
    "filter": ("rvc_filter_clean.py",    "按静音段电平筛掉带伴奏残留的片段"),
    "config": ("make_train_config.py",   "生成 DDSP 训练配置"),
    "patch":  ("patch_train_windows.py", "修 Windows 下的多进程 pickle 崩溃"),
    "status": ("train_status.py",        "查看训练进度与 loss 曲线"),
    "watch":  ("watch_train.py",         "训练健康检查（NaN / 发散 / 检查点）"),
    "infer":  ("run_ddsp.py",            "用训练好的模型做音色转换"),
    "mix":    ("mix.py",                 "转换结果与伴奏对齐混音"),
    "check":  ("check_voicepack.py",     "校验模型文件能否被推理加载"),
    "follow": ("check_vocal.py",         "包络相关：转换是否跟随原唱时间轴"),
    "speaker": ("speaker_id.py",         "说话人相似度：像不像目标"),
    "eval":   ("eval_model.py",          "一条命令跑完上述评估"),
}


def usage() -> int:
    print("用法: python run.py <命令> [参数...]\n")
    print("命令:")
    for name, (script, desc) in COMMANDS.items():
        print(f"  {name:<8} {desc}")
        print(f"           ({script})")
    print("\n加 --help 看某个命令的参数，例如: python run.py mix --help")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "list", "help"):
        return usage()

    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"未知命令: {cmd}\n", file=sys.stderr)
        usage()
        return 2

    script, _ = COMMANDS[cmd]
    path = os.path.join(SCRIPTS, script)
    if not os.path.isfile(path):
        print(f"找不到 {path}，确认脚本还在 scripts/ 下", file=sys.stderr)
        return 1

    # 用当前解释器跑，保持环境一致；工作目录固定在仓库根，脚本里的相对路径才成立
    return subprocess.run([sys.executable, path, *argv[1:]], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
