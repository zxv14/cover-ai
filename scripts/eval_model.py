"""Evaluate a trained DDSP-SVC model end to end, in one command.

Does the four things that actually answer "is this model usable":
  1. check_voicepack  - will inference even load it?
  2. inference        - convert a real song vocal into the trained voice
  3. check_vocal      - objective sanity (does it follow the performance?)
  4. mix              - audible cover against the backing track

Defaults are wired for this project so that after training finishes the whole
evaluation is a single call:

  python scripts/eval_model.py --exp tools/DDSP-SVC-6.3/exp/sayue --tag sayue

  python scripts/eval_model.py --model <path.pt> --test-vocal <wav> --tag custom
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = r"D:\17551\Documents\ai-cover-ddsp"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
REPO = os.path.join(ROOT, "tools", "DDSP-SVC-6.3")
DEFAULT_VOCAL = os.path.join(ROOT, "work", "sep_output", "reference-original", "vocals.wav")
DEFAULT_BACKING = os.path.join(ROOT, "work", "backing-official.wav")


def run(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = os.path.join(ROOT, "work", "cache", "matplotlib")
    env["HF_HOME"] = os.path.join(ROOT, "work", "cache", "hf")
    if capture:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    return subprocess.run(cmd, env=env).returncode, ""


def pick_checkpoint(exp: str, prefer: str) -> str:
    if prefer:
        return prefer
    ckpts = glob.glob(os.path.join(exp, "model_*.pt"))
    if not ckpts:
        raise SystemExit(f"no model_*.pt in {exp}")
    # highest step number, not newest mtime: a low-step file can be touched later
    def step_of(path: str) -> int:
        try:
            return int(os.path.basename(path).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    return max(ckpts, key=step_of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=os.path.join(REPO, "exp", "sayue"))
    ap.add_argument("--model", default="", help="explicit checkpoint; default = highest step in --exp")
    ap.add_argument("--test-vocal", default=DEFAULT_VOCAL)
    ap.add_argument("--backing", default=DEFAULT_BACKING)
    ap.add_argument("--tag", default="sayue")
    ap.add_argument("--key", type=float, default=0.0, help="pitch shift in semitones")
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--skip-mix", action="store_true")
    args = ap.parse_args()

    report: dict = {"tag": args.tag, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    model = pick_checkpoint(args.exp, args.model)
    print(f"model: {model}")
    report["model"] = model
    report["model_step"] = os.path.basename(model)

    # ---- 1. can it load at all -------------------------------------------------
    print("\n[1/4] check_voicepack")
    code, out = run([PY, os.path.join(ROOT, "scripts", "check_voicepack.py"),
                     "--model", model, "--repo", REPO])
    verdict = "PASS" if "=> this voice pack can be used" in out else "FAIL"
    report["voicepack"] = verdict
    for line in out.splitlines():
        if "BLOCKER" in line or "=>" in line or "load_model_vocoder" in line:
            print("   " + line.strip())
    if verdict == "FAIL":
        print("   体检不通过，后续步骤没有意义，停止。")
        _write(report, args.tag)
        return 1

    # ---- 2. inference ---------------------------------------------------------
    print("\n[2/4] inference（把测试干声转成该音色）")
    if not os.path.isfile(args.test_vocal):
        print(f"   缺少测试干声: {args.test_vocal}")
        return 1
    out_vocal = os.path.join(ROOT, "output", f"{args.tag}-vocal-raw.wav")
    t0 = time.time()
    code, out = run([PY, os.path.join(ROOT, "scripts", "run_ddsp.py"),
                     "--repo", REPO, "--model", model, "--input", args.test_vocal,
                     "--output", out_vocal, "--key", str(args.key), "--step", str(args.step)])
    report["inference_seconds"] = round(time.time() - t0, 1)
    report["vocal_out"] = out_vocal
    ok = os.path.isfile(out_vocal)
    report["inference_ok"] = ok
    print(f"   exit={code} 耗时 {report['inference_seconds']}s 产出 {'OK' if ok else '失败'}")
    if not ok:
        _write(report, args.tag)
        return 1

    # ---- 3. objective checks --------------------------------------------------
    print("\n[3/4] check_vocal（客观校验）")
    code, out = run([PY, os.path.join(ROOT, "scripts", "check_vocal.py"),
                     "--original", args.test_vocal, "--converted", out_vocal])
    for line in out.splitlines():
        if "envelope correlation" in line or "peak" in line or "frames below" in line:
            print("   " + line.strip())
    corr = None
    for line in out.splitlines():
        if "envelope correlation" in line and ":" in line:
            try:
                corr = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    report["envelope_correlation"] = corr
    print(f"   判定: {'健康（>0.8，跟随原唱时间线）' if corr and corr > 0.8 else '偏低，需人工试听'}")

    # timbre check: envelope correlation saturates early, so also measure whether
    # the spectrum actually moved toward the target speaker
    target_dir = os.path.join(ROOT, "work", "dataset-sayue", "train", "audio")
    if os.path.isdir(target_dir):
        print("\n[3.5/4] speaker_similarity（音色是否真的靠拢）")
        code, out = run([PY, os.path.join(ROOT, "scripts", "speaker_similarity.py"),
                         "--input", args.test_vocal, "--converted", out_vocal,
                         "--target-dir", target_dir])
        for line in out.splitlines():
            if "vs 目标音色" in line or "接近目标的程度" in line or "噪声底" in line:
                print("   " + line.strip())
        moved = None
        for line in out.splitlines():
            if "接近目标的程度" in line and ":" in line:
                try:
                    moved = float(line.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
        report["timbre_gain_db"] = moved
        if moved is not None:
            print(f"   判定: {'音色明显向目标移动' if moved > 1.0 else '移动不明显，模型可能还没学到音色'}")

    # ---- 4. mix ---------------------------------------------------------------
    if not args.skip_mix and os.path.isfile(args.backing):
        print("\n[4/4] mix（混音成品）")
        out_mix = os.path.join(ROOT, "output", f"{args.tag}-cover.wav")
        code, out = run([PY, os.path.join(ROOT, "scripts", "mix.py"),
                         "--vocal", out_vocal, "--backing", args.backing,
                         "--out", out_mix, "--vocal-db", "3", "--limit", "--limit-range-db", "10"])
        for line in out.splitlines():
            if "mix " in line or "limiter" in line or "full scale" in line:
                print("   " + line.strip())
        report["mix_out"] = out_mix
        report["mix_exists"] = os.path.isfile(out_mix)
        print(f"   -> {out_mix}")
    else:
        print("\n[4/4] mix 跳过")

    _write(report, args.tag)
    print(f"\n报告: {os.path.join(ROOT, 'output', f'{args.tag}-eval.json')}")
    return 0


def _write(report: dict, tag: str) -> None:
    path = os.path.join(ROOT, "output", f"{tag}-eval.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
