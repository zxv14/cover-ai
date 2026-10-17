"""Thin wrapper that runs DDSP-SVC 6.3 inference with this project's paths.

Kept as a script so every run is reproducible and logged the same way.

  python run_ddsp.py --input <vocal.wav> --tag <name> [--key 0] [--step 50]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = r"D:\17551\Documents\ai-cover-ddsp"
DDSP = os.path.join(ROOT, "tools", "DDSP-SVC-6.3")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
MODEL = os.path.join(ROOT, "models", "sakiko-6.3", "sakiko.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--repo", default=DDSP,
                    help="DDSP-SVC checkout to run (default: 6.3; pass the 6.2 tree to test 6.2)")
    ap.add_argument("--model", default=MODEL, help="speaker .pt; its config.yaml must sit next to it")
    ap.add_argument("--key", type=float, default=0.0, help="pitch shift in semitones")
    ap.add_argument("--step", type=int, default=50, help="reflow sampling steps")
    ap.add_argument("--method", default="euler", choices=["euler", "rk4"])
    ap.add_argument("--pitch-extractor", default="rmvpe")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = args.output or os.path.join(ROOT, "output", os.path.splitext(os.path.basename(args.input))[0] + "-sakiko.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = [
        PY, "main_reflow.py",
        "-i", args.input,
        "-m", args.model,
        "-o", out,
        "-k", str(args.key),
        "-id", "1",
        "-step", str(args.step),
        "-method", args.method,
        "-pe", args.pitch_extractor,
    ]
    if args.device:
        cmd += ["-d", args.device]

    print("repo:", args.repo)
    print("model:", args.model)
    print("cmd:", " ".join(cmd), flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=args.repo)
    print(f"\nexit={proc.returncode} elapsed={time.time() - started:.1f}s")
    if os.path.isfile(out):
        print(f"output {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
