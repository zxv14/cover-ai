"""Watch a DDSP-SVC run for signs of going off the rails.

Reading only. Answers four questions each time it runs:
  1. is it actually progressing (step / speed / ETA)
  2. is the training loss still going down
  3. is the validation loss continuing to improve, or has it turned
  4. did the log develop NaNs, inf losses or tracebacks

Exit code is 0 when nothing looks wrong, 1 when a warning fires, so it can be
used as a gate.

  python scripts/watch_train.py --exp tools/DDSP-SVC-6.3/exp/sayue-sing \
      --log logs/41-train-sing-task.log
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = r"D:\17551\Documents\ai-cover-ddsp"


def parse_log(path: str) -> dict:
    out = {"steps": [], "train_losses": [], "val": [], "bad": []}
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "loss:" in line and "step" in line:
                m_step = re.search(r"step:\s*(\d+)", line)
                m_loss = re.search(r"loss:\s*([\d.]+)", line)
                if m_step and m_loss:
                    out["steps"].append(int(m_step.group(1)))
                    out["train_losses"].append(float(m_loss.group(1).rstrip(".")))
            if line.strip().startswith("loss:") and "step:" not in line:
                m = re.search(r"loss:\s*([\d.]+)", line)
                if m:
                    out["val"].append(float(m.group(1).rstrip(".")))
            low = line.lower()
            if any(k in low for k in ("nan", "inf", "traceback", "out of memory", "cuda error")):
                if "infomax" not in low and "inference" not in low:
                    out["bad"].append(line[:160])
    return out


def trend(values: list[float]) -> str:
    if len(values) < 3:
        return "样本不足"
    recent = values[-6:]
    first, last = sum(recent[: len(recent) // 2]) / max(len(recent) // 2, 1), \
        sum(recent[len(recent) // 2:]) / max(len(recent) - len(recent) // 2, 1)
    delta = last - first
    if delta < -0.01:
        return f"下降中 ({delta:+.3f})"
    if delta > 0.03:
        return f"上升 ({delta:+.3f}) 警告"
    return f"走平 ({delta:+.3f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--log", default="", help="default: <exp>/log_info.txt, which survives restarts")
    ap.add_argument("--window", type=int, default=6, help="how many recent points to judge on")
    args = ap.parse_args()

    log_path = args.log or os.path.join(args.exp, "log_info.txt")
    data = parse_log(log_path)
    warn = False

    ckpts = sorted(glob.glob(os.path.join(args.exp, "model_*.pt")),
                   key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
    latest = os.path.basename(ckpts[-1]) if ckpts else "无"
    step = data["steps"][-1] if data["steps"] else 0

    # speed from the log's own timestamps
    speed = None
    if os.path.isfile(args.log):
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-40:]
        times = [re.search(r"time:\s*(\d+):(\d+):([\d.]+)", ln) for ln in tail]
        steps = [re.search(r"step:\s*(\d+)", ln) for ln in tail]
        tm = [(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), int(s.group(1)))
              for m, s in zip(times, steps) if m and s]
        if len(tm) >= 2 and tm[-1][0] > tm[0][0]:
            speed = (tm[-1][1] - tm[0][1]) / (tm[-1][0] - tm[0][0])

    print(f"  step          : {step:,}   最新检查点 {latest}")
    if speed:
        eta = (50000 - step) / speed / 3600
        print(f"  speed         : {speed:.2f} step/s   ETA 50k 约 {eta:.1f} 小时")

    tl = data["train_losses"][-1] if data["train_losses"] else float("nan")
    print(f"  训练 loss     : {tl:.3f}   近期趋势 {trend(data['train_losses'])}")

    if data["val"]:
        print(f"  验证 loss     : {len(data['val'])} 个点   {trend(data['val'])}")
        print(f"                  序列: " + " → ".join(f"{v:.3f}" for v in data["val"][-8:]))
        best = min(data["val"])
        best_idx = data["val"].index(best)
        if len(data["val"]) - 1 - best_idx >= 5:
            print(f"                  注意: 最佳值 {best:.3f} 出现在 {best_idx + 1} 个点之前，"
                  f"已连续 {len(data['val']) - 1 - best_idx} 次没有刷新")
            warn = True
        if data["val"][-1] > best + 0.08:
            print(f"                  警告: 当前 {data['val'][-1]:.3f} 比最佳值高 {data['val'][-1] - best:+.3f}")
            warn = True

    if data["bad"]:
        print(f"  日志异常      : {len(data['bad'])} 条")
        for line in data["bad"][-3:]:
            print(f"                  {line}")
        warn = True
    else:
        print("  日志异常      : 无 NaN / inf / traceback")

    if not ckpts:
        print("  检查点        : 还没有，无法恢复")
        warn = True

    print(f"  判定          : {'需要人工介入' if warn else '正常，继续'}")
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main())
