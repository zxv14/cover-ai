"""One-glance training status: progress, speed, loss trend, overfitting signal.

Written because the interesting question during a run is never "what step is it"
but "is validation still improving, or is it starting to overfit".

  python scripts/train_status.py --exp exp/sayue --repo tools/DDSP-SVC-6.3
"""
from __future__ import annotations

import argparse
import os
import re
import time

ROOT = r"D:\17551\Documents\ai-cover-ddsp"

STEP_RE = re.compile(r"step:\s*(\d+)")
# the log prints 'loss: {:.3f}. ' - capture digits and a single decimal point only,
# otherwise the trailing sentence period gets swallowed and float() fails
LOSS_RE = re.compile(r"loss:\s*(\d+\.\d+)")
SPEED_RE = re.compile(r"batch/s:\s*([\d.]+)")
TIME_RE = re.compile(r"time:\s*([\d:]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=os.path.join(ROOT, "tools", "DDSP-SVC-6.3", "exp", "sayue"))
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "25-train-sayue.log"))
    ap.add_argument("--tail", type=int, default=6)
    args = ap.parse_args()

    print(f"exp: {args.exp}")
    info = os.path.join(args.exp, "log_info.txt")
    if not os.path.isfile(info):
        print("  no log_info.txt yet")
        return 1

    with open(info, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # training steps (epoch lines) and validation losses ("loss: x." after a banner)
    last_step = 0
    speeds: list[float] = []
    for line in lines:
        m = STEP_RE.search(line)
        if m:
            last_step = int(m.group(1))
        s = SPEED_RE.search(line)
        if s:
            speeds.append(float(s.group(1)))

    val_losses: list[float] = []
    for i, line in enumerate(lines):
        if "validation" in line.lower() and i + 1 < len(lines):
            m = LOSS_RE.search(lines[i + 1])
            if m:
                val_losses.append(float(m.group(1)))

    train_losses = [float(m.group(1)) for m in (LOSS_RE.search(l) for l in lines) if m]
    train_losses = [v for v in train_losses if v < 5]  # drop validation values

    print(f"  step            : {last_step:,}")
    if speeds:
        recent = speeds[-50:]
        print(f"  speed           : {sum(recent) / len(recent):.2f} batch/s (近 {len(recent)} 条)")
        total = re.search(r"time:\s*([\d:.]+)", lines[-1])
        if total:
            print(f"  elapsed         : {total.group(1)}")
        if last_step and recent:
            rate = sum(recent) / len(recent)
            for target in (20_000, 50_000, 100_000):
                if last_step < target:
                    eta = (target - last_step) / rate
                    print(f"  ETA {target // 1000:>3}k 步   : {eta / 3600:.1f} 小时")

    if train_losses:
        head = sum(train_losses[:20]) / min(len(train_losses), 20)
        tail = sum(train_losses[-20:]) / min(len(train_losses), 20)
        print(f"  训练 loss       : 开头 {head:.3f} → 现在 {tail:.3f}")

    if val_losses:
        print(f"  验证 loss       : " + " → ".join(f"{v:.3f}" for v in val_losses))
        if len(val_losses) >= 3:
            best = min(val_losses)
            best_i = val_losses.index(best)
            since = len(val_losses) - 1 - best_i
            print(f"  最佳验证 loss   : {best:.3f}（第 {best_i + 1} 次验证，即约 {best_i * 1000:,} 步）")
            if since >= 3 and val_losses[-1] > best * 1.03:
                print(f"  ** 过拟合信号 ** 验证 loss 已连续 {since} 次未改善，比最佳高 "
                      f"{(val_losses[-1] / best - 1) * 100:.1f}%")
                print(f"     建议：用 model_{best_i * 1000}.pt（若已被删则用最近的永久检查点），别再往下训")
            else:
                print("  判定            : 暂未过拟合，继续观察")

    ckpts = sorted(f for f in os.listdir(args.exp) if f.startswith("model_") and f.endswith(".pt"))
    if ckpts:
        print(f"  检查点 ({len(ckpts)})    : " + ", ".join(ckpts[-8:]))
        newest = max(ckpts, key=lambda f: os.path.getmtime(os.path.join(args.exp, f)))
        print(f"  最新            : {newest}  "
              f"({os.path.getsize(os.path.join(args.exp, newest)) / 1e6:.0f} MB, "
              f"{time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(os.path.join(args.exp, newest))))})")
    else:
        print("  检查点          : 还没有")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
