"""Filter an RVC training set down to the clips whose gaps are actually quiet.

MDX23C leaves accompaniment bleed behind: measured over 1764 slices the gap
level has a median of -39.8 dBFS while the voice sits near -12 dBFS, so more
than half the set carries audible music in the pauses and the model learns it.

This keeps only clips whose 10th-percentile frame level is at or below
--threshold, moves everything else into a sibling `_rejected` folder (nothing is
deleted), then writes the `filelist.txt` that RVC's train.py expects:

    <gt_wav>|<feature_npy>|<speaker_id>

  python scripts/rvc_filter_clean.py --exp rvc/logs/sayue --threshold -45
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import soundfile as sf


def gap_level(path: str, frame_seconds: float = 0.02) -> float:
    """10th percentile of short-term RMS, in dBFS - a proxy for bleed level."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    hop = max(int(sr * frame_seconds), 1)
    frames = [np.sqrt(np.mean(mono[i:i + hop] ** 2)) for i in range(0, max(len(mono) - hop, 1), hop)]
    db = 20 * np.log10(np.array(frames, dtype=np.float64) + 1e-12)
    return float(np.percentile(db, 10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="RVC experiment dir, e.g. rvc/logs/sayue")
    ap.add_argument("--threshold", type=float, default=-45.0, help="keep clips with gap level <= this")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = os.path.abspath(args.exp)
    gt = os.path.join(exp, "0_gt_wavs")
    feat = os.path.join(exp, "3_feature768")
    f0 = os.path.join(exp, "2a_f0")
    f0nsf = os.path.join(exp, "2b-f0nsf")

    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(gt, "*.wav")))
    print(f"  评估 {len(names)} 段（阈值 {args.threshold:.0f} dBFS）")

    keep, reject = [], []
    for i, name in enumerate(names, 1):
        q = gap_level(os.path.join(gt, name + ".wav"))
        (keep if q <= args.threshold else reject).append((name, q))
        if i % 300 == 0:
            print(f"    已评估 {i}/{len(names)}", flush=True)

    total = len(keep) + len(reject)
    print(f"  保留 {len(keep)} 段（{100*len(keep)/max(total,1):.1f}%），"
          f"剔除 {len(reject)} 段")
    if keep:
        qs = np.array([q for _, q in keep])
        print(f"  保留部分间隙电平: 中位 {np.median(qs):.1f}  最差 {qs.max():.1f} dBFS")

    if args.dry_run:
        return 0

    rej_root = os.path.join(os.path.dirname(exp), os.path.basename(exp) + "_rejected")
    for sub in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"):
        os.makedirs(os.path.join(rej_root, sub), exist_ok=True)

    moved = 0
    for name, _ in reject:
        for sub, ext in (("0_gt_wavs", ".wav"), ("3_feature768", ".npy"),
                         ("2a_f0", ".npy"), ("2b-f0nsf", ".npy")):
            src = os.path.join(exp, sub, name + ext)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(rej_root, sub, name + ext))
                moved += 1
    print(f"  已移出 {moved} 个文件 -> {rej_root}")

    mute_wav = os.path.join(os.path.dirname(exp), "mute", "0_gt_wavs", "mute48k.wav")
    mute_npy = os.path.join(os.path.dirname(exp), "mute", "3_feature768", "mute.npy")
    lines = []
    if os.path.isfile(mute_wav) and os.path.isfile(mute_npy):
        lines.append(f"{mute_wav}|{mute_npy}|{args.speaker}")
    else:
        print(f"  警告: 缺少 mute 占位文件，跳过那一行（{mute_wav}）")

    for name, _ in keep:
        lines.append(
            f"{os.path.join(gt, name + '.wav')}|"
            f"{os.path.join(feat, name + '.npy')}|{args.speaker}"
        )

    out = os.path.join(exp, "filelist.txt")
    with open(out, "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))
    print(f"  已写 {out}（{len(lines)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
