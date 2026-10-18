"""Convert a long vocal in slices so a small machine can keep up.

A 220 s pass holds the whole thing in VRAM and pins the machine; doing it in
45 s slices keeps peak memory flat and lets an interrupted run resume, because
finished slices are left on disk and skipped next time.

Slices are taken with a small overlap and cross-faded when stitched, so the
seams do not click.

  python scripts/chunked_convert.py --input lead.wav --output lead-conv.wav \
      --chunk 45 --overlap 2 --python <venv python> --rvc <rvc dir> \
      --model assets/weights/sayue.pth --index logs/sayue/xxx.index --pitch 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf


def run(cmd: list[str], cwd: str) -> int:
    print(f"    $ {' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk", type=float, default=45.0)
    ap.add_argument("--overlap", type=float, default=2.0)
    ap.add_argument("--python", required=True, help="python of the RVC venv")
    ap.add_argument("--rvc", required=True, help="RVC repo dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.5)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--tmp", default="")
    args = ap.parse_args()

    y, sr = sf.read(args.input, always_2d=True, dtype="float32")
    # RVC 的输出固定是 48k。如果输入不是 48k，后面算重叠和截断就会用错样本数，
    # 整段会被悄悄压缩（之前就踩过：220s 变成了 202s，越往后错位越多）。
    if sr != 48000:
        import librosa
        y = np.stack([librosa.resample(y[:, c], orig_sr=sr, target_sr=48000)
                      for c in range(y.shape[1])], axis=1).astype(np.float32)
        print(f"  输入重采样 {sr} -> 48000")
        sr = 48000
    total = len(y) / sr
    tmp = args.tmp or (os.path.splitext(args.input)[0] + "_chunks")
    os.makedirs(tmp, exist_ok=True)

    step = max(args.chunk - args.overlap, 5.0)
    starts = []
    t = 0.0
    while t < total - 0.05:
        starts.append(round(t, 2))
        t += step
    print(f"  {total:.1f}s -> {len(starts)} 块（每块 {args.chunk}s，重叠 {args.overlap}s）")

    made = skipped = failed = 0
    for i, st in enumerate(starts):
        seg_in = os.path.join(tmp, f"in_{i:02d}.wav")
        seg_out = os.path.join(tmp, f"out_{i:02d}.wav")
        if os.path.isfile(seg_out):
            skipped += 1
            continue
        a = int(st * sr)
        b = min(int((st + args.chunk) * sr), len(y))
        sf.write(seg_in, y[a:b], sr, subtype="PCM_24")
        cmd = [args.python, "-m", "infer.cli", "--model", args.model,
               "--input", seg_in, "--output", seg_out,
               "--pitch", str(args.pitch), "--f0-method", "rmvpe",
               "--protect", str(args.protect), "--index-rate", str(args.index_rate),
               "--rms-mix-rate", "1.0"]
        if args.index:
            cmd += ["--index", args.index]
        print(f"  [{i + 1}/{len(starts)}] {st:.1f}s", flush=True)
        if run(cmd, args.rvc) == 0 and os.path.isfile(seg_out):
            made += 1
        else:
            failed += 1
            print("    失败", flush=True)

    done = sum(1 for i in range(len(starts)) if os.path.isfile(os.path.join(tmp, f"out_{i:02d}.wav")))
    print(f"  块完成 {done}/{len(starts)}（本次新做 {made}，跳过 {skipped}，失败 {failed}）")
    if done < len(starts):
        print("  还有块没转完，重跑本命令会继续")
        return 1

    # 交叉淡化拼接
    ov = int(args.overlap * sr)
    pieces = []
    for i in range(len(starts)):
        seg, _ = sf.read(os.path.join(tmp, f"out_{i:02d}.wav"), always_2d=True, dtype="float32")
        if i > 0 and ov > 0 and len(pieces[-1]) >= ov and len(seg) >= ov:
            fade = np.linspace(0, 1, ov, dtype=np.float32)[:, None]
            tail = pieces[-1][-ov:]
            pieces[-1] = pieces[-1][:-ov]
            seg = seg.copy()
            seg[:ov] = tail * (1 - fade) + seg[:ov] * fade
        pieces.append(seg)
    out = np.concatenate(pieces, axis=0)[: len(y)]
    sf.write(args.output, out, sr, subtype="PCM_24")
    print(f"  写出 {args.output}（{len(out) / sr:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
