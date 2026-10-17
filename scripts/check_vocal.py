"""Objective sanity checks on a converted vocal stem.

Answers three questions without listening:
  1. Does the converted vocal follow the original performance? (envelope correlation)
  2. Are there isolated spikes inflating the peak? (top-sample analysis)
  3. Did anything degenerate - silence, clipping, DC offset, wrong length?
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: str, sr_target: int = 44100) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr != sr_target:
        from math import gcd

        g = gcd(int(sr), sr_target)
        audio = resample_poly(audio, sr_target // g, sr // g).astype(np.float32)
    return audio


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def envelope(audio: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    hop = max(int(sr * frame_ms / 1000), 1)
    n = len(audio) // hop
    trimmed = audio[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(np.square(trimmed.astype(np.float64)), axis=1) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="vocal stem fed into the converter")
    ap.add_argument("--converted", required=True, help="converter output")
    args = ap.parse_args()

    orig = load_mono(args.original)
    conv = load_mono(args.converted)

    print(f"original : {len(orig) / 44100:8.2f}s  peak {db(np.max(np.abs(orig))):6.2f} dBFS  "
          f"rms {db(np.sqrt(np.mean(orig ** 2))):6.2f} dBFS  dc {np.mean(orig):+.2e}")
    print(f"converted: {len(conv) / 44100:8.2f}s  peak {db(np.max(np.abs(conv))):6.2f} dBFS  "
          f"rms {db(np.sqrt(np.mean(conv ** 2))):6.2f} dBFS  dc {np.mean(conv):+.2e}")

    n = min(len(orig), len(conv))
    eo, ec = envelope(orig[:n], 44100), envelope(conv[:n], 44100)
    m = min(len(eo), len(ec))
    eo, ec = eo[:m], ec[:m]
    corr = float(np.corrcoef(eo, ec)[0, 1])
    print(f"\nenvelope correlation (50 ms frames, {m} frames): {corr:+.4f}")
    print(f"  (1.0 = same performance timing; >0.8 is a healthy conversion)")

    # silence share
    thr = 10 ** (-45 / 20)
    print(f"  frames below -45 dBFS: original {(eo < thr).mean() * 100:5.1f}%  converted {(ec < thr).mean() * 100:5.1f}%")
    print(f"  envelope ratio (conv/orig), median: {db(np.median(ec / np.maximum(eo, 1e-9))):+.2f} dB")

    # spike analysis on the converted audio
    absd = np.abs(conv)
    peak = float(absd.max())
    top = np.sort(absd)[-max(int(len(absd) * 0.0001), 1):]
    print(f"\nspike check: peak {db(peak):+.2f} dBFS, top 0.01% mean {db(top.mean()):+.2f} dBFS "
          f"(crest {db(peak) - db(np.sqrt(np.mean(conv ** 2))):.1f} dB)")
    idx = np.argsort(absd)[-5:][::-1]
    print("  loudest samples (time, dBFS): " + ", ".join(f"{i / 44100:.2f}s {db(absd[i]):+.1f}" for i in sorted(idx)))
    for label, frac in (("first 0.2 s", 0.2), ("last 0.2 s", -0.2)):
        seg = conv[: int(0.2 * 44100)] if frac > 0 else conv[int(-0.2 * 44100):]
        print(f"  {label}: peak {db(np.max(np.abs(seg))):+.2f} dBFS  rms {db(np.sqrt(np.mean(seg ** 2))):+.2f} dBFS")

    clipped = int(np.sum(np.abs(conv) >= 0.999))
    print(f"  samples at/above full scale: {clipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
