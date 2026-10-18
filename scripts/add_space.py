"""主唱加混响，然后和伴唱、伴奏混在一起。

主唱刚去过混响，本身没有空间感；伴唱保留了原版的混响。这里给主唱补一份
卷积混响（用合成的脉冲响应，不是延迟回声），让两条人声处在同一个空间里，
再依次叠上伴唱和伴奏。

  python scripts/add_space.py --lead 主唱.wav --harmony 伴唱.wav \
      --backing 伴奏.wav --out 成品.wav --wet 0.22 --fade-in 0.2
"""
from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def load(p: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(p, always_2d=True, dtype="float32")
    return y, sr


def resample_to(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y
    import librosa
    out = [librosa.resample(y[:, c], orig_sr=sr_from, target_sr=sr_to) for c in range(y.shape[1])]
    return np.stack(out, axis=1).astype(np.float32)


def make_ir(sr: int, decay: float = 0.42, length: float = 1.1, seed: int = 0) -> np.ndarray:
    """指数衰减噪声：简单但够用的房间感。"""
    n = int(sr * length)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 2)).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t / decay)[:, None]
    pre = int(sr * 0.012)          # 预延迟，保住字头
    ir = np.zeros((n + pre, 2), dtype=np.float32)
    ir[pre:] = noise * env
    k = np.ones(8, dtype=np.float32) / 8      # 稍削极高频，真实房间不会那么亮
    for ch in range(2):
        ir[:, ch] = np.convolve(ir[:, ch], k, mode="same")
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def reverb(y: np.ndarray, sr: int, wet: float) -> np.ndarray:
    if y.ndim == 1 or y.shape[1] == 1:
        y = np.repeat(y.reshape(-1, 1), 2, axis=1)
    ir = make_ir(sr)
    out = np.zeros((len(y) + len(ir) - 1, 2), dtype=np.float32)
    for ch in range(2):
        out[:, ch] = fftconvolve(y[:, ch], ir[:, ch])[: len(out)]
    out = out[: len(y)]
    return (1 - wet) * y + wet * out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def blend(mix: np.ndarray, other: np.ndarray, sr: int, other_sr: int,
          gain_db: float, label: str) -> np.ndarray:
    other = resample_to(other, other_sr, sr)
    if other.ndim == 1 or other.shape[1] == 1:
        other = np.repeat(other.reshape(-1, 1), 2, axis=1)
    m = min(len(other), len(mix))
    gain = 10 ** (gain_db / 20) * (rms(mix) / max(rms(other[:m]), 1e-9))
    mix[:m] += other[:m] * gain
    print(f"  混入{label} {m / sr:.1f}s，增益 {20 * np.log10(gain):+.1f} dB")
    return mix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", required=True, help="转换后的主唱（干声）")
    ap.add_argument("--harmony", default="", help="伴唱，留空则不加")
    ap.add_argument("--backing", default="", help="伴奏，留空则不加")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wet", type=float, default=0.22)
    ap.add_argument("--harmony-db", type=float, default=-3.0)
    ap.add_argument("--backing-db", type=float, default=-4.0)
    ap.add_argument("--fade-in", type=float, default=0.0, help="主唱淡入秒数，缓解第一句突兀")
    args = ap.parse_args()

    lead, sr = load(args.lead)
    n = len(lead)
    print(f"  主唱 {n / sr:.1f}s @ {sr}Hz")

    if args.fade_in > 0:
        nf = min(int(args.fade_in * sr), len(lead))
        lead[:nf] *= np.linspace(0, 1, nf, dtype=np.float32)[:, None]
        print(f"  主唱淡入 {args.fade_in * 1000:.0f} ms")

    lead = reverb(lead, sr, args.wet)
    print(f"  加混响 wet={args.wet}")

    mix = lead.copy()
    if args.harmony:
        h, hsr = load(args.harmony)
        mix = blend(mix, h, sr, hsr, args.harmony_db, "伴唱")
    if args.backing:
        b, bsr = load(args.backing)
        mix = blend(mix, b, sr, bsr, args.backing_db, "伴奏")

    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix = np.tanh(mix / peak * 1.1) / np.tanh(1.1)
        print("  软限幅")
    mix *= 10 ** (-1 / 20) / (float(np.max(np.abs(mix))) + 1e-9)

    sf.write(args.out, mix, sr, subtype="PCM_24")
    print(f"  写出 {args.out}（{len(mix) / sr:.1f}s，峰值 -1 dBFS）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
