"""Mix a converted vocal stem with a backing track.

Pure numpy/soundfile/scipy - no pydub, no ffmpeg round trip, so the sample
values stay exactly what the models produced.

  python mix.py --vocal out/vocal.wav --backing work/backing.wav --out out/final.wav

Steps: resample both to the target rate, match channel counts, gain-match the
vocal to the backing by RMS, sum, then peak-limit to --peak-dbfs.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def load(path: str, target_sr: int, target_ch: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != target_sr:
        from math import gcd

        g = gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, target_sr // g, sr // g, axis=0).astype(np.float32)
    if audio.shape[1] != target_ch:
        if target_ch == 1:
            audio = audio.mean(axis=1, keepdims=True)
        else:  # mono -> stereo
            audio = np.repeat(audio[:, :1], target_ch, axis=1)
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def active_level(audio: np.ndarray, sr: int, frame_ms: float = 50.0, pct: float = 90.0) -> float:
    """Level while the part is actually playing.

    Plain RMS is misleading for gappy material (a lead with long rests reads as
    very quiet, so RMS matching blasts it). The 90th percentile of 50 ms frame
    levels tracks "how loud this is when it sounds".
    """
    hop = max(int(sr * frame_ms / 1000), 1)
    n = len(audio) // hop
    if n == 0:
        return rms(audio)
    frames = audio[: n * hop].reshape(n, hop, -1).astype(np.float64)
    levels = np.sqrt(np.mean(np.square(frames), axis=(1, 2)) + 1e-12)
    return float(np.percentile(levels, pct))


def stats(audio: np.ndarray, label: str) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"  {label:<22} peak {db(peak):7.2f} dBFS   rms {db(rms(audio)):7.2f} dBFS   "
          f"len {audio.shape[0] / 1:.0f} samples")


def soft_limit(audio: np.ndarray, sr: int, ceiling_dbfs: float = -1.0, range_db: float = 6.0,
               attack_ms: float = 4.0, release_ms: float = 150.0) -> tuple[np.ndarray, float]:
    """Peak limiter + normalise: squashes the top `range_db` of peaks, then
    brings the new peak back up to the ceiling. That is what actually lowers the
    crest factor, so the preview is louder without clipping.

    Returns (audio, crest_reduction_db).
    """
    ceiling = 10 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio.astype(np.float32), 0.0

    # internal threshold sits `range_db` below the program peak
    threshold = peak * (10 ** (-range_db / 20.0))
    env = np.abs(audio).max(axis=1) if audio.ndim > 1 else np.abs(audio)

    a_att = float(np.exp(-1.0 / max(sr * attack_ms / 1000.0, 1.0)))
    a_rel = float(np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0)))

    # gain from the INSTANTANEOUS peak: a one-pole envelope never reaches brief
    # transients, which is exactly what a limiter has to catch on this material
    gain = np.ones_like(env)
    over = env > threshold
    gain[over] = threshold / env[over]
    a_g = float(np.exp(-1.0 / max(sr * 2.0 / 1000.0, 1.0)))
    prev = 1.0
    for i, g in enumerate(gain):
        coeff = a_g if g < prev else a_att
        prev = coeff * prev + (1.0 - coeff) * g
        gain[i] = prev

    shaped = audio * gain[:, None] if audio.ndim > 1 else audio * gain

    before = db(peak) - db(rms(audio))
    new_peak = float(np.max(np.abs(shaped))) if shaped.size else 0.0
    norm = (ceiling / new_peak) if new_peak > 0 else 1.0
    shaped = shaped * norm
    after = db(float(np.max(np.abs(shaped)))) - db(rms(shaped))
    return shaped.astype(np.float32), before - after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--backing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampling-rate", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--vocal-db", type=float, default=0.0, help="extra gain on the vocal after matching")
    ap.add_argument("--backing-db", type=float, default=0.0, help="gain on the backing")
    ap.add_argument("--match-rms", action="store_true", default=True,
                    help="scale the vocal so its RMS matches the backing's")
    ap.add_argument("--no-match-rms", dest="match_rms", action="store_false")
    ap.add_argument("--peak-dbfs", type=float, default=-1.0)
    ap.add_argument("--limit", action="store_true",
                    help="apply the feed-forward limiter (loud, listenable preview)")
    ap.add_argument("--limit-range-db", type=float, default=6.0,
                    help="how much of the peak range the preview limiter squashes")
    args = ap.parse_args()

    vocal = load(args.vocal, args.sampling_rate, args.channels)
    backing = load(args.backing, args.sampling_rate, args.channels)
    print("loaded:")
    stats(vocal, "vocal")
    stats(backing, "backing")

    vocal_gain = 10 ** (args.vocal_db / 20.0)
    if args.match_rms:
        vl = active_level(vocal, args.sampling_rate)
        bl = active_level(backing, args.sampling_rate)
        if vl > 1e-9:
            vocal_gain *= bl / vl
            print(f"  active-level match (90th pct of 50 ms frames): "
                  f"vocal {db(vl):.2f} -> {db(bl):.2f} dB  ({db(bl / vl):+.2f} dB)")
    backing_gain = 10 ** (args.backing_db / 20.0)

    n = max(vocal.shape[0], backing.shape[0])
    mix = np.zeros((n, args.channels), dtype=np.float64)
    mix[: vocal.shape[0]] += vocal * vocal_gain
    mix[: backing.shape[0]] += backing * backing_gain

    peak = float(np.max(np.abs(mix)))
    target = 10 ** (args.peak_dbfs / 20.0)
    if peak > target:
        applied = target / peak
        mix *= applied
        print(f"  peak limit: {db(peak):.2f} -> {args.peak_dbfs:.2f} dBFS ({db(applied):+.2f} dB)")

    out = mix.astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, out, args.sampling_rate, subtype="PCM_24")
    print("result:")
    stats(out, "mix")
    print(f"  written {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if args.limit:
        loud, reduction = soft_limit(out, args.sampling_rate, ceiling_dbfs=args.peak_dbfs, range_db=args.limit_range_db)
        loud_path = args.out.replace(".wav", "").replace(".WAV", "") + "-loud.wav"
        sf.write(loud_path, loud, args.sampling_rate, subtype="PCM_24")
        print("limited preview:")
        stats(loud, "loud mix")
        print(f"  limiter reduction {reduction:+.2f} dB   written {loud_path} "
              f"({os.path.getsize(loud_path) / 1e6:.1f} MB)")
        over = int(np.sum(np.abs(loud) >= 1.0))
        print(f"  samples at/above full scale: {over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
