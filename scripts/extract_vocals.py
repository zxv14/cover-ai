"""Turn raw stream recordings into a DDSP-SVC training set (dry voice clips).

Design note - why this does NOT run a vocal separator by default
---------------------------------------------------------------
Separators (MelBand-Roformer / UVR / MDX) are trained on music. On a stream whose
mic is already dry they mostly add artefacts and cost ~35 min per hour of audio.
`analyze_stream.py` measures the noise floor: a 5th-percentile frame level near
-55 dBFS means music never plays continuously, so there is nothing to subtract.
Pass `--use-separated <dir>` to consume separator output instead when the source
really is a mixed song.

Pipeline
--------
  1. read mono 44.1 kHz, optional rumble high-pass
  2. silence-aware slicing (the DDSP repo's own slicer)
  3. per-clip QC: length, level, active ratio, clipping, DC offset
  4. write <out>/train/audio/*.wav + <out>/val/audio/*.wav, plus a report

  python scripts/extract_vocals.py --input work/streams/*.wav --out work/dataset-sayue \
      --repo tools/DDSP-SVC-6.3 --val-clips 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import soundfile as sf

ROOT = r"D:\17551\Documents\ai-cover-ddsp"


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def collect(inputs: list[str], vocals_only: bool = False) -> list[str]:
    files: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("wav", "flac"):
                files.extend(glob.glob(os.path.join(item, "**", f"*.{ext}"), recursive=True))
        elif os.path.isfile(item):
            files.append(item)
    if vocals_only:
        # separator output dirs also contain instrumental.wav - never train on that
        files = [f for f in files if "vocal" in os.path.basename(f).lower()]
    return sorted(set(files))


def clip_features(seg: np.ndarray, sr: int) -> dict:
    peak = float(np.max(np.abs(seg))) if seg.size else 0.0
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
    # energy-based activity: how much of the clip is above (peak - 35 dB)
    intervals = librosa.effects.split(seg, top_db=35)
    active = sum(e - s for s, e in intervals) / max(len(seg), 1)
    clipped = int(np.sum(np.abs(seg) >= 0.999))
    dc = float(np.mean(seg)) if seg.size else 0.0
    # crude music/video-game detector: spectral flatness of the quietest part.
    # Steady broadband hiss or a music bed keeps the floor "full"; a dry mic
    # between words drops to near-silence.
    hop = 512
    frames = librosa.util.frame(seg, frame_length=1024, hop_length=hop)
    if frames.shape[1] > 3:
        energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0) + 1e-12)
        quiet = energy <= np.percentile(energy, 25)
        floor = float(np.percentile(energy, 5))
    else:
        floor = rms
    return {
        "peak": peak, "rms": rms, "active": float(active),
        "clipped": clipped, "dc": dc, "floor": floor,
        "duration": len(seg) / sr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=[], help="source audio; not needed with --use-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.path.join(ROOT, "tools", "DDSP-SVC-6.3"))
    ap.add_argument("--sampling-rate", type=int, default=44100)
    ap.add_argument("--min-sec", type=float, default=2.5)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--db-thresh", type=float, default=-40.0, help="slicer silence threshold")
    ap.add_argument("--min-rms-dbfs", type=float, default=-45.0)
    ap.add_argument("--min-active", type=float, default=0.35, help="fraction of the clip above the energy gate")
    ap.add_argument("--max-dc", type=float, default=0.01)
    ap.add_argument("--highpass", type=float, default=60.0, help="Hz, removes rumble/DC; 0 disables")
    ap.add_argument("--val-clips", type=int, default=10)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-separated", default="", help="if set, read these vocals.wav files instead of the raw input")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from slicer import Slicer  # noqa: E402

    sr = args.sampling_rate
    slicer = Slicer(sr=sr, threshold=args.db_thresh, min_length=int(args.min_sec * 1000))

    if not args.input and not args.use_separated:
        ap.error("give --input or --use-separated")
    files = collect([args.use_separated] if args.use_separated else args.input,
                    vocals_only=bool(args.use_separated))
    if not files:
        print("no input audio found")
        return 1

    kept: list[np.ndarray] = []
    report: list[dict] = []
    rejects = {"too_quiet": 0, "too_sparse": 0, "dc": 0, "clipped": 0, "short": 0}

    for path in files:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        if args.highpass > 0:
            from scipy.signal import butter, sosfilt

            sos = butter(2, args.highpass / (sr / 2), btype="highpass", output="sos")
            audio = sosfilt(sos, audio).astype(np.float32)
        chunks = slicer.slice(audio)
        if "split_time" in chunks:
            chunks = {"0": chunks}

        file_kept = 0
        for _k, chunk in chunks.items():
            if chunk.get("slice"):
                continue
            a, b = (int(x) for x in chunk["split_time"].split(","))
            seg_full = audio[a:b]
            if len(seg_full) / sr < args.min_sec:
                rejects["short"] += 1
                continue
            limit = int(args.max_sec * sr)
            pieces = [seg_full[i:i + limit] for i in range(0, len(seg_full), limit)]
            for seg in pieces:
                if len(seg) / sr < args.min_sec:
                    rejects["short"] += 1
                    continue
                f = clip_features(seg, sr)
                if f["clipped"] > 4:
                    rejects["clipped"] += 1
                    continue
                if db(f["rms"]) < args.min_rms_dbfs:
                    rejects["too_quiet"] += 1
                    continue
                if f["active"] < args.min_active:
                    rejects["too_sparse"] += 1
                    continue
                if abs(f["dc"]) > args.max_dc:
                    rejects["dc"] += 1
                    continue
                kept.append(seg.astype(np.float32))
                report.append({"file": os.path.basename(path), "start": a / sr, "duration": f["duration"],
                               "rms_dbfs": db(f["rms"]), "active": f["active"], "floor_dbfs": db(f["floor"])})
                file_kept += 1
        print(f"  {os.path.basename(path):<44} {len(audio) / sr / 60:5.1f} 分钟 -> {file_kept:5d} 段", flush=True)

    if not kept:
        print("nothing passed QC")
        return 1

    total_min = sum(len(c) for c in kept) / sr / 60
    print(f"\n通过 QC: {len(kept)} 段 / {total_min:.1f} 分钟")
    print(f"被丢弃: {rejects}")

    floors = np.array([r["floor_dbfs"] for r in report])
    print(f"片段底噪分位: 5% {np.percentile(floors, 5):.1f} | 50% {np.percentile(floors, 50):.1f} "
          f"| 95% {np.percentile(floors, 95):.1f} dBFS   (越高越可能有持续 BGM)")

    random.seed(args.seed)
    random.shuffle(kept)
    if args.max_clips and len(kept) > args.max_clips:
        kept = kept[:args.max_clips]
    val_n = min(args.val_clips, max(len(kept) // 20, 1))
    val, train = kept[:val_n], kept[val_n:]

    for split, items in (("train", train), ("val", val)):
        out_dir = os.path.join(args.out, split, "audio")
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "*.wav")):
            os.remove(old)
        for i, clip in enumerate(items):
            sf.write(os.path.join(out_dir, f"{split}_{i:05d}.wav"), clip, sr, subtype="PCM_16")
        print(f"  wrote {len(items):5d} clips -> {out_dir}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "extract_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"kept": len(kept), "minutes": total_min, "rejects": rejects,
                   "clips": report}, fh, ensure_ascii=False, indent=1)
    print(f"\n报告: {os.path.join(args.out, 'extract_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
