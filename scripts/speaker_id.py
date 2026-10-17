"""Measure speaker identity with a d-vector embedding, not spectrum.

The earlier "timbre" metric compared mean log-mel envelopes, which tracks
frequency balance rather than who is speaking. It reported +7.9 dB of movement
toward the target while the user's ears said "does not sound like her at all" -
so this exists to measure the thing that actually matters.

Reference points printed alongside every number:
  * target-internal  - two halves of her own clips: the noise floor
  * target-vs-source - how far apart the two voices are to begin with

  python scripts/speaker_id.py --target-dir work/dataset-sayue/train/audio \
      --audio work/songs/sep/original/vocals.wav output/sayue-original-vocal.wav
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

ENCODER = VoiceEncoder("cpu")


def embed(path: str, max_seconds: float = 30.0) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:  # noqa: BLE001
        return None
    if len(audio) < sr // 2:
        return None
    mono = audio.mean(axis=1)[: int(max_seconds * sr)]
    wav = preprocess_wav(mono, sr)
    if len(wav) < 8000:
        return None
    return ENCODER.embed_utterance(wav)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True, help="folder of the target speaker's clips")
    ap.add_argument("--audio", nargs="+", required=True, help="files to score against the target")
    ap.add_argument("--max-target", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.target_dir, "*.wav")))
    if not clips:
        print(f"no clips in {args.target_dir}")
        return 1
    random.seed(args.seed)
    clips = random.sample(clips, min(args.max_target, len(clips)))

    embs = [e for e in (embed(c) for c in clips) if e is not None]
    if not embs:
        print("could not embed any target clip")
        return 1
    mat = np.stack(embs)
    centroid = mat.mean(axis=0)

    half = len(mat) // 2
    floor = cos(mat[:half].mean(axis=0), mat[half:].mean(axis=0)) if half >= 2 else float("nan")

    print(f"  目标音色: {len(embs)} 段   目标内部一致性(噪声底) {floor:.4f}")
    print(f"  {'音频':<44} {'与目标相似度':>12}  {'判定':<12}")
    results = []
    for path in args.audio:
        e = embed(path)
        if e is None:
            print(f"  {os.path.basename(path):<44} {'读取失败':>12}")
            continue
        s = cos(e, centroid)
        results.append((path, s))
        # 0.75+ 通常被认为是同一说话人（resemblyzer 余弦的经验区间）
        verdict = ("像本人" if s >= 0.80 else
                   "部分像" if s >= 0.70 else
                   "不像" if s >= 0.55 else "完全不像")
        print(f"  {os.path.basename(path):<44} {s:12.4f}  {verdict:<12}")

    if len(results) >= 2:
        print("\n  参考：两两之间的相似度")
        embs2 = [(p, embed(p)) for p, _ in results]
        embs2 = [(p, e) for p, e in embs2 if e is not None]
        for i in range(len(embs2)):
            for j in range(i + 1, len(embs2)):
                print(f"    {os.path.basename(embs2[i][0])[:24]:<24} vs "
                      f"{os.path.basename(embs2[j][0])[:24]:<24} {cos(embs2[i][1], embs2[j][1]):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
