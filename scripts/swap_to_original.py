"""把指定区间换回原唱（模型在间奏上会凭空生成噪声）。

RVC 的输入如果是静音，模型仍会输出 -50 dBFS 左右的噪声；间奏段落因此很脏。
这段的处理是：原唱该段有实际内容时按电平对齐后替换，是静音时直接替换，
不做任何增益（否则会把底噪放大 40 多 dB）。
"""
import sys

import librosa
import numpy as np
import soundfile as sf

CONV = sys.argv[1]
ORIG = sys.argv[2]
OUT = sys.argv[3]
A, B = float(sys.argv[4]), float(sys.argv[5])     # 要替换的区间（秒）
PAD = 2.0                                          # 两侧留的余量
FADE = 1.0

conv, sr = sf.read(CONV, always_2d=True, dtype="float32")
orig, sr2 = sf.read(ORIG, always_2d=True, dtype="float32")
om = orig.mean(axis=1)
if sr2 != sr:
    om = librosa.resample(om, orig_sr=sr2, target_sr=sr)

lo, hi = A - PAD, B + PAD
seg_a, seg_b = int(lo * sr), int(hi * sr)
a = float(np.sqrt(np.mean(conv[seg_a:seg_b] ** 2)))
b = float(np.sqrt(np.mean(om[seg_a:seg_b] ** 2)))

gain = 1.0
if b > 1e-5 and (a / max(b, 1e-9)) < 100:          # 原唱真的有内容才对齐
    gain = a / b
status = "静音，不增益" if gain == 1.0 else "有内容，增益 %+.1f dB" % (20 * np.log10(gain))
print("  原唱该段 RMS %.1f dBFS -> %s" % (20 * np.log10(max(b, 1e-12)), status))

om = om * gain
out = conv.copy()
off = int(lo * sr)
n = min(len(om), len(conv)) - off
t = (np.arange(n) / sr) + lo
fi = np.clip((t - (A - FADE)) / FADE, 0, 1)
fo = 1 - np.clip((t - B) / FADE, 0, 1)
w = np.minimum(fi, fo).astype(np.float32)[:, None]
out[off:off + n] = conv[off:off + n] * (1 - w) + om[off:off + n, None] * w

sf.write(OUT, out, sr, subtype="PCM_24")
print("  已替换 %.1f-%.1f 秒（各 %.0f 秒淡入淡出）" % (A, B, FADE))
