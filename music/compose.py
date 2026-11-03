"""FLEXKIN Private Spa — original ambient piece, rendered offline to WAV.

Key of D-flat major, 60 BPM, 16 chords x 12 s = 192 s, built to loop seamlessly
(the reverb and release tails past the end are wrapped back onto the start).

Layers
  pad      warm additive pad, 3 detuned voices per note, slowly breathing brightness
  halo     very quiet octave-up layer that blooms in the middle of each chord
  bass     soft sine + 2nd harmonic on the root
  piano    felt-piano model (inharmonic partials, two-stage decay, 3 detuned strings),
           sparse broken chords, never a lead melody
  air      barely-there filtered noise for room tone
  space    stereo convolution hall (~5.5 s), frequency-dependent decay
"""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile
import sys

SR = 44100
rng = np.random.default_rng(20261103)
CHORD = 12.0                       # seconds per chord
TAIL = 14.0                        # rendered past the end, then wrapped
OUT = sys.argv[1] if len(sys.argv) > 1 else 'flexkin-spa.wav'

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)

# (bass, pad voicing, piano pool) — voicings move by step so changes feel like breathing
A = [
    (37, [56, 60, 63, 65], [68, 72, 75, 77, 80]),   # Dbmaj9
    (34, [56, 60, 61, 65], [68, 72, 73, 77]),       # Bbm9
    (42, [58, 60, 61, 65], [70, 72, 73, 77]),       # Gbmaj9(#11)
    (44, [56, 58, 61, 63], [68, 70, 73, 75]),       # Ab sus4(add9)
    (41, [56, 60, 63, 65], [68, 72, 75, 77]),       # Db/F
    (42, [58, 61, 65, 66], [70, 73, 77, 78]),       # Gbmaj7
    (39, [58, 61, 63, 68], [70, 73, 75, 80]),       # Ebm11
    (44, [56, 58, 61, 63], [68, 70, 73, 75]),       # Ab sus4
]
B = [
    (37, [56, 60, 63, 65], [72, 75, 77, 80]),       # Dbmaj9
    (41, [56, 60, 63, 65], [68, 72, 75, 77]),       # Fm7
    (42, [58, 61, 65, 68], [70, 73, 77, 80]),       # Gbmaj9
    (44, [56, 60, 63, 65], [68, 72, 75]),           # Db/Ab
    (34, [56, 60, 61, 65], [68, 72, 73, 77]),       # Bbm9
    (39, [58, 61, 65, 66], [70, 73, 77, 78]),       # Ebm9
    (42, [58, 60, 61, 65], [70, 72, 73, 77]),       # Gbmaj7#11
    (44, [56, 58, 61, 63], [68, 70, 73, 75]),       # Ab sus4 -> back to Db
]
PROG = A + B
LEN = CHORD * len(PROG)
N = int((LEN + TAIL) * SR)

padL = np.zeros(N); padR = np.zeros(N)
pnoL = np.zeros(N); pnoR = np.zeros(N)
bass = np.zeros(N)

def env_adsr(n, a, r, total):
    t = np.arange(n) / SR
    e = np.ones(n)
    e = np.minimum(e, t / a)
    rel_start = total - r
    e = np.where(t > rel_start, e * np.clip(1 - (t - rel_start) / r, 0, 1), e)
    return np.sin(np.clip(e, 0, 1) * np.pi / 2) ** 2      # smooth S-curve

def add(buf, start, sig):
    i = int(start * SR); j = min(N, i + len(sig))
    buf[i:j] += sig[:j - i]

# ---------- pad ----------
def pad_note(m, dur, bright_phase, amp=1.0, harmonics=12, voices=3):
    f = mtof(m)
    n = int(dur * SR); t = np.arange(n) / SR
    # brightness breathes on a ~24 s cycle
    bright = 0.55 + 0.25 * np.sin(2 * np.pi * (t / 24.0) + bright_phase)
    out = np.zeros(n)
    for v in range(voices):
        cents = (v - (voices - 1) / 2) * 5.0 + rng.normal(0, .8)
        fv = f * 2 ** (cents / 1200)
        vib = 1 + 0.0012 * np.sin(2 * np.pi * (0.13 + .05 * v) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(fv * vib) / SR
        for k in range(1, harmonics + 1):
            if fv * k > 7000: break
            a = (1 / k ** 1.7) * np.exp(-(k - 1) * (1.15 - bright) * 0.9)
            out += a * np.sin(k * ph + rng.uniform(0, 6.28))
    # slow per-note swell so the chord never sits perfectly still
    sw = 0.85 + 0.15 * np.sin(2 * np.pi * t / rng.uniform(9, 15) + rng.uniform(0, 6.28))
    return out * sw * amp / voices

for ci, (root, voicing, _) in enumerate(PROG):
    start = ci * CHORD - 1.5 if ci else 0.0          # each chord fades in under the last
    dur = CHORD + 6.5
    for vi, m in enumerate(voicing):
        sig = pad_note(m, dur, bright_phase=ci * .7 + vi) * env_adsr(int(dur * SR), 4.5, 6.0, dur)
        pan = 0.5 + (vi - 1.5) * 0.18                # spread voices across the field
        add(padL, start, sig * np.cos(pan * np.pi / 2) * .16)
        add(padR, start, sig * np.sin(pan * np.pi / 2) * .16)
    # halo: octave-up bloom in the middle of the chord, very quiet, wide
    for vi, m in enumerate(voicing[1:]):
        hd = 9.0; hs = ci * CHORD + 2.5
        sig = pad_note(m + 12, hd, ci + vi, harmonics=3, voices=2) * env_adsr(int(hd * SR), 4.0, 4.5, hd)
        pan = 0.15 if vi % 2 == 0 else 0.85
        add(padL, hs, sig * np.cos(pan * np.pi / 2) * .045)
        add(padR, hs, sig * np.sin(pan * np.pi / 2) * .045)
    # bass: soft fundamental + a touch of 2nd harmonic
    dur_b = CHORD + 5.0
    n = int(dur_b * SR); t = np.arange(n) / SR
    f = mtof(root)
    b = np.sin(2 * np.pi * f * t) + .25 * np.sin(2 * np.pi * 2 * f * t + .3) + .08 * np.sin(2 * np.pi * 3 * f * t)
    add(bass, start, b * env_adsr(n, 3.5, 5.0, dur_b) * .11)

# ---------- felt piano ----------
def piano_note(m, vel, length=9.0):
    f = mtof(m)
    n = int(length * SR); t = np.arange(n) / SR
    Bc = 0.00035 * (f / 261.6) ** .5                 # string stiffness (inharmonicity)
    base = 7.5 * (261.6 / f) ** .55                  # lower notes ring longer
    out = np.zeros(n)
    for k in range(1, 15):
        fk = k * f * np.sqrt(1 + Bc * k * k)
        if fk > 9000: break
        a = (1 / k ** 1.25) * np.exp(-(k - 1) * (0.62 - .35 * vel))   # felt: soft highs
        t2 = base / (1 + .22 * (k - 1)); t1 = t2 / 7
        e = .62 * np.exp(-t / t1) + .38 * np.exp(-t / t2)
        for d in (-.7, 0, .8):                       # three strings, slightly apart -> gentle beating
            out += a * e * np.sin(2 * np.pi * fk * 2 ** (d / 1200) * t + rng.uniform(0, 6.28)) / 3
    att = np.minimum(1, t / .006)
    # felt hammer: a soft, low thump
    h = rng.normal(0, 1, int(.03 * SR)); h = sosfilt(butter(2, 900, 'low', fs=SR, output='sos'), h)
    h *= np.exp(-np.arange(len(h)) / (.006 * SR))
    out[:len(h)] += h * .05
    fade = np.ones(n); fr = int(1.0 * SR); fade[-fr:] = np.linspace(1, 0, fr)
    return out * att * fade * vel

# sparse patterns (beat offsets inside a 12-beat chord); index into the chord's pool
PATTERNS = [
    [(0.6, 0), (3.1, 2), (6.4, 1)],
    [(0.4, 0), (2.6, 1), (4.9, 3), (9.2, 2)],
    [(1.2, 1), (5.0, 2)],
    [(0.5, 0), (3.8, 2), (7.5, 3)],
    [(0.8, 1), (2.9, 0), (6.9, 2), (10.1, 1)],
    [(1.0, 0), (6.0, 2)],
]
for ci, (root, voicing, pool) in enumerate(PROG):
    pat = PATTERNS[(ci * 5 + ci // 4) % len(PATTERNS)]
    # a low anchor note on every other chord, very soft
    if ci % 2 == 0:
        m = root + 24 if root + 24 < 64 else root + 12
        add_t = ci * CHORD + .15 + rng.normal(0, .03)
        s = piano_note(m, .26, 10.0); pan = .42
        add(pnoL, add_t, s * np.cos(pan * np.pi / 2)); add(pnoR, add_t, s * np.sin(pan * np.pi / 2))
    for (beat, idx) in pat:
        m = pool[idx % len(pool)]
        tt = ci * CHORD + beat + rng.normal(0, .04)
        vel = float(np.clip(rng.normal(.34, .05), .24, .44))
        s = piano_note(m, vel)
        pan = .35 + (m - 66) / 40                     # higher notes drift slightly right
        add(pnoL, tt, s * np.cos(pan * np.pi / 2)); add(pnoR, tt, s * np.sin(pan * np.pi / 2))

# ---------- air (room tone) ----------
noise = rng.normal(0, 1, (2, N))
sos = butter(2, [180, 1400], 'band', fs=SR, output='sos')
air = np.stack([sosfilt(sos, noise[c]) for c in range(2)])
tt = np.arange(N) / SR
air *= (0.6 + 0.4 * np.sin(2 * np.pi * tt / 31.0))[None] * .0045

# ---------- hall reverb ----------
def make_ir(seconds=6.0, seed=0):
    r = np.random.default_rng(seed)
    n = int(seconds * SR); t = np.arange(n) / SR
    bands = [((None, 400), 1.05), ((400, 2500), .85), ((2500, None), .42)]
    ir = np.zeros(n)
    for (lo, hi), tau in bands:
        x = r.normal(0, 1, n)
        if lo is None: so = butter(2, hi, 'low', fs=SR, output='sos')
        elif hi is None: so = butter(2, lo, 'high', fs=SR, output='sos')
        else: so = butter(2, [lo, hi], 'band', fs=SR, output='sos')
        ir += sosfilt(so, x) * np.exp(-t / tau)
    ir *= np.minimum(1, t / .02)                     # soft onset
    pre = np.zeros(int(.028 * SR))                   # pre-delay
    ir = np.concatenate([pre, ir])
    return ir / np.sqrt(np.sum(ir ** 2))

irL, irR = make_ir(seed=1), make_ir(seed=2)
sendL = padL * .75 + pnoL * .55 + bass * .1
sendR = padR * .75 + pnoR * .55 + bass * .1
wetL = fftconvolve(sendL, irL)[:N]; wetR = fftconvolve(sendR, irR)[:N]

dryL = padL * .55 + pnoL * .5 + bass + air[0]
dryR = padR * .55 + pnoR * .5 + bass + air[1]
L = dryL + wetL * .9
R = dryR + wetR * .9

# ---------- wrap the tail onto the start so the file loops without a seam ----------
M = int(LEN * SR)
L[:N - M] += L[M:]; R[:N - M] += R[M:]
L, R = L[:M], R[:M]

# ---------- master: gentle tone shaping, soft saturation, level ----------
hp = butter(2, 32, 'high', fs=SR, output='sos')
shelf = butter(1, 7500, 'low', fs=SR, output='sos')
mix = np.stack([sosfilt(hp, L), sosfilt(hp, R)])
mix = .7 * mix + .3 * np.stack([sosfilt(shelf, mix[0]), sosfilt(shelf, mix[1])])
rms = np.sqrt(np.mean(mix ** 2))
mix *= (10 ** (-19 / 20)) / rms                      # about -19 dBFS RMS
mix = np.tanh(mix * 1.1) / 1.1
peak = np.max(np.abs(mix))
if peak > .89: mix *= .89 / peak
print(f'length {M / SR:.1f}s  rms {20 * np.log10(np.sqrt(np.mean(mix ** 2))):.1f} dBFS  peak {20 * np.log10(np.max(np.abs(mix))):.1f} dBFS')
wavfile.write(OUT, SR, (mix.T * 32767).astype(np.int16))
