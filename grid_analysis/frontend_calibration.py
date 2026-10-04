"""Field calibration of the grid recordings from the documented analog front end (2026-10-04).

Chain (IEEE Sensors paper, Herath et al., Fig. 7; Phase-1 report Fig. 3.13; Phase-2 report bias network):
  loop (300 turns, 1 m, EMF = j w N A B)
  -> differential OPA227 LNA, Av1 = R2/R1 = 100 with Rin = 2 R1 = 2 kOhm, which with the loop impedance gives
     H1 = Av1 / (1 + R/(2 R1)) / (1 + j w tau1),  tau1 = L / (2 R1 + R)
  -> VGA (jumpers), 101 V/V for the reported overall gain of 1e4
  -> LM741 Sallen-Key low-pass, R = 43 k, C = 1 nF, K = 1 + 56k/100k = 1.56 (f0 3.70 kHz, Q 0.69)
  -> Phase-2 bias / level-shift network: x0.33 (text) or x0.25 (schematic), first-order high-pass 160-340 Hz
  -> 12-bit ADC, 3.3 V full scale.
Uncertain parameters are enumerated (loop L and R, bias ratio, high-pass corner) and the spread is reported.

Outputs: |H_B| (counts/T) vs frequency against the earlier flat-gain assumption; inter-harmonic floors in
fT/rtHz; field of the tracked motor line at the nearest antenna, compared with the dipole field of the
source-model 5-inch motor.
"""
import itertools, os, sys
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else None
N, AREA = 300, np.pi * 0.5 ** 2
CNT = 4096 / 3.3
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
dist = lambda c, ch: float(np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]))

def H_B(f, L, R, bias, fhp, vga=101.0):
    w = 2 * np.pi * f; s = 1j * w
    lna = 100.0 / (1 + R / 2000.0) / (1 + s * L / (2000.0 + R))
    w0 = 1 / (43e3 * 1e-9); K = 1.56; Q = 1 / (3 - K)
    lpf = K / (1 - (w / w0) ** 2 + 1j * w / (Q * w0))
    hp = (1j * f / fhp) / (1 + 1j * f / fhp)
    return CNT * bias * hp * lpf * vga * lna * s * N * AREA

GRID = list(itertools.product((0.226, 0.407), (126.0, 159.0), (0.25, 0.33), (160.0, 340.0)))
H_old = lambda f: CNT / 3 * 1000.0 * 2 * np.pi * f * N * AREA          # earlier assumption (env_report.py)

def spread(f):
    v = np.array([np.abs(H_B(f, *p)) for p in GRID]); return v.min(), np.median(v), v.max()

if __name__ == '__main__':
    print('|H_B| relative to the earlier flat-gain assumption (min / median / max over parameter grid):')
    for f in (200, 600, 1200, 3000, 6000, 8000):
        lo, md, hi = spread(f)
        print(f'  {f:5d} Hz: x{lo / H_old(f):5.2f} / x{md / H_old(f):5.2f} / x{hi / H_old(f):5.2f}')
    f1 = [(2000 + R) / (2 * np.pi * L) for L, R, *_ in GRID]
    print(f'LNA pole (loop L with 2 kOhm input): {min(f1):.0f}-{max(f1):.0f} Hz')

    # inter-harmonic floors (same selection as env_report.py)
    P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']
    harm = np.zeros_like(f, bool)
    for k in range(1, 200):
        harm |= np.abs(f - 50.0 * k) <= 5
    IH = ~harm & (np.abs(f - 5930) > 30)
    print('\nInter-harmonic floor, fT/rtHz: new chain [min-max over grid] (earlier conditional value)')
    for fr, (lo, hi) in ((600, (200, 900)), (3000, (1500, 4500)), (8000, (7000, 9500))):
        row = []
        for ch in 'ABCD':
            far = [c for c in range(1, 26) if dist(c, ch) >= 4.0]
            bg = np.median(np.stack([PP[PN.index(f'c{c:02d}_{ch}.mat')] for c in far]), 0)
            a = np.sqrt(np.median(bg[IH & (f > lo) & (f < hi)]))
            s_lo, _, s_hi = spread(fr)
            row.append(f'{ch} {a / s_hi * 1e15:6.1f}-{a / s_lo * 1e15:6.1f} ({a / H_old(fr) * 1e15:6.1f})')
        print(f'  ~{fr:4d} Hz: ' + ';  '.join(row))

    # motor-line field at the antenna nearest the drone vs the dipole field of one source-model motor
    A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
    print('\nTracked motor line (one motor), nearest antenna: field amplitude [grid min-max] vs dipole prediction')
    ratios = []
    for c in sorted(A):
        rec = A[c]['rec']; fc = float(A[c]['fc'])
        k = int(np.argmax(np.abs(rec))); ch = 'ABCD'[k]; r = dist(c, ch)
        lo, md, hi = spread(fc)
        B_md = 2 * np.abs(rec[k]) / md                                    # peak field, median calibration
        # source model: 5-inch phase-lead moment 1.06e-3 A m^2 nominal (x5 either way); dipole field 1-2 x mu0 m/(4 pi r^3)
        Bp = 1e-7 * 1.06e-3 / r ** 3
        ratios.append(B_md / Bp)
        if c in (1, 5, 7, 13, 21, 25):
            print(f'  cell {c:2d} ant {ch} r {r:4.2f} m f {fc:6.1f} Hz: B = {2*np.abs(rec[k])/hi*1e12:7.1f}-{2*np.abs(rec[k])/lo*1e12:7.1f} pT; '
                  f'dipole (axial-equatorial, nominal) {Bp*1e12:6.1f}-{2*Bp*1e12:6.1f} pT')
    ratios = np.array(ratios)
    print(f'  measured/predicted (equatorial, nominal moment) over cells: median {np.median(ratios):.2f}, IQR {np.percentile(ratios,25):.2f}-{np.percentile(ratios,75):.2f}')
    if EXP:
        x = sio.loadmat(os.path.join(EXP, 'c01_A.mat'))['x'].ravel()
        print(f'\nraw sample check c01_A: dtype {x.dtype}, min {x.min()}, max {x.max()}, unique {len(np.unique(x))}')
