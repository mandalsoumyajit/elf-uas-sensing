"""What does coherent processing across nodes buy? Curtain-crossing detection with four array processors.

Geometry: vertical square lattice (curtain) of pitch p; drone crosses at the worst point (cell centre).
Signal: one motor line, per-node amplitude from the orientation-averaged dipole field (scalar loop: per-axis
2/3 of |B|^2; triaxial node: full |B|^2, i.e. x3). Frequency wander means:
  * untracked coherence time tau = 0.05 s (lines wander 10-30 Hz, measured);
  * a tracker needs SNR >= gamma_th in its loop bandwidth B_L = 20 Hz (per 1/B_L = 0.05 s).
Dwell T = 2 s (crossing at ~5 m/s). Pd 0.9, Pfa 1e-3 per decision, divided by 500 Hz x tau frequency cells.
Processors:
  P1 incoherent array  : sum over nodes and T/tau chunks of |X|^2 (no tracking).
  P2 self-tracked      : the strongest node must track by itself (s_max >= gamma_th); then coherent over T,
                         model-free spatial combining of the K nearest nodes (GLRT, 2K dof).
  P3 joint-tracked     : tracking on the coherently combined array signal (sum of node SNRs >= gamma_th);
                         then as P2.
  P4 joint + matched   : as P3 but spatially matched to a position/orientation hypothesis (2 dof),
                         Pfa further divided by 100 position/orientation cells (search penalty).
Untracked fall-back: if a tracked processor cannot track, it reverts to P1.
"""
import json, os, sys
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
from range_budget import wire_loop_noise
import importlib.util
spec = importlib.util.spec_from_file_location('rbc', os.path.join(HERE, 'range_by_class.py'))
SCEN = {'quiet outdoor': None}
src = open(os.path.join(HERE, 'range_by_class.py')).read().split("SCEN =")[0]
ns = {'__file__': os.path.join(HERE, 'range_by_class.py')}; exec(src, ns)                                     # scenario functions only (no main run)
SC = {'quiet outdoor': ns['natural'], 'semi-urban outdoor': ns['semi_urban'], 'indoor lab': ns['indoor_lab']}

TAU, BL, PFA0, K = 0.05, 20.0, 1e-3, 4
T = float(os.environ.get("DWELL_T", "2.0"))
GTH = [float(x) for x in os.environ.get("GTH_DB", "6,10").split(",")]
PFA = PFA0 / (500 * TAU)
Kk = np.arange(-6, 7) + 0.5

def node_snr_per_s(p, m, n, triax):
    X, Y = np.meshgrid(Kk * p, Kk * p)
    r = np.sqrt(X ** 2 + Y ** 2).ravel()
    b2 = (1e-7 * m / r ** 3) ** 2 * (2.0 if triax else 2.0 / 3.0)   # mean-square per-axis peak amplitude^2 sum
    return np.sort(b2 / (2 * n ** 2))[::-1]          # SNR per second of coherent integration, strongest first

def pd(dof, lam, pfa):
    return ncx2.sf(chi2.isf(pfa, dof), dof, lam)

def detect(proc, p, m, n, triax, gth_db):
    s = node_snr_per_s(p, m, n, triax)               # per-second coherent SNR per node
    gth = 10 ** (gth_db / 10)
    def incoherent():
        M = int(T / TAU); N = len(s)
        return pd(2 * N * M, 2 * M * np.sum(s * TAU), PFA)
    if proc == 'P1':
        return incoherent()
    can_track = (s[0] / BL >= gth) if proc == 'P2' else (np.sum(s[:K]) / BL >= gth)
    if not can_track:
        return incoherent()
    if proc in ('P2', 'P3'):
        return pd(2 * K, 2 * T * np.sum(s[:K]), PFA)
    return pd(2, 2 * T * np.sum(s), PFA / 100)

def max_pitch(proc, m, n, triax, gth_db):
    f = lambda p: detect(proc, p, m, n, triax, gth_db) - 0.9
    if f(0.05) < 0:
        return 0.0
    ps = np.linspace(0.05, 200, 4000); v = np.array([f(x) for x in ps])
    i = np.argmax(v < 0)                              # first failure (Pd non-monotone near tracking threshold)
    return brentq(f, ps[i - 1], ps[i]) if i > 0 else ps[-1]

R = json.load(open(os.path.join(HERE, 'drone_source_results.json')))
def line(cls, band):
    flo, fhi = band
    Ls = [L for L in R[cls]['lines'] if flo <= L['f'] <= fhi and not L['source'].startswith('rotor')]
    return max(Ls, key=lambda L: L['m'][1] / max(1.0, L['f'] / 1e4))   # strong line, mild preference for low f
CASES = [('indoor 5-inch, <=10 kHz', '5-inch FPV (6S)', 'indoor lab', (100, 1e4)),
         ('urban photo 11", <=10 kHz', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', (100, 1e4)),
         ('urban heavy-lift, <=10 kHz', 'heavy-lift (30" props, 12S)', 'semi-urban outdoor', (100, 1e4)),
         ('remote photo 11", <=10 kHz', 'photo/mid (11" props, 4S)', 'quiet outdoor', (100, 1e4))]
out = []
print(f'Max curtain pitch (m) for Pd 0.9 at a cell-centre crossing; dwell T = {T} s; tracking threshold 6 / 10 dB in {BL:.0f} Hz')
print(f'{"case":28s} {"node":9s} {"line":>16s} | {"P1 incoh":>9s} {"P2 self-track":>14s} {"P3 joint-track":>15s} {"P4 joint+matched":>17s}')
for lab, cls, bg, band in CASES:
    L = line(cls, band); m = L['m'][1]; n = float(np.hypot(SC[bg](L['f']), wire_loop_noise(L['f'])[0]))
    for triax in (False, True):
        cells = []
        for proc in ('P1', 'P2', 'P3', 'P4'):
            a = max_pitch(proc, m, n, triax, GTH[0]); b = max_pitch(proc, m, n, triax, GTH[1])
            cells.append(f'{a:5.1f}/{b:5.1f}')
            out.append(dict(case=lab, triax=triax, proc=proc, pitch_6dB=a, pitch_10dB=b, f=L['f'], src=L['source']))
        print(f'{lab:28s} {"triaxial" if triax else "scalar":9s} {L["source"][:10]+" "+str(int(L["f"])):>16s} | '
              f'{cells[0]:>9s} {cells[1]:>14s} {cells[2]:>15s} {cells[3]:>17s}')
json.dump(out, open(os.path.join(HERE, f'array_processors_T{T:g}_g{GTH[0]:g}.json'), 'w'), indent=1)
