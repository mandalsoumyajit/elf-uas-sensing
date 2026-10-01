"""Motor fundamental (ELF, wandering, needs tracking) vs ESC PWM carrier (VLF/LF, clock-stable) for curtain detection,
assuming a 100 kHz receiver bandwidth.

Per class: strongest fundamental-band line (phase leads at f_e) and strongest PWM-frequency line (ESC switching
loop / phase-lead ripple / DC-bus PWM, combined in quadrature since they share the carrier frequency).
Processors (curtain, worst point, K = 4 nearest nodes, model-free coherent combining, Pd 0.9, Pfa 1e-3):
  fund-incoh   : incoherent energy on the fundamental (tau = 0.05 s chunks), all nodes
  fund-joint   : fundamental with joint array tracking, threshold 0 dB in 20 Hz (optimistic tracker)
  pwm-coherent : PWM carrier integrated coherently over the dwell T with no tracking; search over a +-1 kHz
                 carrier uncertainty (RC-oscillator ESC clocks) -> 2000*T frequency cells in the Pfa budget
Dwell T = 2 s (curtain crossing) and 30 s (loitering / hover near an asset).
Noise: scenario ASDs from range_by_class (floored at 20-30 fT/rtHz above 10 kHz; UNMEASURED above 10 kHz).
"""
import copy, json, os, sys
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
from range_budget import wire_loop_noise
import drone_source as DS
src = open(os.path.join(HERE, 'range_by_class.py')).read().split("SCEN =")[0]
ns = {'__file__': os.path.join(HERE, 'range_by_class.py')}; exec(src, ns)
SC = {'quiet outdoor': ns['natural'], 'semi-urban outdoor': ns['semi_urban'], 'indoor lab': ns['indoor_lab']}

C = DS.CLASSES
fpv48 = copy.deepcopy(C['5-inch FPV (6S)']); fpv48['f_pwm'] = 48e3
dji = copy.deepcopy(C['photo/mid (11" props, 4S)']); dji['f_pwm'] = 85e3     # DJI-like: 77-93 kHz PWM reported
VAR = {'5-inch FPV, 24 kHz PWM': C['5-inch FPV (6S)'], '5-inch FPV, 48 kHz PWM': fpv48,
       'photo 11", 20 kHz PWM': C['photo/mid (11" props, 4S)'], 'photo 11", DJI-like 85 kHz PWM': dji,
       'heavy-lift, 16 kHz PWM': C['heavy-lift (30" props, 12S)']}
USE = {'5-inch FPV, 24 kHz PWM': ['indoor lab', 'semi-urban outdoor'], '5-inch FPV, 48 kHz PWM': ['indoor lab', 'semi-urban outdoor'],
       'photo 11", 20 kHz PWM': ['semi-urban outdoor', 'quiet outdoor'], 'photo 11", DJI-like 85 kHz PWM': ['semi-urban outdoor', 'quiet outdoor'],
       'heavy-lift, 16 kHz PWM': ['semi-urban outdoor', 'quiet outdoor']}
TAU, BL, K, PFA0 = 0.05, 20.0, 4, 1e-3
Kk = np.arange(-6, 7) + 0.5

def node_s(p, m, n):
    X, Y = np.meshgrid(Kk * p, Kk * p); r = np.sqrt(X ** 2 + Y ** 2).ravel()
    return np.sort((1e-7 * m / r ** 3) ** 2 * (2 / 3) / (2 * n ** 2))[::-1]

def pd(dof, lam, pfa):
    return ncx2.sf(chi2.isf(pfa, dof), dof, lam)

def det(proc, p, m, n, T):
    s = node_s(p, m, n)
    if proc == 'pwm-coherent':
        return pd(2 * K, 2 * T * s[:K].sum(), PFA0 / (2000 * T))
    pfa = PFA0 / (500 * TAU); M = int(T / TAU)
    inc = pd(2 * len(s) * M, 2 * M * np.sum(s * TAU), pfa)
    if proc == 'fund-incoh' or s[:K].sum() / BL < 1.0:
        return inc
    return pd(2 * K, 2 * T * s[:K].sum(), pfa)

def pitch(proc, m, n, T):
    f = lambda p: det(proc, p, m, n, T) - 0.9
    ps = np.linspace(0.05, 300, 6000); v = np.array([f(x) for x in ps]); i = np.argmax(v < 0)
    return brentq(f, ps[i - 1], ps[i]) if i > 0 else 0.0

rows = []
print('Curtain pitch (m), scalar nodes, nominal moments [PWM moment low-high range for pwm-coherent]; T = 2 s / 30 s')
print(f'{"drone":32s} {"background":20s} {"f_e line":>14s} {"PWM line":>16s} | {"fund-incoh":>11s} {"fund-joint":>11s} {"pwm-coherent":>22s}')
for name, c in VAR.items():
    op, L = DS.lines_for(c)
    fund = max([x for x in L if x[1] == 'phase leads' and abs(x[0] - op['f_e']) < 1], key=lambda x: x[2][1])
    pw = [x for x in L if abs(x[0] - c['f_pwm']) < 1]
    m_pwm = tuple(np.sqrt(sum(x[2][i] ** 2 for x in pw)) for i in range(3))
    for bg in USE[name]:
        nfund = float(np.hypot(SC[bg](fund[0]), wire_loop_noise(fund[0])[0]))
        npwm = float(np.hypot(SC[bg](c['f_pwm']), wire_loop_noise(c['f_pwm'])[0]))
        cells = []
        for T in (2.0, 30.0):
            a = pitch('fund-incoh', fund[2][1], nfund, T); b = pitch('fund-joint', fund[2][1], nfund, T)
            pp = [pitch('pwm-coherent', m_pwm[i], npwm, T) for i in range(3)]
            cells.append((a, b, pp))
            rows.append(dict(drone=name, bg=bg, T=T, fund_incoh=a, fund_joint=b, pwm=pp, f_e=op['f_e'], f_pwm=c['f_pwm'],
                             m_fund=fund[2][1], m_pwm=m_pwm, n_fund=nfund, n_pwm=npwm))
        print(f'{name:32s} {bg:20s} {op["f_e"]:6.0f} Hz {fund[2][1]:.0e} {c["f_pwm"]/1e3:5.0f}kHz {m_pwm[1]:.0e} | '
              f'{cells[0][0]:4.1f}/{cells[1][0]:4.1f}  {cells[0][1]:4.1f}/{cells[1][1]:4.1f}  '
              f'{cells[0][2][1]:4.1f}/{cells[1][2][1]:4.1f} [{cells[0][2][0]:.1f}-{cells[0][2][2]:.1f}]')
json.dump(rows, open(os.path.join(HERE, 'vlf_lines.json'), 'w'), indent=1, default=float)
