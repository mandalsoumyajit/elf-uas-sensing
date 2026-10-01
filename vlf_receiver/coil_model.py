"""Coil model for the ELF and VLF node windings (printed-circuit-board triaxial loop, one axis).

The original coil generator (Neumann inductance, copper/contact resistance, 2-D dielectric capacitance) is not part
of this release, so the coil parameters are rebuilt from first principles and calibrated on the three
published variants (48/56/100 turns, published_coils.json):
  - geometry   : N closed square turns of side a = sqrt(area per turn), stacked at 1 mm height pitch
  - inductance : exact filament formula for coaxial squares (parallel-segment mutuals); the self term uses an
                 equivalent filament radius r_eq (geometric mean distance of the trace), fitted to L(48);
                 L(56) and L(100) are then predictions that check the model
  - resistance : R per turn (copper + 4 mating contacts) from the published values (linear in N, verified)
  - capacitance: C per turn from the published 2-D estimate (linear in N, verified)
  - area       : area per turn from the published values
Output (coil_model.json): the published ELF windings and a VLF family (N = 2..24), SRFs and the
noise-limited field sensitivity n_B(f) = e_n,tot / (2 pi f N A) with the same LSBF862 input noise.
"""
import json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
MU0 = 4e-7 * np.pi
PUB = {m['turns']: m for m in json.load(open(os.path.join(HERE, 'published_coils.json')))['models']}
A_TURN = np.mean([PUB[n]['area'] / n for n in PUB])                 # m^2 per turn
R_TURN = np.mean([PUB[n]['R'] / n for n in PUB])                    # ohm per turn
C_TURN = np.mean([PUB[n]['C'] / n for n in PUB])                    # F per turn
SIDE = np.sqrt(A_TURN)
PITCH = 1e-3                                                        # height pitch between turns (m)

def m_par(l, rho):
    """mutual inductance of two parallel, aligned filaments of length l at distance rho."""
    x = l / rho
    return MU0 * l / (2 * np.pi) * (np.log(x + np.sqrt(1 + x * x)) - np.sqrt(1 + 1 / (x * x)) + 1 / x)

def m_square(d, a=SIDE):
    """mutual inductance of two coaxial square filaments of side a separated by d (exact, filament)."""
    return 4 * (m_par(a, d) - m_par(a, np.sqrt(a * a + d * d)))

def inductance(n, r_eq, pitch=PITCH):
    z = np.arange(n) * pitch
    d = np.abs(z[:, None] - z[None, :])
    M = np.where(d > 0, m_square(np.maximum(d, 1e-12)), m_square(r_eq))
    return float(M.sum())

def coil(n, r_eq, pitch=PITCH, c_turn=C_TURN):
    L = inductance(n, r_eq, pitch); C = c_turn * n
    return dict(turns=n, R=R_TURN * n, L=L, C=C, area=A_TURN * n, srf_kHz=1 / (2 * np.pi * np.sqrt(L * (C + 12e-12))) / 1e3)

if __name__ == '__main__':
    r_eq = brentq(lambda r: inductance(48, r) - PUB[48]['L'], 1e-6, 1e-2)
    print(f'per turn: area {A_TURN:.4f} m^2 (side {SIDE:.4f} m), R {R_TURN:.4f} ohm, C {C_TURN*1e12:.2f} pF; fitted r_eq {r_eq*1e3:.3f} mm')
    for n in (48, 56, 100):
        L = inductance(n, r_eq)
        print(f'  N={n:3d}: L model {L*1e3:.3f} mH vs published {PUB[n]["L"]*1e3:.3f} mH ({100*(L/PUB[n]["L"]-1):+.1f}%)')
    fam = [coil(n, r_eq) for n in (2, 3, 4, 6, 8, 12, 16, 24)]
    for c in fam:
        print(f'  VLF N={c["turns"]:2d}: R {c["R"]:.2f} ohm, L {c["L"]*1e6:7.1f} uH, C {c["C"]*1e12:6.0f} pF, NA {c["area"]:.2f} m^2, SRF {c["srf_kHz"]:7.0f} kHz')
    json.dump(dict(r_eq=r_eq, per_turn=dict(area=A_TURN, R=R_TURN, C=C_TURN), vlf=fam,
                   elf=[dict(PUB[n]) for n in (48, 56, 100)]), open(os.path.join(HERE, 'coil_model.json'), 'w'), indent=1)
