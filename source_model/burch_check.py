"""Check the half-space ground model against Burch et al., IEEE TAP 66(11) 2018 (rotating 3.1 A m^2 magnet).

Lighthouse orientation: m = M0 (x + j y), source 2.2 m above ground, receiver ~0.35 m (coil centres),
f = 500 Hz, offsets 10-128 m. Reported: |Br| 58 dBpT at 10 m, -2 dBpT at 100 m (1/r^3);
Bphi 6 dB below Br; Bz 16 dB below Br at 37 m; Bz-Br phase from 0 deg (short range) to -90 deg (long range);
moment 2 A m^2 (PEC-ground fit) to 4 A m^2 (free-space fit), true 3.1 A m^2.
"""
import numpy as np
import empymod

M0, HS, HR, F = 3.1, 2.2, 0.35, 500.0
MU0 = 4e-7 * np.pi

def fields(r, rho):
    out = []
    for ab_r in (4, 5, 6):                      # receiver Hx (radial), Hy (azimuthal), Hz
        hx = empymod.dipole(src=[0, 0, -HS], rec=[r, 0, -HR], depth=[0], res=[2e14, rho], freqtime=F, ab=int(f'{ab_r}4'), verb=0)
        hy = empymod.dipole(src=[0, 0, -HS], rec=[r, 0, -HR], depth=[0], res=[2e14, rho], freqtime=F, ab=int(f'{ab_r}5'), verb=0)
        # empymod magnetic-source fields are normalized by 1/(i omega mu0): multiply back
        out.append(MU0 * M0 * (hx + 1j * hy) * (1j * 2 * np.pi * F * MU0))
    return np.array(out)                          # Br, Bphi, Bz (T); empymod z is positive down

for rho in (2e14, 1000.0, 100.0, 10.0, 1.0):
    lab = 'free space' if rho > 1e10 else f'{rho:g} ohm m'
    print(f'\n{lab}:')
    for r in (10, 18, 37, 91, 128):
        Br, Bp, Bz = fields(float(r), rho)
        db = lambda b: 20 * np.log10(abs(b) / 1e-12)
        print(f'  r {r:4d} m: |Br| {db(Br):6.1f} dBpT, Bphi-Br {db(Bp)-db(Br):6.1f} dB, Bz-Br {db(Bz)-db(Br):6.1f} dB, '
              f'phase(Bz/Br) {np.degrees(np.angle(Bz/Br)):7.1f} deg')
