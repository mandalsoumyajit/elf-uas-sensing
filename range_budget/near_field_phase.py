"""Phase of the exact oscillating magnetic-dipole field vs the naive retarded delay r/c.

Exact fields (exp(+j w t) convention), up to common constants:
  B_r     ~ 2 cos(th) (1/r^3 + j k/r^2) exp(-j k r)
  B_theta ~   sin(th) (1/r^3 + j k/r^2 - k^2/r) exp(-j k r)
Compare the phase with the source at each range to -k r (what a propagation delay r/c would give).
"""
import numpy as np

c = 299792458.0
for f in (665.0, 16e3, 1e6, 30e6):
    k = 2 * np.pi * f / c
    print(f'\nf = {f:>10.0f} Hz, wavelength {c/f/1e3:10.3f} km')
    for r in (1.0, 5.0, 20.0):
        br = (1 + 1j * k * r) * np.exp(-1j * k * r)
        bt = (1 + 1j * k * r - (k * r) ** 2) * np.exp(-1j * k * r)
        print(f'  r={r:5.1f} m  naive delay phase {-k*r:+.3e} rad ({r/c*1e9:5.2f} ns) | '
              f'B_r phase {np.angle(br):+.3e} rad (apparent delay {-np.angle(br)/(2*np.pi*f)*1e9:+.2e} ns) | '
              f'B_th phase {np.angle(bt):+.3e} rad (apparent delay {-np.angle(bt)/(2*np.pi*f)*1e9:+.2e} ns)')
