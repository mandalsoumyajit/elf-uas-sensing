"""Ground-reflected (secondary) field tables for magnetic dipoles in air above a homogeneous conductive half-space.

For source and receiver both in air, the secondary field depends only on the horizontal offset rho and on the
sum of heights Hs = h_src + h_rec (quasi-static in air; displacement currents negligible below 100 kHz).
In the local frame (x along the source->receiver horizontal direction, z up) the secondary B per unit moment is
    S = [[Sxx, 0, Sxz], [0, Syy, 0], [Szx, 0, Szz]]
Five complex functions of (rho, Hs) are tabulated with empymod (total minus free space), with the overall
empymod scale fixed by matching its free-space field to the analytic dipole tensor (mu0/4pi = 1e-7 T m/A).
Stored as S * (rho^2 + Hs^2)^(3/2) (smooth, ~image-dipole scaled) for log-log interpolation.
"""
import os
import numpy as np
import empymod

HERE = os.path.dirname(os.path.abspath(__file__))
FREQS = [366.0, 802.0, 1074.0, 16e3, 24e3, 48e3, 85e3]
RHOG = [30.0, 100.0, 1000.0]
RHO = np.logspace(-1, np.log10(500), 70)
HS = np.logspace(np.log10(0.5), np.log10(400), 60)
AB = {'xx': 44, 'xz': 46, 'yy': 55, 'zx': 64, 'zz': 66}     # receiver digit first, source second (4/5/6 = Hx/Hy/Hz)

def field(ab, hs, hr, f, rho_g):
    return np.asarray(empymod.dipole(src=[0, 0, -hs], rec=[RHO, np.zeros_like(RHO), -hr], depth=[0],
                                     res=[2e14, rho_g], freqtime=f, ab=ab, verb=0))

def analytic_free(hs, hr):
    d = np.c_[RHO, np.zeros_like(RHO), np.full_like(RHO, hr - hs)]       # z up
    r = np.linalg.norm(d, axis=1); u = d / r[:, None]
    T = 1e-7 / r[:, None, None] ** 3 * (3 * u[:, :, None] * u[:, None, :] - np.eye(3))
    return T

if __name__ == '__main__':
    out = {}
    for f in FREQS:
        # empymod scale and sign convention (z positive down) from free space at a reference geometry
        hs0, hr0 = 3.0, 1.0
        T = analytic_free(hs0, hr0)
        e_xx = field(44, hs0, hr0, f, 2e14); e_zz = field(66, hs0, hr0, f, 2e14); e_xz = field(46, hs0, hr0, f, 2e14)
        kappa = np.median(T[:, 0, 0] / e_xx)
        sz = np.sign(np.real(np.median(T[:, 0, 2] / (kappa * e_xz))))      # z-flip sign for mixed components
        chk = max(np.max(np.abs(kappa * e_zz - T[:, 2, 2]) / np.abs(T[:, 2, 2])),
                  np.max(np.abs(sz * kappa * e_xz - T[:, 0, 2]) / (np.abs(T[:, 0, 2]) + 1e-3 * np.abs(T[:, 0, 0]))))
        print(f'f = {f/1e3:6.2f} kHz: empymod scale |kappa| = {abs(kappa):.3e}, free-space check max rel err {chk:.1e}', flush=True)
        for rho_g in RHOG:
            tab = {}
            for name, ab in AB.items():
                sgn = sz if name in ('xz', 'zx') else 1.0
                rows = []
                for H in HS:
                    hs, hr = H / 2, H / 2
                    sec = field(ab, hs, hr, f, rho_g) - field(ab, hs, hr, f, 2e14)
                    rows.append(sgn * kappa * sec * (RHO ** 2 + H ** 2) ** 1.5)
                tab[name] = np.array(rows).T                      # (rho, Hs)
            out[(f, rho_g)] = tab
            # report: secondary/direct at a few geometries (node at 8 m, drone at 20 m, offsets)
            for rr, hs_, hr_ in ((5.0, 20.0, 8.0), (15.0, 20.0, 8.0), (15.0, 5.0, 1.5)):
                i = np.argmin(np.abs(RHO - rr)); j = np.argmin(np.abs(HS - (hs_ + hr_)))
                Sd = np.abs(tab['zz'][i, j]) / (RHO[i] ** 2 + HS[j] ** 2) ** 1.5
                dd = np.hypot(RHO[i], hs_ - hr_); Dz = 1e-7 / dd ** 3 * abs(3 * ((hs_ - hr_) / dd) ** 2 - 1)
                print(f'   rho_g {rho_g:6.0f}: |S_zz|/|direct_zz| at rho {RHO[i]:4.1f} m, h_s {hs_:.0f}, h_r {hr_:.1f}: {Sd / Dz:6.3f}', flush=True)
    np.save(os.path.join(HERE, 'ground_tables.npy'), dict(tabs=out, RHO=RHO, HS=HS, FREQS=FREQS, RHOG=RHOG), allow_pickle=True)
