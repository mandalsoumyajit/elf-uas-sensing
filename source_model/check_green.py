"""Validate site_model.Green (table interpolation + rotation) against direct empymod at arbitrary azimuths."""
import numpy as np
import empymod
import site_model as SM
import ground_tables as GTB

rng = np.random.default_rng(3)
for f, rho_g in ((1074.0, 100.0), (85e3, 30.0), (24e3, 1000.0)):
    G = SM.Green(f, rho_g)
    hs0, hr0 = 3.0, 1.0
    T = GTB.analytic_free(hs0, hr0)
    kappa = np.median(T[:, 0, 0] / GTB.field(44, hs0, hr0, f, 2e14))
    errs = []
    for _ in range(6):
        src = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(2, 30)])
        node = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(1.5, 25)])
        Gm = G(src[None], node[None])[0, 0]
        E = np.zeros((3, 3), complex)
        for i in range(3):
            for j in range(3):
                v = empymod.dipole(src=[src[0], src[1], -src[2]], rec=[node[0], node[1], -node[2]], depth=[0], res=[2e14, rho_g],
                                   freqtime=f, ab=int(f'{4 + i}{4 + j}'), verb=0)
                E[i, j] = kappa * v * (-1 if (i == 2) != (j == 2) else 1)      # z-down -> z-up
        errs.append(np.linalg.norm(Gm - E) / np.linalg.norm(E))
    print(f'f {f/1e3:6.2f} kHz rho_g {rho_g:6.0f}: rel. error table vs direct empymod: max {max(errs):.2e}')
# PEC-like limit: vertical moment image should be reversed, horizontal same sign
f, rho_g = 85e3, 30.0
print('(sanity) secondary/direct for vertical moment directly above, h_s 2 m, h_r 1.5 m (expect -> -(d/Hs)^3 as rho_g -> 0):')
G = SM.Green(f, rho_g); src = np.array([[0, 0, 2.0]]); node = np.array([[0.01, 0, 1.5]])
tot = G(src, node)[0, 0]; fr = SM.AM.dipole_tensor(node - src)[0]
print('   zz', (tot[2, 2] - fr[2, 2]) / fr[2, 2], ' image-PEC value', -(0.5 / 3.5) ** 3, '| xx', (tot[0, 0] - fr[0, 0]) / fr[0, 0], 'PEC', (0.5 / 3.5) ** 3)
