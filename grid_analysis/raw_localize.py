"""Localization tests on the raw grid data (1-s windows, 7 band powers x 4 channels).

E1 random split classification (report-like protocol)
E2 time-blocked classification (train first 70% of each 316 s record, test last 30%, 5-window gap)
E3 leave-one-cell-out (LOCO) position regression: the held-out cell is never seen in training
E4 LOCO physical model: per-channel power = floor + S / r^n from corner antennas, located by grid search
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'raw_windows.npz'))
X, y, t = D['X'], D['y'], D['t']
cells = np.arange(1, 26)
xy = np.array([[(c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5] for c in cells])     # cell centres (m)
XY = xy[y - 1]
rng = np.random.default_rng(0)

def cell_of(p):
    col = np.clip(np.floor(p[:, 0]), 0, 4); row = np.clip(np.floor(4.999 - p[:, 1]), 0, 4)
    return (row * 5 + col + 1).astype(int)

def report(name, pred_xy, true_c):
    err = np.linalg.norm(pred_xy - xy[true_c - 1], axis=1)
    pc = cell_of(pred_xy)
    cheb = np.maximum(np.abs((pc - 1) % 5 - (true_c - 1) % 5), np.abs((pc - 1) // 5 - (true_c - 1) // 5))
    print(f'{name:48s} median err {np.median(err):.2f} m, mean {err.mean():.2f} m | exact cell {np.mean(pc == true_c):.2f}, within 1 cell {np.mean(cheb <= 1):.2f}')
    return err

base = np.linalg.norm(xy - [2.5, 2.5], axis=1)
print(f'baseline "always centre": mean err {base.mean():.2f} m, exact cell 0.04')

# E1/E2 classification
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
idx = rng.permutation(len(y)); ntr = int(0.7 * len(y))
m = clf().fit(X[idx[:ntr]], y[idx[:ntr]])
print(f'E1 random split 70/30 classification accuracy: {accuracy_score(y[idx[ntr:]], m.predict(X[idx[ntr:]])):.3f} (chance 0.04)')
n_per = np.array([np.sum(y == c) for c in cells])
cut = 0.7 * n_per[y - 1]
tr = t < cut - 5; te = t >= cut
m = clf().fit(X[tr], y[tr])
pe = m.predict(X[te])
print(f'E2 time-blocked classification accuracy:        {accuracy_score(y[te], pe):.3f}')
# confusion structure of E2 errors: label-adjacent vs spatially-adjacent
wrong = pe != y[te]
dl = np.abs(pe[wrong] - y[te][wrong])
sp = np.maximum(np.abs((pe[wrong] - 1) % 5 - (y[te][wrong] - 1) % 5), np.abs((pe[wrong] - 1) // 5 - (y[te][wrong] - 1) // 5))
print(f'   E2 errors: {wrong.sum()} ; label distance 1-2: {np.mean(dl <= 2):.2f} ; spatial neighbours (Chebyshev 1): {np.mean(sp == 1):.2f}')

# E3 LOCO regression
preds = {k: np.zeros((len(y), 2)) for k in ('kNN', 'ridge')}
for c in cells:
    te = y == c; tr = ~te
    for k, mdl in (('kNN', make_pipeline(StandardScaler(), KNeighborsRegressor(25))),
                   ('ridge', make_pipeline(StandardScaler(), Ridge(1.0)))):
        mdl.fit(X[tr], XY[tr]); preds[k][te] = np.clip(mdl.predict(X[te]), 0, 5)
for k in preds:
    report(f'E3 LOCO {k} (per 1-s window)', preds[k], y)
    per_cell = np.array([np.median(preds[k][y == c], axis=0) for c in cells])
    report(f'E3 LOCO {k} (median over each cell record)', per_cell, cells)

# E4 physical model on excess power, LOCO
Pw = 10 ** X.reshape(len(y), 4, 7)          # linear band power, [window, channel, band]
feat = np.log10(Pw[:, :, 4] + Pw[:, :, 5])   # 5.2-6.9 kHz humps (drone harmonics), per channel
def ant(d):
    return np.array([[-d, 5 + d], [5 + d, 5 + d], [5 + d, -d], [-d, -d]])   # A,B,C,D corners
def model(params, pts):
    d, n = params[0], params[1]; lg = params[2:6]; lf = params[6:10]
    r = np.linalg.norm(pts[:, None, :] - ant(d)[None], axis=2)
    return np.log10(10 ** lf[None] + 10 ** lg[None] * r ** (-n))
cell_mean = np.array([feat[y == c].mean(0) for c in cells])
gx, gy = np.meshgrid(np.linspace(0, 5, 51), np.linspace(0, 5, 51)); G = np.c_[gx.ravel(), gy.ravel()]
pred4 = np.zeros((len(y), 2)); fits = []
for c in cells:
    trc = cells != c
    p0 = np.r_[0.3, 3.0, cell_mean[trc].max(0), cell_mean[trc].min(0)]
    fit = least_squares(lambda p: (model(p, xy[trc]) - cell_mean[trc]).ravel(), p0,
                        bounds=(np.r_[0, 0.5, [-10] * 8], np.r_[2, 8, [10] * 8]))
    fits.append(fit.x[:2])
    MG = model(fit.x, G)
    te = y == c
    res = ((feat[te][:, None, :] - MG[None]) ** 2).sum(2)
    pred4[te] = G[np.argmin(res, axis=1)]
fits = np.array(fits)
print(f'E4 fitted antenna offset d = {fits[:,0].mean():.2f} m, decay exponent n = {fits[:,1].mean():.2f} (power ~ r^-n; dipole amplitude r^-3 -> n=6)')
report('E4 LOCO physical model (per 1-s window)', pred4, y)
per_cell4 = np.array([np.median(pred4[y == c], axis=0) for c in cells])
e = report('E4 LOCO physical model (median over each record)', per_cell4, cells)
np.savez(os.path.join(HERE, 'raw_localize.npz'), pred_knn=preds['kNN'], pred_phys=pred4, y=y, xy=xy)
print('per-cell errors (m), physical model:', np.round(e, 2))
