"""Array model, part 2: localization Cramer-Rao bound from coherent complex line amplitudes.

Model per node channel: a_i = e^{j phi} * nhat_i . D(r_i - r) m,  m = M u (linearly polarised line moment, unknown
direction & magnitude), common phase phi.  Unknowns theta = (x, y, z, m_x, m_y, m_z, phi): 7 real.
Measurement noise: complex circular, E|w|^2 = sigma^2 = 2 n^2 / T  (so |a|^2/sigma^2 = coherent SNR over T).
Fisher information I = (2/sigma^2) Re(J^H J), J = da/dtheta (numerical). Position CRLB = sqrt(trace(I^-1)[0:3]).
Cases:
  (a) curtain (x = 0 plane, pitch p): drone at x = 1 m and 3 m in front of the curtain, random (y, z) in a cell,
      random orientation; scalar (normal x) vs triaxial; fundamental (T = 2 s) vs PWM carrier (T = 2 s).
  (b) the measured 4-antenna grid (5 m square, vertical loops at the corners facing the centre, heights 1 m),
      SNR set from the measured coherent line SNRs (1 s windows) -> compare with the 0.35 m measured median error.
"""
import copy, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import array_model as AM

def amp(theta, nodes, normals):
    r = theta[:3]; m = theta[3:6]; ph = theta[6]
    B = np.einsum('nij,j->ni', AM.dipole_tensor(nodes - r[None, :]), m)
    return np.exp(1j * ph) * np.einsum('ni,ni->n', B, normals)

def crlb_pos(r, m, nodes, normals, n, T, free=(0, 1, 2, 3, 4, 5, 6)):
    """free: indices of unknown parameters (0-2 position, 3-5 moment, 6 phase); position CRLB over free position axes."""
    th = np.r_[r, m, 0.3]
    J = np.empty((len(nodes), len(free)), complex)
    for j, k in enumerate(free):
        h = 1e-4 if k < 3 else max(1e-9, 1e-4 * np.linalg.norm(m)) if k < 6 else 1e-4
        e = np.zeros(7); e[k] = h
        J[:, j] = (amp(th + e, nodes, normals) - amp(th - e, nodes, normals)) / (2 * h)
    sig2 = 2 * n ** 2 / T
    I = 2 / sig2 * np.real(J.conj().T @ J)
    if np.linalg.cond(I) > 1e14:
        return np.inf
    C = np.linalg.inv(I)
    pos = [j for j, k in enumerate(free) if k < 3]
    return float(np.sqrt(max(np.trace(C[np.ix_(pos, pos)]), 0)))

def node_set(nodes, triax):
    if not triax:
        return nodes, np.tile([1.0, 0.0, 0.0], (len(nodes), 1))
    return np.repeat(nodes, 3, axis=0), np.tile(np.eye(3), (len(nodes), 1))

def curtain_case(m, n, p, triax, xoff, T=2.0, ntrial=200, seed=1):
    rng = np.random.default_rng(seed)
    base = AM.curtain_nodes(p, 30.0, Y=40.0)
    nodes, normals = node_set(base, triax)
    out = []
    for _ in range(ntrial):
        r = np.array([xoff, rng.uniform(0, p), rng.uniform(5, 25)])
        u = rng.normal(size=3); mv = m * u / np.linalg.norm(u)
        out.append(crlb_pos(r, mv, nodes, normals, n, T))
    return float(np.median(out)), float(np.percentile(out, 90))

if __name__ == '__main__':
    C = AM.DS.CLASSES
    dji = copy.deepcopy(C['photo/mid (11" props, 4S)']); dji['f_pwm'] = 85e3
    print('(a) Curtain localization CRLB, median [90th pct] position error (m), T = 2 s window')
    rows = []
    for name, c, bg in (('5-inch FPV', C['5-inch FPV (6S)'], 'indoor lab'), ('DJI-like photo', dji, 'semi-urban outdoor'),
                        ('heavy-lift', C['heavy-lift (30" props, 12S)'], 'semi-urban outdoor')):
        op, L = AM.DS.lines_for(c)
        fund = max([x for x in L if x[1] == 'phase leads' and abs(x[0] - op['f_e']) < 1], key=lambda x: x[2][1])
        m_pwm = float(np.sqrt(sum(x[2][1] ** 2 for x in L if abs(x[0] - c['f_pwm']) < 1)))
        n_f = float(np.hypot(AM.SC[bg](fund[0]), AM.wire_loop_noise(fund[0])[0]))
        n_p = float(np.hypot(AM.SC[bg](c['f_pwm']), AM.wire_loop_noise(c['f_pwm'])[0]))
        for lab, m, n, p in (('fundamental, pitch 4 m', fund[2][1], n_f, 4.0), ('PWM, pitch 11 m', m_pwm, n_p, 11.0),
                             ('PWM, pitch 16 m', m_pwm, n_p, 16.0)):
            for xoff in (1.0, 3.0):
                a = curtain_case(m, n, p, False, xoff); b = curtain_case(m, n, p, True, xoff)
                rows.append(dict(case=name, bg=bg, line=lab, xoff=xoff, scalar=a, triax=b))
                print(f'  {name:15s} {bg:19s} {lab:24s} x={xoff:.0f} m: scalar {a[0]:7.3f} [{a[1]:7.3f}]   triaxial {b[0]:7.3f} [{b[1]:7.3f}]', flush=True)

    # (b) measured grid geometry: corner antennas 0.1 m outside a 5 m square, loop normals toward the centre, 1 m high
    print('\n(b) Measured 4-antenna grid geometry, 1 s windows, SNR from the measured coherent line SNRs')
    A = np.load(os.path.join(HERE, '..', 'grid_analysis', 'complex_amps.npy'), allow_pickle=True).item()
    corners = np.array([[-0.1, 5.1, 1.0], [5.1, 5.1, 1.0], [5.1, -0.1, 1.0], [-0.1, -0.1, 1.0]])
    ctr = np.array([2.5, 2.5, 1.0]); nrm = (ctr - corners); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    res = []
    for cell in range(1, 26):
        pos = np.array([(cell - 1) % 5 + 0.5, 4.5 - (cell - 1) // 5, 1.0])
        # pick a moment magnitude/noise so that the model SNRs match the measured 1 s SNRs on average:
        snr1 = A[cell]['nv1']; a1 = np.mean(np.abs(A[cell]['w1']) ** 2, axis=0)
        meas = 10 * np.log10(np.maximum(a1 / snr1 - 1, 1e-3))                  # measured per-1 s coherent SNR (dB)
        rng = np.random.default_rng(cell); errs = []
        for _ in range(50):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            a = amp(np.r_[pos, u, 0.0], corners, nrm)
            mod = 10 * np.log10(np.abs(a) ** 2)                                  # model |a|^2 for unit moment, sigma^2 = 1
            g = np.mean(meas - mod)                                              # common scale to match measured SNRs
            n_eff = 1.0 / np.sqrt(2 * 10 ** (g / 10))                            # so that |a|^2/(2 n^2 / T) = SNR with T = 1
            errs.append(crlb_pos(pos, u, corners, nrm, n_eff, 1.0, free=(0, 1, 3, 4, 5, 6)))
        res.append(np.median(errs))
    print(f'  CRLB per 1 s window over 25 cells: median {np.median(res):.3f} m, 90th pct {np.percentile(res, 90):.3f} m, max {np.max(res):.3f} m'
          f'  (measured inversion error: median 0.35 m -> bias-limited if CRLB << 0.35 m)')
    json.dump(dict(curtain=rows, grid_crlb=[float(v) for v in res]), open(os.path.join(HERE, 'array_model_loc.json'), 'w'), indent=1)
