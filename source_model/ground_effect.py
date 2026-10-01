"""How much does a conductive (and weakly magnetic) earth change the field of a drone magnetic dipole?

Exact frequency-domain fields of x/y/z magnetic dipoles above a homogeneous half-space (empymod), compared
with free space. Metric: relative change of the full 3x3 source->receiver field tensor (Frobenius norm),
plus the change of the single largest component. Geometry: drone at height h_s, receiver (loop centre)
at 1 m height, horizontal offset r.
"""
import numpy as np
import empymod

FREQS = [100.0, 1e3, 5e3, 2e4, 5e4]
RHOS = [0.25, 3.0, 30.0, 300.0, 3000.0]      # seawater, wet clay, typical soil, dry soil, rock (ohm m)
CASES = [(2.0, 3.0), (2.0, 10.0), (10.0, 10.0), (20.0, 30.0), (50.0, 100.0)]   # (h_s, r) metres
Z_REC = 1.0

def tensor(h_s, r, f, rho, chi=0.0):
    G = np.zeros((3, 3), complex)
    for i in range(3):          # receiver Hx,Hy,Hz
        for j in range(3):      # source mx,my,mz
            ab = int(f'{4 + i}{4 + j}')
            G[i, j] = empymod.dipole(src=[0, 0, -h_s], rec=[r, 0, -Z_REC], depth=[0], res=[2e14, rho],
                                     freqtime=f, ab=ab, mpermH=[1, 1 + chi], mpermV=[1, 1 + chi], verb=0)
    return G

def rel_change(h_s, r, f, rho, chi=0.0):
    G0 = tensor(h_s, r, f, 2e14)                 # free space (ground = air)
    G = tensor(h_s, r, f, rho, chi)
    k = np.unravel_index(np.argmax(np.abs(G0)), G0.shape)
    return np.linalg.norm(G - G0) / np.linalg.norm(G0), abs(G[k] / G0[k]) - 1, np.degrees(np.angle(G[k] / G0[k]))

if __name__ == '__main__':
    print('Relative change of the 3x3 dipole field tensor due to the ground (Frobenius %), [largest component: '
          'amplitude %, phase deg]; receiver at 1 m height')
    for h_s, r in CASES:
        print(f'\n-- drone height {h_s:.0f} m, horizontal offset {r:.0f} m --')
        print('   rho(ohm m) ' + ''.join(f'{f/1e3:>16.1f} kHz' for f in FREQS))
        for rho in RHOS:
            cells = []
            for f in FREQS:
                fr, da, dp = rel_change(h_s, r, f, rho)
                cells.append(f'{100*fr:7.2f}% [{100*da:+5.1f}%,{dp:+5.1f}]')
            delta = 503 * np.sqrt(rho / np.array(FREQS))
            print(f'   {rho:9.2f}  ' + ' '.join(f'{c:>20s}' for c in cells) + f'   (skin depth {delta.min():.0f}-{delta.max():.0f} m)')
    print('\nMagnetic soil (chi = 0.01, rho = 300 ohm m), 1 kHz:')
    for h_s, r in CASES:
        fr, da, dp = rel_change(h_s, r, 1e3, 300.0, chi=0.01)
        print(f'   h_s {h_s:4.0f} m, r {r:5.0f} m: {100*fr:6.2f}% (largest component {100*da:+.2f}%)')
