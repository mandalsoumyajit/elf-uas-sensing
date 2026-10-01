"""Per-file quality stats and Welch PSDs of the raw grid recordings (ADC counts)."""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, 'raw_export')
FS = 20000.0
NPER = 2 ** 16          # 0.305 Hz resolution

def one(path):
    d = sio.loadmat(path)
    x = d['x'].ravel().astype(np.float64)
    clip = np.mean((x <= 0) | (x >= 4095))
    f, p = welch(x - x.mean(), fs=FS, nperseg=NPER, noverlap=NPER // 2, window='hann', detrend=False)
    return os.path.basename(path), clip, x.mean(), x.std(), p.astype(np.float32), f

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(EXP, 'c*_?.mat')))
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, files))
    names = [r[0] for r in res]
    np.savez_compressed(os.path.join(HERE, 'raw_psd.npz'), names=np.array(names),
                        clip=np.array([r[1] for r in res]), mean=np.array([r[2] for r in res]),
                        std=np.array([r[3] for r in res]), psd=np.stack([r[4] for r in res]), f=res[0][5])
    print('file      clip%   mean    std')
    for r in res:
        print(f'{r[0]:10s} {100*r[1]:6.2f} {r[2]:7.1f} {r[3]:6.1f}')
