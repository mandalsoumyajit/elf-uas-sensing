"""Link-budget estimate of single-axis detection range for a motor tone.

Signal: one motor line of peak moment m (A m^2), per-axis field amplitude averaged over random
relative orientation, A(r) = sqrt(2/3) * (mu0/4pi) * m / r^3  (quasistatic dipole).
Noise: one-sided field ASD n = sqrt(n_amb^2 + n_inst^2) (T/sqrt(Hz)), white near the tone.
Detector: FFT of segments of length Tc; the statistic sums |X|^2 over M = T/Tc segments at the tone bin
(M = 1 is coherent over T). Per-bin SNR per segment = A^2 Tc / (2 n^2).
Threshold: false-alarm probability 1e-3 per record over a searched band (frequency unknown),
detection probability 0.9. Required SNR from exact (non)central chi-square statistics.

Instrument noise:
  wire loop (report Antenna 4, octagonal): 300 turns, ~1 m mean diameter, SWG 27 or SWG 35,
     OPA227-class preamp (3 nV/rtHz, 0.4 pA/rtHz), assumed AFE gain 1000 and x1/3 attenuation
     into a 12-bit 3.3 V ADC at 10 kS/s (quantization referred to the loop input).
  PCB coil: manuscript Table (48-/100-turn): 12.94 / 7.11 fT/rtHz at 1 kHz, scaled as 1/f.
Everything marked ASSUMED is a placeholder to be replaced by measured values.
"""
import os
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MU0_4PI = 1e-7
KB, TEMP = 1.380649e-23, 300.0
RHO_CU = 1.72e-8

# ---------------- source ----------------
M_NOMINAL = 2e-3            # A m^2, simulation axial moment (ASSUMED; not calibrated at these frequencies)
def amp(r, m):              # per-axis peak amplitude, orientation-averaged
    return np.sqrt(2 / 3) * MU0_4PI * m / r ** 3

# ---------------- instrument noise ----------------
def wire_loop_noise(f, swg_diam_mm=0.4166, turns=300, diam=1.0, gain=1000.0, L=0.2,
                    en=3e-9, inoise=0.4e-12, fs=1e4, atten=1 / 3, bits=12, vref=3.3):
    area = np.pi * (diam / 2) ** 2 * 0.95                         # octagon ~95% of circle (ASSUMED)
    R = RHO_CU * turns * np.pi * diam / (np.pi * (swg_diam_mm * 1e-3 / 2) ** 2)
    e_th = np.sqrt(4 * KB * TEMP * R)
    e_i = inoise * np.abs(R + 2j * np.pi * f * L)                  # L ASSUMED
    e_q = (vref / 2 ** bits) / np.sqrt(12) / np.sqrt(fs / 2) / atten / gain
    e_tot = np.sqrt(e_th ** 2 + en ** 2 + e_i ** 2 + e_q ** 2)
    nB = e_tot / (2 * np.pi * f * turns * area)
    return nB, dict(R=R, e_th=e_th, e_amp=en, e_i=e_i, e_q=e_q, turn_area=turns * area)

def pcb_noise(f, nB_1k):
    return nB_1k * (1e3 / f)

# ---------------- detection ----------------
def required_snr(M, pfa_bin, pd=0.9):
    """Per-segment per-bin SNR needed when summing M segments (2M-dof chi-square)."""
    dof = 2 * M
    thr = chi2.isf(pfa_bin, dof)
    return brentq(lambda s: ncx2.sf(thr, dof, 2 * M * s) - pd, 1e-6, 1e6)

def max_range(m, n, T, Tc, band_hz=500.0, pfa_record=1e-3, pd=0.9):
    M = max(1, int(round(T / Tc)))
    nbins = max(1, band_hz * Tc)
    s = required_snr(M, pfa_record / nbins, pd)
    # A(r)^2 Tc / (2 n^2) = s  ->  r
    A_req = np.sqrt(2 * s * n ** 2 / Tc)
    return (np.sqrt(2 / 3) * MU0_4PI * m / A_req) ** (1 / 3)

if __name__ == '__main__':
    for f in (330.0, 665.0):
        print(f'\n=== tone frequency {f:.0f} Hz ===')
        for name, dmm in (('Antenna 4, SWG 27', 0.4166), ('Antenna 4, SWG 35', 0.2134)):
            nB, parts = wire_loop_noise(f, swg_diam_mm=dmm)
            print(f'{name}: R={parts["R"]:.0f} ohm, turn area {parts["turn_area"]:.0f} m^2; input noise nV/rtHz: '
                  f'thermal {parts["e_th"]*1e9:.2f}, amp {parts["e_amp"]*1e9:.2f}, i*Z {parts["e_i"]*1e9:.2f}, ADC {parts["e_q"]*1e9:.2f}'
                  f'  -> field {nB*1e15:.1f} fT/rtHz')
        print(f'PCB coil 100/48 turns: {pcb_noise(f, 7.11e-15)*1e15:.1f} / {pcb_noise(f, 12.94e-15)*1e15:.1f} fT/rtHz')

    f = 665.0
    n_inst, _ = wire_loop_noise(f, swg_diam_mm=0.2134)
    ambients = [0.03e-12, 0.3e-12, 3e-12, 30e-12]
    print('\nMax range (m) for m = 2e-3 A m^2 at 665 Hz, wire loop (SWG 35), Pd 0.9, Pfa 1e-3/record over 500 Hz')
    print(f"{'ambient pT/rtHz':>16} {'0.2 s':>7} {'150x0.2 s incoh':>16} {'60 s incoh(0.2)':>16} {'60 s incoh(1 s)':>16} {'60 s coherent':>14}")
    rows = []
    for na in ambients:
        n = np.hypot(na, n_inst)
        r = [max_range(M_NOMINAL, n, 0.2, 0.2), max_range(M_NOMINAL, n, 30, 0.2), max_range(M_NOMINAL, n, 60, 0.2),
             max_range(M_NOMINAL, n, 60, 1.0), max_range(M_NOMINAL, n, 60, 60)]
        rows.append(r)
        print(f'{na*1e12:16.2f} ' + ' '.join(f'{x:>{w}.1f}' for x, w in zip(r, (7, 16, 16, 16, 14))))
    print('Range scales as m^(1/3): x0.1 moment -> x0.46 range; x10 -> x2.15.')

    # Consistency with the SCF dataset: per-image visibility to ~2 m, session average (150 windows) to ~3 m
    ratio = max_range(M_NOMINAL, 1e-12, 30, 0.2) / max_range(M_NOMINAL, 1e-12, 0.2, 0.2)
    print(f'\nModel range gain, 150 incoherent 0.2 s windows vs one: x{ratio:.2f} (dataset: ~2 m -> ~3 m, x1.5)')
    n_implied = brentq(lambda n: max_range(M_NOMINAL, n, 0.2, 0.2) - 2.0, 1e-15, 1e-9)
    print(f'Background implied by 2 m single-window visibility at m = 2e-3 A m^2: {n_implied*1e12:.1f} pT/rtHz '
          f'(scales as m; the images are not an optimal detector, so this is an upper bound on n/m)')

    # Figure: range vs integration time
    T = np.logspace(np.log10(0.2), np.log10(300), 30)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cols = plt.get_cmap('viridis')(np.linspace(0.1, 0.85, len(ambients)))
    for na, c in zip(ambients, cols):
        n = np.hypot(na, n_inst)
        ax.plot(T, [max_range(M_NOMINAL, n, t, 0.2) for t in T], color=c, lw=1.6, label=f'{na*1e12:g} pT/√Hz')
        ax.plot(T, [max_range(M_NOMINAL, n, t, t) for t in T], color=c, lw=1.0, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('record length T (s)'); ax.set_ylabel('detection range (m), Pd 0.9')
    ax.set_title('m = 2e-3 A m², 665 Hz; solid: incoherent (0.2 s segments), dashed: coherent', fontsize=8)
    ax.axvline(0.2, color='0.6', lw=0.8, ls=':'); ax.text(0.21, ax.get_ylim()[0] * 1.1, 'one SCF image', fontsize=7, color='0.4')
    ax.legend(title='ambient ASD', fontsize=7, title_fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'range_vs_T.png'), dpi=160)
    print('saved range_vs_T.png')
