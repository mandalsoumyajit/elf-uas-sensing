"""Cost of co-locating the ELF and VLF windings: effect of a coupled ELF winding on the VLF receiver (LTspice).

VLF channel: 12-turn coil, retuned active damping (CF = 0.086 C, RINT = 101 ohm, leak pole 1.6 kHz) or a 24-turn coil
(RINT = 399 ohm), VLF output filters, unchanged front end.  The ELF winding (100 or 48 turns) is added as a second
inductor coupled with K = k (from coupling.py: offset 0.09, nested 0.35, shared boards 0.53-0.59, plus k = 0),
terminated by its own active damping in the VLF band: C_self + C_f + 12 pF in parallel with the synthesized
critical-damping conductance (noiseless; the published ELF design), and driven by the same uniform field
(EMF scaled by the ratio of turn areas).  Outputs: VLF transfer (filtered/EMF) ripple across 10-100 kHz relative to
k = 0, and the field-equivalent noise at 24, 48 and 85 kHz.
"""
import json
import numpy as np
import spice as S

COILS = {c['turns']: c for c in json.load(open(S.HERE / 'coil_model.json'))['vlf']}
ELF = {m['turns']: m for m in json.load(open(S.HERE / 'coil_model.json'))['elf']}
CFE = {48: 120e-12, 56: 150e-12, 100: 270e-12}
KPU = json.load(open(S.HERE / 'coupling.json'))
VLF = {12: 101.0, 24: 399.0}

def add_elf(core, nE, k, ratio):
    e = ELF[nE]; c = e['C'] + CFE[nE] + 12e-12
    g = e['R'] * c / e['L'] + 2 * np.sqrt(c / e['L'])
    block = (f'EELF eE 0 emf 0 {ratio:.6g}\nRELF eE e1 {e["R"]:.6g}\nLELF e1 gE {e["L"]:.6g} Rser=0\n'
             f'CELF gE 0 {c:.6g}\nRDELF gE 0 {1/g:.6g} noiseless\nRBLE gE 0 10Meg\nK1 LCOIL LELF {k:.4f}\n')
    return core.replace('CSELF gate 0 {Ccoil}', 'CSELF gate 0 {Ccoil}\n' + block)

if __name__ == '__main__':
    res = []
    for nV, rint in VLF.items():
        c = COILS[nV]
        base = S.core_text(c['R'], c['L'], c['C'], 0.086 * c['C'], rint, 1 / (2 * np.pi * 1.6e3 * 10e-9), 'output_filters_vlf.inc')
        ref = None
        for nE in (100, 48):
            ks = sorted({0.0} | {round(KPU[f'{nE}|{min(nV,16)}|{a}']['k'], 3) for a in ('offset', 'nested', 'shared')})
            for k in ks:
                core = add_elf(base, nE, max(k, 1e-6), ELF[nE]['area'] / c['area'])
                name = f'cpl_v{nV}_e{nE}_k{int(1000*k)}'
                a = S.run(name, core, 'ac', fstart=1e3, fstop=3e6); z = S.run(name, core, 'noise', fstart=1e3, fstop=3e6)
                f = a['frequency'].real; H = a['v(filtered)']
                if k == 0.0:
                    ref = (f, H)
                band = (f >= 1e4) & (f <= 1e5)
                rip = 20 * np.log10(np.abs(H[band]) / np.abs(np.interp(f[band], ref[0], np.abs(ref[1]))))
                fz = z['frequency'].real; en = z['v(inoise)'].real
                nB = {int(x / 1e3): float(np.interp(x, fz, en) / (2 * np.pi * x * c['area']) * 1e15) for x in (24e3, 48e3, 85e3)}
                worst = float(f[band][np.argmax(np.abs(rip))])
                res.append(dict(vlf_turns=nV, elf_turns=nE, k=k, ripple_db_min=float(rip.min()), ripple_db_max=float(rip.max()),
                                worst_f=worst, nB_fT=nB))
                print(f'VLF {nV:2d} / ELF {nE:3d}, k = {k:.3f}: VLF response vs uncoupled {rip.min():+6.1f} .. {rip.max():+6.1f} dB '
                      f'(worst at {worst/1e3:5.1f} kHz); nB 24/48/85 kHz = {nB[24]:.2f}/{nB[48]:.2f}/{nB[85]:.2f} fT', flush=True)
    json.dump(res, open(S.HERE / 'coupled_study.json', 'w'), indent=1)
