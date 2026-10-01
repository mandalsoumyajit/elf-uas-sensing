"""LTspice driver for the ELF and VLF node receiver studies.

Builds a netlist from the published version-1 receiver core (core.inc) with the coil (R, L, C), the active-damping
network (CF, RINT, RLEAK) and the output-filter file substituted, runs LTspice in batch mode (ASCII raw output) and
returns the AC transfer and input-referred noise.

The vendor device models are not distributed with this repository (see README.md); they must be placed in this
folder before running. The LTspice executable is taken from the LTSPICE environment variable, or the default
Windows install location.
"""
import os, re, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LT = Path(os.environ.get('LTSPICE', Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/ADI/LTspice/LTspice.exe'))
CORE = (HERE / 'core.inc').read_text()
VENDOR = {'ADA4511.lib': 'Analog Devices ADA4511 SPICE macro model (analog.com product page)',
          'LSBF862_from_pdf.lib': 'Linear Systems LSBF862 JFET model (linearsystems.com), model name LSBF862_UPDATED',
          'clamp_diode.lib': 'the 1N4148 .model line from LTspice\'s standard.dio'}

def check_models(opamp='ADA4511'):
    need = dict(VENDOR)
    if opamp != 'ADA4511':
        need[f'{opamp}.lib'] = f'Analog Devices {opamp} SPICE macro model'
        del need['ADA4511.lib']
    missing = [f'  {f}: {src}' for f, src in need.items() if not (HERE / f).exists()]
    if missing:
        raise FileNotFoundError('vendor models missing from ' + str(HERE) + ' (see README.md):\n' + '\n'.join(missing))
    if not LT.exists():
        raise FileNotFoundError(f'LTspice not found at {LT}; set the LTSPICE environment variable')

def core_text(R, L, C, CF, RINT, RLEAK, filters, ccomp=None, opamp='ADA4511', rshunt=None):
    s = CORE
    if opamp != 'ADA4511':
        s = s.replace('.include ADA4511.lib', f'.include {opamp}.lib').replace(' ADA4511\n', f' {opamp}\n')
    if filters != 'output_filters.inc':                    # VLF filter file uses a placeholder op-amp name
        ftxt = (HERE / filters).read_text().replace(' OPAMP\n', f' {opamp}\n')
        (HERE / f'_{opamp}_{filters}').write_text(ftxt); filters = f'_{opamp}_{filters}'
    s = s.replace('.include coil_sources.lib\n', '')
    s = re.sub(r'\.param Lscale=1 Ccoil=\{Cestimated\(Nturn\)\}', f'.param Lscale=1 Ccoil={C:.6g}', s)
    s = s.replace('RCOIL emf rl {Rcoil(Nturn)}', f'RCOIL emf rl {R:.6g}')
    s = s.replace('LCOIL rl gate {Lcoil(Nturn)*Lscale} Rser=0', f'LCOIL rl gate {L:.6g} Rser=0')
    s = s.replace('.include coil_capacitance.lib\n', '')
    s = re.sub(r'\.param CF=\{table\(Nturn[^}]*\)\} CINT=10n RLEAK=\{table\(Nturn[^}]*\)\}',
               f'.param CF={CF:.6g} CINT=10n RLEAK={RLEAK:.6g}', s)
    s = re.sub(r'\.param GCRIT=.*\n', '', s)
    s = re.sub(r'\.param RINT=\{table\(Nturn[^}]*\)\}', f'.param RINT={RINT:.6g}', s)
    s = s.replace('.include output_filters.inc', f'.include {filters}')
    if ccomp is not None:
        s = re.sub(r'CCOMP drain2 supply2 \S+', f'CCOMP drain2 supply2 {ccomp:.6g}', s)
    if rshunt is not None:                                   # passive damping resistor across the coil input
        s = s.replace('RBLEED gate 0 10Meg', f'RBLEED gate 0 10Meg\nRSHUNTP gate 0 {rshunt:.6g}')
    for tok in ('Nturn)', 'Cestimated', 'table('):
        assert tok not in s, tok
    return s

def raw(path):
    b = path.read_bytes()
    t = b.decode('utf-16' if b.startswith(b'\xff\xfe') else ('utf-16-le' if b[:2] == b'T\x00' else 'cp1252')).replace('\r\n', '\n')
    h, v = t.split('Values:\n', 1)
    nv = int(re.search(r'No. Variables:\s*(\d+)', h)[1])
    names = [l.split()[1].lower() for l in h.split('Variables:\n', 1)[1].split('Values:')[0].splitlines() if l.strip()][:nv]
    vals = []
    for l in v.splitlines():
        if not l.strip():
            continue
        z = l.split()[-1]
        vals.append(complex(*map(float, z.split(','))) if ',' in z else float(z))
    return dict(zip(names, np.asarray(vals).reshape(-1, nv).T))

def run(name, core, analysis, mode=1, fstart=10, fstop=10e6):
    """Run one LTspice job ('ac' or 'noise'). Jobs must run sequentially: concurrent batch runs collide."""
    check_models(re.search(r'\.include (\S+)\.lib\n\.include bias_hints', core)[1])
    (HERE / f'{name}_core.inc').write_text(core)
    line = f'.ac dec 100 {fstart} {fstop}\n.save V(gate) V(out) V(filtered)' if analysis == 'ac' else \
           f'.noise V(filtered) VEMF dec 100 {fstart} {fstop}'
    (HERE / f'{name}_{analysis}.cir').write_text(f'receiver study\n.include {name}_core.inc\n.param Nturn=0 Mode={mode} EAMP=0 FREQ=1k\n'
                                                f'{line}\n.options plotwinsize=0\n.end\n')
    kw = {}
    if os.name == 'nt':
        si = subprocess.STARTUPINFO(); si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; si.wShowWindow = 0; kw['startupinfo'] = si
    p = subprocess.run([str(LT), '-b', '-ascii', f'{name}_{analysis}.cir'], cwd=HERE, capture_output=True, timeout=300, **kw)
    log = (HERE / f'{name}_{analysis}.log').read_text(errors='replace')
    assert p.returncode == 0 and 'Total elapsed time' in log and not re.search('Fatal Error|Time step too small|singular', log, re.I), log[-2000:]
    return raw(HERE / f'{name}_{analysis}.raw')
