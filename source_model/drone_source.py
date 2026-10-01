"""Step 1: equivalent-source ELF/VLF magnetic model of multirotor drones, parameterized by drone class.

Each motor contributes oscillating/rotating magnetic dipole moments at discrete spectral lines:
  rotor   : residual (imbalance) dipole of the PM rotor, rotating at f_mech
  leads   : ESC->motor phase leads (3 conductors), phase current at f_e and its harmonics
            (six-step: 6k+-1 with ~1/n roll-off; FOC: mostly f_e), plus PWM ripple at f_pwm
  bus     : DC leads (battery->ESC or along-arm power feed): 6 f_e commutation ripple (six-step),
            residual PWM ripple after the ESC bulk capacitors
  esc     : ESC switching loop (bridge <-> DC-link capacitor), chopped current at f_pwm
  winding : net dipole of the stator winding at f_e (placeholder bracket; computed in step 2)
Every quantity is carried as (low, nominal, high) to keep the uncertainty explicit.
Operating point: hover thrust per motor from AUW; RPM from a published thrust-table point (RPM ~ sqrt(T));
electrical power from the same point scaled as T^1.5 (momentum theory); currents from Kt = 60/(2*pi*Kv).
Field scale: orientation-averaged per-axis peak amplitude B = sqrt(2/3) * 1e-7 * m / r^3.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MU0 = 4e-7 * np.pi

# ----------------------------------------------------------------------------------------------
# Drone classes. Sources for the motor rows are listed in SOURCE_MODEL_PLAN_2026-10-01.md.
# "ref" = one published thrust-table point (thrust_g, rpm, current_A, voltage_V).
# Geometry and wiring are typical-build estimates (ranges where uncertain).
# ----------------------------------------------------------------------------------------------
CLASSES = {
    'micro (sub-250 g, 3", 4S)': dict(
        auw_kg=0.25, n_motors=4, arm_m=0.065, motor='T-Motor F1404 3800Kv', kv=3800, pole_pairs=6,
        ref=(304.0, 35355, 13.49, 15.2), v_batt=14.8, stator_d_mm=14, stator_h_mm=4, n_mag=12, mag_t_mm=1.0,
        drive='six-step', f_pwm=48e3, esc='central', lead_len_m=(0.03, 0.05, 0.07), lead_pitch_m=(1.2e-3, 1.5e-3, 2.5e-3),
        bus_len_m=(0.05, 0.08, 0.12), bus_pitch_m=(2e-3, 3e-3, 6e-3), L_ll_H=(8e-6, 15e-6, 30e-6)),
    '5-inch FPV (6S)': dict(
        auw_kg=0.65, n_motors=4, arm_m=0.11, motor='T-Motor F60 Pro V 2207.5 1950Kv', kv=1950, pole_pairs=7,
        ref=(1966.1, 32031, 46.8, 24.7), v_batt=22.2, stator_d_mm=22, stator_h_mm=7, n_mag=14, mag_t_mm=1.5,
        drive='six-step', f_pwm=24e3, esc='central', lead_len_m=(0.05, 0.08, 0.12), lead_pitch_m=(1.5e-3, 2e-3, 4e-3),
        bus_len_m=(0.08, 0.12, 0.18), bus_pitch_m=(3e-3, 5e-3, 10e-3), L_ll_H=(10e-6, 20e-6, 40e-6)),
    'photo/mid (11" props, 4S)': dict(
        auw_kg=2.0, n_motors=4, arm_m=0.275, motor='T-Motor MN3110 700Kv', kv=700, pole_pairs=7,
        ref=(320.0, 5500, 2.3, 14.8), v_batt=14.8, stator_d_mm=31, stator_h_mm=10, n_mag=14, mag_t_mm=2.0,
        drive='FOC', f_pwm=20e3, esc='arm', lead_len_m=(0.03, 0.06, 0.10), lead_pitch_m=(1.5e-3, 2.5e-3, 5e-3),
        bus_len_m=(0.20, 0.25, 0.30), bus_pitch_m=(3e-3, 6e-3, 15e-3), L_ll_H=(40e-6, 80e-6, 160e-6)),
    'heavy-lift (30" props, 12S)': dict(
        auw_kg=20.0, n_motors=4, arm_m=0.60, motor='KDE7215XF-135', kv=135, pole_pairs=11,
        ref=None, prop_d_m=0.775, v_batt=44.4, stator_d_mm=72, stator_h_mm=15, n_mag=22, mag_t_mm=4.0,
        drive='FOC', f_pwm=16e3, esc='arm', lead_len_m=(0.05, 0.10, 0.15), lead_pitch_m=(3e-3, 5e-3, 10e-3),
        bus_len_m=(0.50, 0.60, 0.70), bus_pitch_m=(5e-3, 10e-3, 25e-3), L_ll_H=(50e-6, 100e-6, 200e-6)),
}
# Generic uncertain factors, (low, nominal, high)
MAG_BR = 1.3                                  # NdFeB remanence (T)
ROTOR_IMBALANCE = (0.01, 0.03, 0.10)          # residual rotating dipole / (m_magnet * sqrt(N_mag))
BELL_SHIELD = (0.3, 0.6, 1.0)                 # steel rotor-bell reduction of the residual dipole
SIXSTEP_HARM = {'roll': (2.0, 1.5, 1.0)}      # harmonic amplitude ~ I1 / n^roll (6k+-1); low = steeper roll-off
FOC_THD = (0.01, 0.03, 0.05)                  # residual 5th/7th fraction under FOC
RIPPLE_6FE = (0.05, 0.15, 0.30)               # DC-bus 6f_e ripple / I_dc (six-step)
BUS_PWM_FRAC = (0.005, 0.02, 0.08)            # PWM ripple reaching the DC leads / chopped amplitude
ESC_LOOP_M2 = (0.5e-4, 1.5e-4, 4e-4)          # bridge-capacitor commutation loop area per motor
LEAD_TWIST = (0.3, 0.7, 1.0)                  # effective-area factor of the phase-lead bundle (1 = flat ribbon)
WINDING_K = (1e-4, 1e-3, 5e-3)                # net winding dipole / (total coil ampere-turn-area), placeholder
TRIPLET = lambda f: tuple(f(i) for i in range(3))

def hover_point(c):
    T_g = c['auw_kg'] * 1000 / c['n_motors']
    if c['ref'] is not None:
        Tr, rpm_r, I_r, V_r = c['ref']
        rpm = rpm_r * np.sqrt(T_g / Tr)
        P = I_r * V_r * (T_g / Tr) ** 1.5
    else:   # momentum theory with figure of merit 0.7, drivetrain efficiency 0.85, C_T = 0.10
        T = T_g / 1000 * 9.81; A = np.pi * (c['prop_d_m'] / 2) ** 2
        P = T ** 1.5 / np.sqrt(2 * 1.225 * A) / 0.7 / 0.85
        n = np.sqrt(T / (0.10 * 1.225 * c['prop_d_m'] ** 4)); rpm = 60 * n
    I_dc = P / c['v_batt']
    omega = 2 * np.pi * rpm / 60
    Ke = 60 / (2 * np.pi * c['kv'])                           # V*s/rad, line-line
    tau = 0.85 * P / omega                                    # shaft torque
    duty = min(0.95, Ke * omega / c['v_batt'])                # modulation ~ back-EMF / V_batt
    if c['drive'] == 'six-step':
        I_flat = tau / Ke                                     # conducting phase current
        I1 = 2 * np.sqrt(3) / np.pi * I_flat                  # fundamental of 120-deg quasi-square wave
    else:
        I1 = tau / (np.sqrt(3) / 2 * Ke)                      # sinusoidal phase current peak
        I_flat = I1
    return dict(thrust_g=T_g, rpm=rpm, f_mech=rpm / 60, f_e=c['pole_pairs'] * rpm / 60, P_W=P, I_dc=I_dc,
                tau=tau, duty=duty, I1=I1, I_flat=I_flat)

def lines_for(c):
    op = hover_point(c); fe, fm = op['f_e'], op['f_mech']
    L = []    # (frequency, source, (lo, nom, hi) moment amplitude A m^2, polarization)
    # rotor residual dipole
    w_arc = np.pi * (c['stator_d_mm'] + 2) / c['n_mag'] * 0.8 * 1e-3
    v_mag = w_arc * (c['stator_h_mm'] + 1) * 1e-3 * c['mag_t_mm'] * 1e-3
    m_mag = MAG_BR / MU0 * v_mag
    L.append((fm, 'rotor residual', TRIPLET(lambda i: ROTOR_IMBALANCE[i] * BELL_SHIELD[i] * m_mag * np.sqrt(c['n_mag'])), 'rotating'))
    # phase leads: three-conductor ribbon, net moment L*pitch*sqrt(3)*I_n
    area = TRIPLET(lambda i: c['lead_len_m'][i] * c['lead_pitch_m'][i] * LEAD_TWIST[i])
    L.append((fe, 'phase leads', TRIPLET(lambda i: area[i] * np.sqrt(3) * op['I1']), 'linear'))
    for n in (5, 7, 11, 13):
        if c['drive'] == 'six-step':
            amp = TRIPLET(lambda i: area[i] * np.sqrt(3) * op['I1'] / n ** SIXSTEP_HARM['roll'][i])
        else:
            if n > 7:
                continue
            amp = TRIPLET(lambda i: area[i] * np.sqrt(3) * op['I1'] * FOC_THD[i])
        L.append((n * fe, 'phase leads', amp, 'linear'))
    # PWM ripple in the phase leads: triangular ripple, fundamental ~ (8/pi^2) * dI_pp/2
    d = op['duty']
    dIpp = TRIPLET(lambda i: c['v_batt'] * d * (1 - d) / (c['L_ll_H'][2 - i] * c['f_pwm']))
    L.append((c['f_pwm'], 'phase leads (PWM ripple)', TRIPLET(lambda i: area[i] * (8 / np.pi ** 2) * dIpp[i] / 2), 'linear'))
    # ESC switching loop: chopped DC-link current, fundamental (2/pi) I sin(pi d)
    chop = 2 / np.pi * op['I_flat'] * np.sin(np.pi * d)
    L.append((c['f_pwm'], 'ESC switching loop', TRIPLET(lambda i: ESC_LOOP_M2[i] * chop), 'linear'))
    # DC bus / power leads
    bus_area = TRIPLET(lambda i: c['bus_len_m'][i] * c['bus_pitch_m'][i])
    if c['drive'] == 'six-step':
        L.append((6 * fe, 'DC bus', TRIPLET(lambda i: bus_area[i] * RIPPLE_6FE[i] * op['I_dc']), 'linear'))
    L.append((c['f_pwm'], 'DC bus (PWM)', TRIPLET(lambda i: bus_area[i] * BUS_PWM_FRAC[i] * chop), 'linear'))
    # winding net dipole (placeholder until step 2): total coil ampere-turn-area * K
    turns = 60 / (2 * np.pi * c['kv']) / (2 * 2 * c['pole_pairs'] * (c['stator_h_mm'] * 1e-3) * (c['stator_d_mm'] * 1e-3 / 2) * 0.9)
    tooth_area = (np.pi * c['stator_d_mm'] * 1e-3 / (1.5 * c['n_mag'])) * c['stator_h_mm'] * 1e-3
    L.append((fe, 'winding (placeholder)', TRIPLET(lambda i: WINDING_K[i] * turns * tooth_area * op['I1'] * 1.5 * c['n_mag']), 'rotating'))
    return op, L

def b_at(m, r=1.0):
    return np.sqrt(2 / 3) * 1e-7 * m / r ** 3

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
    from range_budget import max_range, wire_loop_noise
    out = {}
    print('Equivalent-source drone ELF/VLF model, step 1 (per motor; lo / nominal / hi)')
    for name, c in CLASSES.items():
        op, L = lines_for(c)
        print(f'\n=== {name}: {c["motor"]}, AUW {c["auw_kg"]} kg, {c["drive"]}, PWM {c["f_pwm"]/1e3:.0f} kHz, ESC {c["esc"]} ===')
        print(f'  hover: {op["thrust_g"]:.0f} g/motor, {op["rpm"]:.0f} rpm, f_mech {op["f_mech"]:.0f} Hz, f_e {op["f_e"]:.0f} Hz, '
              f'P {op["P_W"]:.0f} W/motor, I_dc {op["I_dc"]:.2f} A, phase I1 {op["I1"]:.2f} A, duty {op["duty"]:.2f}')
        rows = []
        for f, src, m, pol in sorted(L, key=lambda x: x[0]):
            nb, _ = wire_loop_noise(max(f, 100.0))
            rr = [max_range(m[1], np.hypot(na, nb), 60, 60) for na in (0.3e-12, 3e-12)]
            rows.append(dict(f=f, source=src, m=m, pol=pol, B1m_nT=[b_at(x) * 1e9 for x in m], range60s_m=rr))
            print(f'  {f:9.0f} Hz  {src:26s} m = {m[0]:.1e} / {m[1]:.1e} / {m[2]:.1e} A m^2 | B(1 m) {b_at(m[0])*1e9:6.3f} / '
                  f'{b_at(m[1])*1e9:6.3f} / {b_at(m[2])*1e9:6.3f} nT | 60 s coherent range (nominal) {rr[0]:5.1f} m @0.3 pT, {rr[1]:5.1f} m @3 pT')
        out[name] = dict(op=op, lines=rows)
    json.dump(out, open(os.path.join(HERE, 'drone_source_results.json'), 'w'), indent=1, default=float)
