* MFB replacement; two inversions retain overall polarity. Reference drives + inputs.
* Butterworth HP: C1=C3=C4=470n, R2=800, R5=3.6k; 199.54 Hz.
CHP1 out hp_x 470n
CHP2 hp_x hp_m 470n
RHPFB out_hp hp_m 3.6k
RHPG hp_x 0 800
XU3 vcm hp_m vcc 0 out_hp ADA4511
* Butterworth LP: C2=40n, C5=10n, R1=R4=1.13k, R3=565; 9.959 kHz.
RLP1 out_hp lp_x 1.13k
RLP2 lp_x lp_m 565
CLPFB filtered lp_m 10n
CLPG lp_x 0 40n
XU4 vcm lp_m vcc 0 filtered ADA4511
RFINAL filtered vcm 100k
CFINAL filtered 0 100p
* Extra feedback branches relative to Sallen-Key.
CHPFB out_hp hp_x 470n
RLPFB filtered lp_x 1.13k
