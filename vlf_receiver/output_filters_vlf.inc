* VLF channel output filters: same MFB Butterworth topology as output_filters.inc, capacitors rescaled.
* Butterworth HP: C1=C3=C4=9.4n, R2=800, R5=3.6k; ~10.0 kHz.
CHP1 out hp_x 9.4n
CHP2 hp_x hp_m 9.4n
RHPFB out_hp hp_m 3.6k
RHPG hp_x 0 800
XU3 vcm hp_m vcc 0 out_hp OPAMP
* Butterworth LP: C2=3.32n, C5=0.83n, R1=R4=1.13k, R3=565; ~120 kHz.
RLP1 out_hp lp_x 1.13k
RLP2 lp_x lp_m 565
CLPFB filtered lp_m 0.83n
CLPG lp_x 0 3.32n
XU4 vcm lp_m vcc 0 filtered OPAMP
RFINAL filtered vcm 100k
CFINAL filtered 0 100p
CHPFB out_hp hp_x 9.4n
RLPFB filtered lp_x 1.13k
