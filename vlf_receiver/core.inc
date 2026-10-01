* Revised overload-protected circuit; component values frozen independently of source estimates.
* 9 V, three LSBF862 JFETs in two gain stages; exact PDF-extracted model.
.include LSBF862_from_pdf.lib
.include coil_sources.lib
.param Lscale=1 Ccoil={Cestimated(Nturn)} RDRAIN=458.1 RSOURCE=47
.temp 25
VCC vcc 0 9
VEMF emf 0 DC 0 AC 1 SINE(0 {EAMP} {FREQ})
RCOIL emf rl {Rcoil(Nturn)}
LCOIL rl gate {Lcoil(Nturn)*Lscale} Rser=0
CSELF gate 0 {Ccoil}
RBLEED gate 0 10Meg
* The bleed resistor drains static charge; it is not an ESD clamp.
RGTOP vcc bias 200k
RGBOTTOM bias 0 100k
CGBIAS bias 0 10u Rser=0.1
RD vcc drain {RDRAIN}
JLOW mid gate source LSBF862_UPDATED
JUP drain bias mid LSBF862_UPDATED
RS source 0 {RSOURCE}
CS source 0 470u Rser=0.1
.options plotwinsize=0

.include coil_capacitance.lib
* Second, common-source LSBF862 stage; nominal voltage gain adjusted to retain +75 V/V overall.
CCOUP drain gate2 82n
RG2 gate2 0 20k
JSECOND drain2 gate2 source2 LSBF862_UPDATED
RS2 source2 0 47
CS2 source2 0 470u Rser=0.1
* DC supply drop with AC bypass preserves drain swing at low voltage gain.
RFILTER vcc supply2 376.211253684
CFILTER supply2 0 100u Rser=0.1
RD2 supply2 drain2 357.736675518
* Roll off the second stage above the signal band for loop stability.
CCOMP drain2 supply2 2.66889129662e-09

.include ADA4511.lib
.include bias_hints.inc
.options solver=alt itl1=1000
.param CF={table(Nturn,48,120p,56,150p,100,270p)} CINT=10n RLEAK={table(Nturn,48,2.4k,56,3k,100,6.8k)} AFORWARD=75.4 VREF=4.50000000956
.param CTOT={Ccoil+CF+12p}
.param GCRIT={Rcoil(Nturn)*CTOT/Lcoil(Nturn)+2*sqrt(CTOT/Lcoil(Nturn))}
.param RINT={table(Nturn,48,1065.38540163,56,1351.50817849,100,2584.31315529)}
VREF vcm 0 {VREF}
* Channel 1 is a non-inverting unity-gain buffer.
XU1 drain2 out vcc 0 out ADA4511
RLOAD out vcm 100k
CLOAD out 0 100p
RINT out sumi {RINT}
CINT integ sumi {CINT}
RLEAK integ sumi {RLEAK}
XU2 vcm sumi vcc 0 integ ADA4511
CF gate feed {if(Mode==0,1e-24,CF)}
RFEED feed integ {if(Mode==1,0.01,1e12)} Noiseless
RSHUNT feed vcm {if(Mode==2,0.01,1e12)} Noiseless

.include output_filters.inc

.include clamp_diode.lib
DP1 integ clampP1 1N4148
DP2 clampP1 clampP2 1N4148
DP3 clampP2 clampP3 1N4148
DP4 clampP3 clampP4 1N4148
DP5 clampP4 sumi 1N4148
DN1 sumi clampN1 1N4148
DN2 clampN1 clampN2 1N4148
DN3 clampN2 clampN3 1N4148
DN4 clampN3 clampN4 1N4148
DN5 clampN4 integ 1N4148
