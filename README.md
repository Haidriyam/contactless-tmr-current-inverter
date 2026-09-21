![Contactless TMR Current Inverter CI](https://github.com/Haidriyam/contactless-tmr-current-inverter/actions/workflows/devsecops-ci.yml/badge.svg)

# Contactless Multi-Phase Current Metrology via Biot-Savart Inversion

A non-invasive, contactless current metrology and spatial field reconstruction testbed. Using peripheral Tunnel Magnetoresistance (TMR) sensor arrays disposed symmetrically around multicore power transmission conductors, it solves the inverse magnetostatic Biot-Savart problem via Tikhonov-regularized pseudo-inversion, achieving accurate three-phase current recovery under sensor noise and spatial cross-talk without breaking the circuit.

```text
       [ TMR Sensor Ring (M=8) ]
                 ( S1 )
          ( S8 )   ▲   ( S2 )
             │   ┌─┴─┐   │
             │   │Ia │   │        [ Measured B_tangential ]
   ( S7 ) ───┼───┤Ib ├───┼─── ( S3 )          │
             │   │Ic │   │                    ▼
             │   └───┘   │     [ Tikhonov Inversion Engine ]
          ( S6 )       ( S4 )                 │
                 ( S5 )                       ▼
                                  [ Reconstructed Ia, Ib, Ic ]