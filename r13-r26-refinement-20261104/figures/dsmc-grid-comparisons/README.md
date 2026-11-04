# Mesh and DSMC centerline comparisons

All the cavity grids with traceable saved profiles for this release are overlaid on the horizontal and vertical centerlines. The colors and markers identify grids. The legend distinguishes archived, converged and provisional states.

| Model | Kn | Included grids | Profiles |
|---|---:|---|---|
| R13 | 0.05 | N40, N60, N80 | [All panels](R13_KN005_all_profiles.pdf) · [Heat flux](R13_KN005_heat_flux.png) |
| R13 | 0.20 | N40, N60, N80 (provisional) | [All panels](R13_KN020_all_profiles.pdf) · [Heat flux](R13_KN020_heat_flux.png) |
| R26 | 0.05 | N32, N40, N60, N90 (provisional) | [All panels](R26_KN005_all_profiles.pdf) · [Heat flux](R26_KN005_heat_flux.png) · [Third moments](R26_KN005_third_moments.png) · [Delta and A](R26_KN005_delta_and_A.png) |
| R26 | 0.20 | N20, N28, N40, N45, N50, N60, N90, N120 step 6 (provisional) | [All panels](R26_KN020_all_profiles.pdf) · [Flow](R26_KN020_flow.png) · [Heat flux](R26_KN020_heat_flux.png) · [Stress](R26_KN020_stress.png) · [Third moments](R26_KN020_third_moments.png) · [Fourth moments](R26_KN020_fourth_moments.png) · [Delta and A](R26_KN020_delta_and_A.png) |

R26 includes the in-plane third moments, deviatoric fourth moments, the scalar fourth moment Δ, and the reduced tensor `A = R + Δ I / 3`. The out-of-plane diagonal components of stress and `R` follow from the trace-free constraint. The R13 panels cover density, velocity, temperature, heat flux and deviatoric stress.

## Near the moving lid at Kn = 0.20

The [zoomed vertical centerline](R26_KN020_near_lid_qx.png) shows signed horizontal heat flux over `0.85 ≤ y/L ≤ 0.97`. On the same 19 DSMC positions, the RMS difference is `0.005483` for N28, `0.004112` for the archived N90 and `0.004148` for provisional N120 step 6. The N120 difference is about 24.3% below N28. The historical grids include unfinished states and may use different discrete operators, so the comparisons describe the saved profiles. [All eight grid values](near_lid_qx.csv)

## Reference and coordinates

- At Kn = 0.05, all panels use the cell-registered SPARTA LAB56 data over `(100,000, 5,000,000]`, excluding the verified all-zero block ending at step 560,000.
- At Kn = 0.20, density, velocity, temperature and heat flux use the matched historical SPARTA reference from the earlier N28–N120 comparison. Stress and higher moments use the separately identified, cell-registered SPARTA LAB56 data over `(2,000,000, 8,000,000]` from seed `155921`. Both LAB56 reconstructions were checked against the archived density, velocity, temperature and heat-flux fields. The windows and curve labels are retained in the figures.
- For each source, the cuts use its native positions along the line and linear interpolation in the transverse direction to `x/L = 0.5` or `y/L = 0.5`. The RMS comparison interpolates the model only within its covered range to the DSMC positions. The curves retain sign and sample coverage. Single-lineage references do not carry an estimated between-run uncertainty.
- N120 step 6 passed the independent checkpoint physical and continuity checks and remains above the final nonlinear tolerance. The N90 curve at Kn = 0.05 also represents an audited provisional checkpoint. The earlier figures and the case index retain their original checkpoint identities.

[Plot data](centerline_data.csv) · [Source hashes](sources.json) · [Rendering script](render_profiles.py) · [November 3 paper figures](../paper-20261103/)
