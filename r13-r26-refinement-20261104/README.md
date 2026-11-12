# Lid-driven cavity grid refinement

This dated update supplements the protected article reproduction with the matched-wall R13 grid sequence and the independently replayed R26 continuation checkpoints for Kn = 0.05 and 0.20.

## Latest states (11 November 2026)

The [latest-state comparison](figures/latest-20261111/) adds the newest audited grid states. Its data, metrics, source hashes and rendering script are in the same folder.

**R26, Kn = 0.20.** N120 global step 36 (job 66879100, independent RMS 1.07 × 10⁻³) and N180 global step 8 (job 66884487, independent RMS 4.33 × 10⁻³) each passed the nine original step guards. Both remain above the final nonlinear tolerance and are provisional. They are compared with the N90 paper state and the matched SPARTA reference used in the [all-grid comparisons](figures/dsmc-grid-comparisons/).

- On the same 19 near-lid positions (x/L = 0.5, 0.85 ≤ y/L ≤ 0.97), the RMS difference in q<sub>x</sub> from DSMC is 0.004112 for N90, 0.004138 for N120 step 36 and 0.004184 for N180 step 8. The N180 − N120 difference is 8.8 × 10⁻⁵ RMS.
- On both centerlines, the N180 − N120 RMS differences in q<sub>x</sub> and q<sub>y</sub> are at most 3.7 × 10⁻⁵. The corresponding R26 − DSMC differences are 1.2 × 10⁻³ to 4.9 × 10⁻³.
- Over the whole cavity, the RMS change of q<sub>x</sub> from N120 to N180 is 4.2 × 10⁻⁴, compared with 7.1 × 10⁻³ for N120 − DSMC. The change is concentrated at the two lid corners ([signed maps](figures/latest-20261111/R26_KN020_qx_change_maps_latest.png)).
- Between N120 and N180, the fourth-order tensor A changes by 0.19 % (relative L<sup>2</sup>) over 0.05 ≤ x/L, y/L ≤ 0.95 and by 1.6 % over the whole cavity, mostly at the lid corners.

These are unconverged iterates with a shared starting state. N120 was started from the N90 paper state and N180 from an N120 iterate, so part of their agreement is inherited. Their largest residuals are 0.16 (N120) and 0.74 (N180), and the N180 run has had eight steps. From its first to its 36th step the N120 residual RMS fell about twentyfold (2.1 × 10⁻² to 1.07 × 10⁻³) while A changed by 0.29 % over the same interior region; from its first to its eighth step the N180 residual RMS fell about ninefold (3.8 × 10⁻² to 4.33 × 10⁻³) with a 0.09 % change. At these saved states, the change from N120 to N180 is a small fraction of the difference from DSMC, and the near-lid departure from DSMC does not decrease. The differences combine grid, iteration and starting-state effects; they do not establish grid convergence or bound the discretisation error.

**R13, Kn = 0.05.** N100 and N120 meet the practical tolerance, and N180 accepted4 (job 66936129, independent RMS 2.0 × 10⁻⁵) passes all 14 practical checks. The reference pools seven SPARTA lineages from step 100,000 (one to 5,000,000 steps and six to 1,980,000; 1,617,300 samples per cell). Over 0.1 ≤ s/L ≤ 0.9 on both centerlines, the N180 − N120 RMS differences in heat flux are at most 2.2 × 10⁻⁴. Near the lid, on x/L = 0.5 with 0.85 ≤ y/L ≤ 0.985, the largest |q<sub>x</sub> − q<sub>x,DSMC</sub>| falls from 9.9 × 10⁻³ at N120 to 5.3 × 10⁻³ at N180, while the N120-to-N180 change itself reaches 4.6 × 10⁻³. Near the right wall, the corresponding values for q<sub>y</sub> on y/L = 0.5 are 2.0 × 10⁻³, 1.1 × 10⁻³ and 1.1 × 10⁻³. The wall-adjacent R13 heat flux is therefore still grid-dependent at N180 ([near-wall profiles](figures/latest-20261111/R13_KN005_near_wall_q_latest.png)).

## Figures with DSMC

- [Latest states, 11 November 2026](figures/latest-20261111/): R26 N90, N120 step 36 and N180 step 8 at Kn = 0.20, and R13 N100, N120 and N180 at Kn = 0.05.
- [All-grid centerline comparisons](figures/dsmc-grid-comparisons/) cover both Knudsen numbers and include stress, higher moments and Δ where applicable. They use the audited provisional N120 step 6 at Kn = 0.20.
- [Near-lid R26 heat flux at Kn = 0.20](figures/dsmc-grid-comparisons/R26_KN020_near_lid_qx.png) and [RMS differences for all eight grids](figures/dsmc-grid-comparisons/near_lid_qx.csv) compare the same 19 positions.
- [November 3 paper figures](figures/paper-20261103/) collect the prepared physical comparisons and numerical diagnostics.

## R13

The R13 solver uses a conservative finite-volume cavity residual with Maxwell wall conditions and a global mass constraint. The bordered defect-Newton solve uses finite-difference Jacobian-vector products and right-preconditioned GMRES. The continuation advances from a compatible grid solution and checks the final residual and physical consistency independently. The new grid uses the same wall and corner conventions. The final states and independent residual history are under [results/r13](results/r13/).

| Kn | Converged grids | Provisional grids |
|---|---|---|
| 0.05 | N40, N60, N80; N100, N120 and N180 (accepted4) at the practical tolerance | — |
| 0.20 | N40, N60 | N80 (latest audited checkpoint accepted15: independent RMS 1.45 × 10⁻³, 10 of 14 practical checks; not included here) |

- [Kn = 0.05 heat flux](figures/R13_KN005_N40_N60_N80_heat_flux.png) and [all fields](figures/R13_KN005_N40_N60_N80_converged.pdf)
- [Kn = 0.20 heat flux](figures/R13_KN020_heat_flux.png), with the provisional N80 curve labeled
- [N60 profiles for both Knudsen numbers](figures/R13_all_fields_N60.pdf)

The curves sample physical centerlines at each grid's native axial nodes. They do not extrapolate to the wall. Agreement in the interior and differences near the moving lid can be read separately.

## R26

Continuation assembles a grid-specific structured finite-difference Jacobian with an exact analytic mass row. Candidate steps are checked against the scaled nonlinear residual and the raw merit that restores the omitted continuity equation and retains the mass residual. The bounded N120 path uses column-scaled LSMR and checks the direction and line search in the original system. Positivity, conservation diagnostics and the original convergence thresholds remain separate checks.

The independently replayed N90 (Kn = 0.05) and N120 (Kn = 0.20) states are **provisional**. Both replayed candidates decreased the scaled and restored raw merits. The original N120 entry in the case index is the earlier checkpoint. The independently replayed step 5 used in the new profiles passed the held-out continuity limit and remains above the full nonlinear residual tolerance. The [case index](results/cases.json) and [R26 audits](results/r26/) identify the exact states. The [N90 field comparison](figures/R26_KN005_N90_all_fields.pdf) shows earlier provisional iterates. The [historical R26 solver history](figures/R26_historical_solver_history.png) provides separate context and does not establish fine-grid convergence. The latest Kn = 0.20 states, N120 step 36 and N180 step 8, are in [results/r26](results/r26/) with a [guard summary](results/r26/KN020_latest_guards.json), and are described under [Latest states](#latest-states-11-november-2026).

## Additional grid comparisons

- [R13 audited residual histories](figures/R13_audited_convergence.png) show the iteration histories by grid and Knudsen number.
- [R26 heat flux](figures/R26_KN020_N28_N120_heat_flux.png) and [velocity and temperature](figures/R26_KN020_N28_N120_flow_profiles.png) at Kn = 0.20 compare the accepted N28 profiles and provisional N120 profiles with the same SPARTA DSMC reference. The plots retain the coordinate coverage of each data source.
- [R26 archived grids at Kn = 0.05](figures/R26_KN005_archived_grid_profiles.png) and [at Kn = 0.20](figures/R26_KN020_archived_grid_profiles.png) show heat flux and the composite fourth-order tensor. These are saved-state sensitivity comparisons. Some iterations are incomplete and equivalence of the discrete operators across the historical grids remains unverified.

The earlier N28–N120 profiles immediately above use audited continuation step 5. Its checkpoint SHA-256 is `f12c821ab353bab612904aa586d0276d57059f92504a05bd1973d28e228205bd`. The N28 and SPARTA curves use the matched centerline table with SHA-256 `c225cb094a1a8b9c3b9f28b3bba9f7ba461643dbc29084b88f6234b461515535`.

## Code and data

[solver/r13](solver/r13/) contains the matched-grid Maxwell solver and its grid and restart workers. [solver/r26](solver/r26/) contains the pinned R26 physics modules and the captured continuation workers. These Python scientific sources and the case artifacts are archived for inspection. The workers were captured from staged runs and expect their corresponding run-directory configuration and seed. The original article's standalone reproduction is available separately in the repository.

R13 state arrays use NumPy NPY. R26 NPZ checkpoints contain the physical and encoded state and residual arrays. The [case index](results/cases.json) lists units of grid resolution, independent RMS, convergence status and hashes for the archived states. [MANIFEST.sha256](MANIFEST.sha256) records the bundle checksums.
