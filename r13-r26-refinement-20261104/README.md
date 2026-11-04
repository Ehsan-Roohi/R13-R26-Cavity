# Lid-driven cavity grid refinement

This dated update supplements the protected article reproduction with the matched-wall R13 grid sequence and the independently replayed R26 continuation checkpoints for Kn = 0.05 and 0.20.

## R13

The R13 solver uses a conservative finite-volume cavity residual with Maxwell wall conditions and a global mass constraint. The bordered defect-Newton solve uses finite-difference Jacobian-vector products and right-preconditioned GMRES. The continuation advances from a compatible grid solution and checks the final residual and physical consistency independently. The new grid uses the same wall and corner conventions. The final states and independent residual history are under [results/r13](results/r13/).

| Kn | Converged grids | Provisional grids |
|---|---|---|
| 0.05 | N40, N60, N80 | N100 and N120 are being evaluated |
| 0.20 | N40, N60 | N80 |

- [Kn = 0.05 heat flux](figures/R13_KN005_N40_N60_N80_heat_flux.png) and [all fields](figures/R13_KN005_N40_N60_N80_converged.pdf)
- [Kn = 0.20 heat flux](figures/R13_KN020_heat_flux.png), with the provisional N80 curve labeled
- [N60 profiles for both Knudsen numbers](figures/R13_all_fields_N60.pdf)

The curves sample physical centerlines at each grid's native axial nodes. They do not extrapolate to the wall. Agreement in the interior and differences near the moving lid can be read separately.

## R26

Continuation assembles a grid-specific structured finite-difference Jacobian with an exact analytic mass row. Candidate steps are checked against the scaled nonlinear residual and the raw merit that restores the omitted continuity equation and retains the mass residual. The bounded N120 path uses column-scaled LSMR and checks the direction and line search in the original system. Positivity, conservation diagnostics and the original convergence thresholds remain separate checks.

The independently replayed N90 (Kn = 0.05) and N120 (Kn = 0.20) states are **provisional**. Both replayed candidates decreased the scaled and restored raw merits. The N120 replay still missed the original residual and held-out continuity limits. The [case index](results/cases.json) and [R26 audits](results/r26/) identify the exact states. The [N90 field comparison](figures/R26_KN005_N90_all_fields.pdf) shows earlier provisional iterates. The [historical R26 solver history](figures/R26_historical_solver_history.png) provides separate context and does not establish fine-grid convergence.

## Code and data

[solver/r13](solver/r13/) contains the matched-grid Maxwell solver and its grid and restart workers. [solver/r26](solver/r26/) contains the pinned R26 physics modules and the captured continuation workers. These Python scientific sources and the case artifacts are archived for inspection. The workers were captured from staged runs and expect their corresponding run-directory configuration and seed. The original article's standalone reproduction is available separately in the repository.

R13 state arrays use NumPy NPY. R26 NPZ checkpoints contain the physical and encoded state and residual arrays. The [case index](results/cases.json) lists units of grid resolution, independent RMS, convergence status and hashes for the archived states. [MANIFEST.sha256](MANIFEST.sha256) records the bundle checksums.
