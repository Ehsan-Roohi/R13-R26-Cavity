# Time-dependent R13 and R26 solutions and cavity start-up movies (25 November 2026)

Material of appendix "Time-dependent R13 and R26 solutions and the start-up of the cavity flow" of the JFM manuscript
"Heat-flux and higher-moment validation in a rarefied cavity".  Private local paths are replaced by placeholders
(`<project>`, `<revision workspace>`, `<cluster workspace>`, ...); set them to your own locations before running.

* `r13/r13_transient.py` - time-dependent R13: the steady operator of `r13-r26-cavity/code` (assemble_system with the matched
  Maxwell production) plus the time-derivative coefficients diag(1, rho, rho, 3 rho/2, 1 x 5, 0 x 8); linearly implicit BDF2,
  nested-dissection LU reused with defect correction.  `R13_LID_PROFILE=gauss R13_LID_WIDTH=0.0316227766016838` selects the
  smooth lid.  Records: `*.npz.json` (rate history), `DT_CHECK_K20.json` (dt versus dt/2).
* `r26/r26_transient_worker.py` - time-dependent R26 on the steady R26 operator of `r13-r26-refinement-20261104/solver/r26/source`:
  scaled steady residual + d_t (rho, rho u, rho v, rho theta, other moments) by BDF2, modified Newton with the finite-difference
  Jacobian of the steady solver (optionally centred), lid speed ramped over 1 ns, adaptive sub-steps.  `make_packet*.py` build
  the cluster jobs (configurations of the runs in the paper).
* `dsmc/` - SPARTA start-up inputs (gas at rest, 2e5 steps, moments averaged over 500-step blocks; 8 seeds per case, seed index
  in the file name), the packet builders and the seed-average script.
* `movies/` - scripts of the eight movies and of figure `fig_transient_history`; `TRANSIENT_HISTORY.json` holds its numbers.

Results quoted in the paper: R13 (sharp lid) reaches its steady states to 1.6e-8 (Kn 0.2) and 8.6e-11 (Kn 0.05); R26 (smooth lid,
Kn 0.05, 70x70) reaches its steady root to 1.6e-6; R13 with the smooth lid at Kn 0.2 develops a growing mode at the upstream lid
corner after about 43 ns; time-dependent R26 did not pass the lid start-up with the sharp lid or at Kn 0.2.
