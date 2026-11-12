# Latest grid states, 11 November 2026

| Model | Kn | States | Reference | Figures |
|---|---:|---|---|---|
| R26 | 0.20 | N90 paper state; N120 step 36 and N180 step 8 (provisional) | SPARTA matched historical fields | [Heat flux](R26_KN020_heat_flux_latest.png) · [Velocity and temperature](R26_KN020_flow_latest.png) · [Near-lid q<sub>x</sub>](R26_KN020_near_lid_qx_latest.png) · [q<sub>x</sub> change maps](R26_KN020_qx_change_maps_latest.png) |
| R13 | 0.05 | N100, N120 and N180 accepted4 | Seven SPARTA lineages pooled from step 100,000 (one to 5,000,000 steps and six to 1,980,000; 1,617,300 samples per cell) | [Heat flux](R13_KN005_heat_flux_latest.png) · [Near-wall heat flux](R13_KN005_near_wall_q_latest.png) |

Each figure is also available as a vector PDF with the same name.

- The near-lid figure plots q<sub>x</sub> on x/L = 0.5 for 0.8 ≤ y/L ≤ 1. Its second panel shows the signed differences N120 − DSMC, N120 − N90 and N180 − N120 along the full line.
- The maps show the signed q<sub>x</sub> difference N180 − N120 (left) and N120 − DSMC (right) on the DSMC cell centres with one colour scale. The scale is the 99th percentile of the right-hand map, so values beyond it are clipped.
- The near-wall R13 figure shows q<sub>x</sub> on x/L = 0.5 near the lid and q<sub>y</sub> on y/L = 0.5 near the right wall.

## Data

- [latest_centerlines.csv](latest_centerlines.csv) holds every plotted centerline in the column layout of [centerline_data.csv](../dsmc-grid-comparisons/centerline_data.csv). All series use ρ/ρ<sub>0</sub>, u/U<sub>w</sub>, T/T<sub>w</sub> and q/(ρ<sub>0</sub>c<sub>0</sub><sup>3</sup>) with the DSMC c<sub>0</sub>. Model fields are converted with the same factor as the all-grid comparisons.
- [R26_KN020_qx_difference_maps.npz](R26_KN020_qx_difference_maps.npz) holds the two mapped q<sub>x</sub> differences.
- [metrics.json](metrics.json) lists RMS and maximum differences by line and window, and whole-cavity RMS values at Kn = 0.20. Models are interpolated linearly to the DSMC positions inside their own coverage, without extrapolation or smoothing.
- [sources.json](sources.json) gives the SHA-256 of every input. The new model states are in [results/r13](../../results/r13/) and [results/r26](../../results/r26/), and are listed in the [case index](../../results/cases.json).

The R13 profiles recomputed from these states match the previously reviewed profiles exactly. The R26 states are not converged roots, so their differences combine grid and iteration effects and are not error bounds.

To redraw the figures from the data in this folder (requires NumPy and Matplotlib):

```bash
python render_latest.py
```
