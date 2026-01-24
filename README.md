# Physics-Constrained Geometric Compensation via Modal Analysis

This repository implements a physics-constrained geometric compensation framework based on structural modal analysis.

Instead of defining compensation directly in the full geometric displacement field (direct inversion),
we project observed manufacturing deformation onto a reduced, structure-aligned modal subspace and
apply inversion only to the structure-consistent components. This improves robustness against
non-physical noise, sparsity, and scaling artifacts commonly encountered in scan/simulation-driven workflows.

## Key scripts

- `scripts/export_compensation_stl.py`  
  Export compensated STL meshes using (1) direct inversion and (2) single-mode modal compensation (e.g., mode 7).
  Also exports x10 exaggerated versions for visualization.

- `scripts/robustness_benchmark.py`  
  Monte-Carlo robustness benchmark: Raw vs Direct vs Modal under disturbances (noise / sparsity / scaling),
  with paired t-tests and summary CSV output.

- `scripts/modal_energy_mass_metric.py`  
  Mass-metric modal energy analysis: cumulative/incremental explained energy curve and CSV table.

## Inputs expected (from FEM)

- `mode/all_modes.csv` (mode shapes)
- `Msparse.txt` (mass matrix in sparse text format)
- optionally `Ksparse.txt` (stiffness matrix)
- deformation files: `x.txt`, `y.txt`, `z.txt` (or `newx.txt/newy.txt/newz.txt`)
- optional STL for export: `*.stl`
