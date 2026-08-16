# Physics-Constrained Geometric Compensation via Modal Analysis

This repository contains research code for a physics-constrained geometric compensation framework based on structural modal analysis.

Instead of applying compensation directly to the full measured displacement field, the method projects deformation into a structure-aligned modal subspace and performs compensation using the physically consistent component. The scripts also include robustness and modal-energy analyses used to compare modal compensation with direct inversion.

## Repository Contents

- `compensationgen.py`  
  Generates compensated STL meshes using direct inversion and single-mode modal compensation. It also exports 5× exaggerated meshes for visualization.

- `mcmc.py`  
  Runs Monte Carlo robustness comparisons for Raw, Direct, and Modal responses under noise, sparsity, and scaling disturbances, including paired statistical tests.

- `vibfitting.py`  
  Computes mass-metric modal fitting and cumulative/incremental explained-energy curves.

## Requirements

Python dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## FEM Inputs

The analysis scripts expect FEM exports to be placed relative to the repository root. These project-specific inputs are **not included** in this repository.

Common inputs include:

```text
mode/all_modes.csv     modal shapes
Msparse.txt            sparse mass matrix
Ksparse.txt            sparse stiffness matrix (robustness analysis)
x.txt, y.txt, z.txt    XYZ deformation exports for modal-energy analysis
newx.txt, newy.txt,
newz.txt                XYZ deformation exports for compensation/robustness
newplat.stl             nominal STL mesh for compensation export
```

The scripts resolve paths relative to their own repository location, so they can be launched from another working directory without editing absolute paths.

## Usage

Modal energy analysis:

```bash
python vibfitting.py
```

Robustness benchmark:

```bash
python mcmc.py
```

Compensation mesh export:

```bash
python compensationgen.py
```

Generated CSV, PNG, and compensated STL outputs are excluded from version control by `.gitignore`.

## Notes

- Modal numbering follows the FEM export convention used in the research workflow; the compensation example uses mode 7 as the first physical mode.
- `compensationgen.py` currently uses an exaggeration factor of 5 for visualization-only STL outputs.
- This repository contains the computational implementation; FEM source models and simulation exports must be supplied separately.
