# -*- coding: utf-8 -*-
"""
Mass-metric Modal Energy Analysis (XYZ displacement)
---------------------------------------------------
- Modal coefficients obtained via mass-weighted least squares
- Explained energy evaluated in the mass metric
- Prints detailed numeric results (per mode)
- Saves CSV tables and plots

Author: Daniel + ChatGPT
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
import re, time
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix


# ===================== CONFIG ===================== #
BASE_PATH   = Path(r"C:\Users\31746\OneDrive\Desktop\New folder")
MODES_CSV   = BASE_PATH / "mode" / "all_modes.csv"

MASS_FILE   = BASE_PATH / "Msparse.txt"

X_DEFORM_FILE = BASE_PATH / "x.txt"
Y_DEFORM_FILE = BASE_PATH / "y.txt"
Z_DEFORM_FILE = BASE_PATH / "z.txt"

# 模态频率（Hz），长度 = 模态总数（含刚体）
FREQS = np.array([
    0.0,0.0,0.0,
    9.1574e-003,9.6981e-003,9.9905e-003,
    149.19,221.48,414.22,476.29,797.42,804.18,
    984.14,1051.4,1213.3,1337.0,1370.3,1736.0,1740.3,2025.1
])


# ===================== IO UTILS ===================== #
def read_modes_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]
    df["Direction"] = df["Direction"].astype(str).str.strip().str.upper()
    df["Node"]      = pd.to_numeric(df["Node"], errors='coerce').round().astype(int)
    df["Udir"]      = pd.to_numeric(df["Udir"], errors='coerce')
    return df


def build_phi_from_modes(df: pd.DataFrame, num_modes_total: int):
    nodes_sorted = np.sort(df["Node"].unique())
    N = len(nodes_sorted)

    Ux = np.zeros((N, num_modes_total))
    Uy = np.zeros_like(Ux)
    Uz = np.zeros_like(Ux)

    for m in range(1, num_modes_total + 1):
        for lab, Umat in (("X", Ux), ("Y", Uy), ("Z", Uz)):
            sub = df[(df["Mode"] == m) & (df["Direction"] == lab)]
            if not sub.empty:
                Umat[:, m-1] = (
                    sub.set_index("Node")
                       .reindex(nodes_sorted)["Udir"]
                       .fillna(0.0)
                       .to_numpy()
                )

    Phi = np.zeros((3*N, num_modes_total))
    Phi[0::3, :], Phi[1::3, :], Phi[2::3, :] = Ux, Uy, Uz
    return Phi, nodes_sorted


def read_sparse_matrix_txt(path: Path) -> csr_matrix:
    rows, cols, vals = [], [], []
    pat = re.compile(
        r"\[\s*(\d+)\s*,\s*(\d+)\s*\]:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
    )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.replace("D","E").replace("d","E")
            for m in pat.finditer(line):
                i, j, v = int(m.group(1))-1, int(m.group(2))-1, float(m.group(3))
                rows.append(i); cols.append(j); vals.append(v)
    if not rows:
        raise RuntimeError(f"No entries parsed from {path}")
    n = max(max(rows), max(cols)) + 1
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def read_deform_component(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [c.strip().replace("\ufeff","") for c in df.columns]
    df = df.rename(columns={
        "Node Number":"Node",
        "Directional Deformation (mm)":"Udir",
        "Directional Deformation":"Udir",
        "Deformation(mm)":"Udir"
    })
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce").round().astype(int)
    df["Udir"] = pd.to_numeric(df["Udir"], errors="coerce")
    return df[["Node","Udir"]]


def build_disp_from_xyz(x_file,y_file,z_file,nodes):
    dfx = read_deform_component(x_file).rename(columns={"Udir":"UX"})
    dfy = read_deform_component(y_file).rename(columns={"Udir":"UY"})
    dfz = read_deform_component(z_file).rename(columns={"Udir":"UZ"})

    df = (dfx.set_index("Node")
            .join(dfy.set_index("Node"), how="outer")
            .join(dfz.set_index("Node"), how="outer"))
    df = df.reindex(nodes).fillna(0.0)
    df.reset_index(inplace=True)
    return df


def remove_rigid_translation(D):
    D3 = D.reshape(-1,3)
    mean_disp = D3.mean(axis=0)
    print(f"Removed rigid translation (mm): {mean_disp}")
    return (D3 - mean_disp).reshape(-1)


# ===================== CORE MATH ===================== #
def fit_mass_energy(D, Phi, M):
    """
    Mass-weighted least squares fit + mass-metric explained energy
    """
    A = Phi.T @ (M @ Phi)
    b = Phi.T @ (M @ D)

    eps = 1e-12 * abs(np.trace(A)) / max(1, A.shape[0])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(A + eps*np.eye(A.shape[0]), b, rcond=None)

    recon = Phi @ c
    diff  = D - recon

    num = float(diff @ (M @ diff))
    den = float(D @ (M @ D))
    expl = (1 - num/den) * 100 if den > 1e-30 else 100.0
    return expl, c, num, den


def plot_energy(freqs, cumulative, incremental, title, out_path):
    plt.figure(figsize=(9,5))
    x = np.arange(1, len(freqs) + 1)
    plt.plot(x, cumulative, "o-", label="Cumulative explained energy(%)")
    plt.bar(x, incremental, alpha=0.5, label="Incremental explained energy(%)")
    plt.xticks(x[::2])   # 每隔 2 个 mode 显示
    plt.ylim(0, 100)
    plt.xlabel("Mode number")
    plt.ylabel("Explained mass-energy (%)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ===================== MAIN ===================== #
def main():
    t0 = time.time()
    print("="*90)
    print(" Mass-metric Modal Energy Analysis (XYZ displacement) ")
    print("="*90)

    # Load modes
    df_modes = read_modes_csv(MODES_CSV)
    Phi_all, nodes = build_phi_from_modes(df_modes, len(FREQS))
    print(f"Nodes: {len(nodes)}, DOF: {Phi_all.shape[0]}")

    # Load mass matrix
    M = read_sparse_matrix_txt(MASS_FILE)
    ndof = min(M.shape[0], Phi_all.shape[0])
    M = M[:ndof, :ndof]
    Phi_all = Phi_all[:ndof, :]

    # Load displacement
    df_disp = build_disp_from_xyz(X_DEFORM_FILE, Y_DEFORM_FILE, Z_DEFORM_FILE, nodes)
    D = df_disp[['UX','UY','UZ']].to_numpy().reshape(-1)[:ndof]
    D = remove_rigid_translation(D)

    total_M_energy = float(D @ (M @ D))

    print("\n k | freq(Hz) | cumulative(%) | incremental(%) | residual_M | total_M")
    print("-"*90)

    cumulative = []
    incremental = []
    prevE = 0.0

    for k in range(1, len(FREQS)+1):
        E, c, res_M, _ = fit_mass_energy(D, Phi_all[:, :k], M)
        dE = E - prevE

        cumulative.append(E)
        incremental.append(dE)

        print(f"{k:2d} | "
              f"{FREQS[k-1]:8.2f} | "
              f"{E:12.4f} | "
              f"{dE:14.4f} | "
              f"{res_M:10.3e} | "
              f"{total_M_energy:10.3e}")

        prevE = E

    # Save CSV
    df_out = pd.DataFrame({
        "Mode": np.arange(1, len(FREQS)+1),
        "Frequency(Hz)": FREQS,
        "IncrementalEnergy_M(%)": incremental,
        "CumulativeEnergy_M(%)": cumulative
    })

    csv_path = BASE_PATH / "energy_table_mass_metric.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    png_path = BASE_PATH / "energy_mass_metric.png"
    plot_energy(FREQS, cumulative, incremental,
                "Modal energy (mass metric)", png_path)
    print(f"Saved plot: {png_path}")

    print(f"\nTotal runtime: {time.time()-t0:.2f}s")
    print("="*90)
    print("Done.")


if __name__ == "__main__":
    main()
