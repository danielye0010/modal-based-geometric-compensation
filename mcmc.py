import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import ttest_rel

# ==================== CONFIG ==================== #
MODES_CSV = BASE_PATH / "mode" / "all_modes.csv"
MASS_FILE = BASE_PATH / "Msparse.txt"
STIFF_FILE = BASE_PATH / "Ksparse.txt"

X_FILE = BASE_PATH / "newx.txt"
Y_FILE = BASE_PATH / "newy.txt"
Z_FILE = BASE_PATH / "newz.txt"

SKIP_RIGID = 6    
NUM_FLEX = 1       
MC_RUNS = 20
NOISE_LEVEL = 0.05
SPARSE_RATIO = 0.7
SCALE_FACTOR = 5.0
RAND_SEED = 2025
ITER_STEPS = 6
NOISE_PER_STEP = 0.02
# ================================================== #

np.set_printoptions(precision=5, suppress=True)


# ==================================================
# I/O FUNCTIONS
# ==================================================
def read_modes_csv(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    nodes = np.sort(df["Node"].unique())
    N = len(nodes)
    num_modes = int(df["Mode"].max())

    Ux = np.zeros((N, num_modes))
    Uy = np.zeros_like(Ux)
    Uz = np.zeros_like(Ux)

    for m in range(1, num_modes + 1):
        for comp, arr in zip(["X", "Y", "Z"], [Ux, Uy, Uz]):
            sub = df[(df["Mode"] == m) & (df["Direction"] == comp)]
            if not sub.empty:
                arr[:, m - 1] = (
                    sub.set_index("Node")
                    .reindex(nodes)["Udir"]
                    .fillna(0.0)
                    .to_numpy()
                )

    Phi = np.zeros((3 * N, num_modes))
    Phi[0::3], Phi[1::3], Phi[2::3] = Ux, Uy, Uz

    ref_xyz = df[df["Mode"] == 1].groupby("Node")[["X", "Y", "Z"]].mean().reindex(nodes).to_numpy()
    return Phi, ref_xyz, nodes


def read_sparse_matrix_txt(path: Path) -> csr_matrix:
    rows, cols, vals = [], [], []
    pat = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.replace("D", "E").replace("d", "E")
            for m in pat.finditer(line):
                rows.append(int(m.group(1)) - 1)
                cols.append(int(m.group(2)) - 1)
                vals.append(float(m.group(3)))
    n = max(max(rows), max(cols)) + 1
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def read_deform_component(path: Path):
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Node Number": "Node", "Directional Deformation (mm)": "Udir"})
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce").astype(int)
    df["Udir"] = pd.to_numeric(df["Udir"], errors="coerce")
    return df[["Node", "Udir"]]


def read_xyz_disp(x, y, z, nodes):
    dx = read_deform_component(x).set_index("Node")["Udir"]
    dy = read_deform_component(y).set_index("Node")["Udir"]
    dz = read_deform_component(z).set_index("Node")["Udir"]
    df = pd.concat([dx, dy, dz], axis=1)
    df.columns = ["UX", "UY", "UZ"]
    df = df.reindex(nodes).fillna(0.0)
    return df.to_numpy().reshape(-1)


# ==================================================
# METRICS FUNCTIONS
# ==================================================
def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def curvature_rms(D, pts):
    N = pts.shape[0]
    D3 = D.reshape(-1, 3)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=6)
    cv = []
    for i in range(N):
        diff = D3[idx[i]] - D3[i]
        cv.append(np.mean(np.linalg.norm(diff, axis=1)))
    return float(np.mean(cv))


def energy_ratio_K(D_num, D_ref, K):
    num = float(D_num @ (K @ D_num))
    den = float(D_ref @ (K @ D_ref))
    return num / den if den > 1e-12 else np.nan


# ==================================================
# COMPENSATION OPERATORS
# ==================================================
def fit_modal_M_weighted(D, Phi, M):
    A = Phi.T @ (M @ Phi)
    b = Phi.T @ (M @ D)
    eps = 1e-12 * np.trace(A) / max(1, A.shape[0])
    c = np.linalg.solve(A + eps * np.eye(A.shape[0]), b)
    return Phi @ c, c


def comp_raw(D):
    return D.copy()


def comp_direct(D):
    return -D


def comp_modal(D, Phi_vib, M):
    Dout, _ = fit_modal_M_weighted(D, Phi_vib, M)
    return Dout


# ==================================================
# DISTURBANCES
# ==================================================
def disturb_none(D, **kwargs):
    return D.copy()


def disturb_noise(D, ratio, rng):
    return D * (1.0 + rng.standard_normal(len(D)) * ratio)


def disturb_sparse(D, keep_ratio, rng):
    g = rng.random(len(D) // 3) < keep_ratio
    mask = np.repeat(g, 3)
    return np.where(mask, D, 0.0)


def disturb_scale(D, scale, rng):
    return D * scale


# ==================================================
# ROBUSTNESS + CONVERGENCE
# ==================================================
def run_validation(D_true, method, disturb, kwargs, Phi_vib, M, K, ref_xyz,
                   runs=20, iters=6, seed=0):

    rng = np.random.default_rng(seed)
    RMS, CURV, GAM, RHO = [], [], [], []

    for _ in range(runs):
        Dp = disturb(D_true, **kwargs, rng=rng)

        # --- static ---
        if method == "raw":
            Dout = comp_raw(Dp)
        elif method == "direct":
            Dout = comp_direct(Dp)
        else:
            Dout = comp_modal(Dp, Phi_vib, M)

        RMS.append(rms(Dout - D_true))
        CURV.append(abs(curvature_rms(Dout, ref_xyz) - curvature_rms(D_true, ref_xyz)))
        GAM.append(energy_ratio_K(Dout, D_true, K))

        # --- dynamic ---
        Dk = Dp.copy()
        normT = np.linalg.norm(D_true)
        err_hist = []
        for _ in range(iters):
            if method == "raw":
                Dk_next = comp_raw(Dk)
            elif method == "direct":
                Dk_next = comp_direct(Dk)
            else:
                Dk_next = comp_modal(Dk, Phi_vib, M)

            err = np.linalg.norm(Dk_next - D_true) / normT
            err_hist.append(err)
            Dk = Dk_next + NOISE_PER_STEP * rng.standard_normal(len(Dk_next))

        RHO.append(err_hist[-1] / err_hist[0] if err_hist[0] > 1e-12 else 1.0)

    return dict(
        RMS=np.mean(RMS),
        Curv=np.mean(CURV),
        Gamma=np.mean(GAM),
        Rho=np.mean(RHO),
        RMS_list=RMS, Curv_list=CURV, Gamma_list=GAM
    )


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("\n================ ROBUSTNESS: Raw vs Direct vs Modal ================\n")

    Phi_all, ref_xyz, nodes = read_modes_csv(MODES_CSV)
    M = read_sparse_matrix_txt(MASS_FILE)
    K = read_sparse_matrix_txt(STIFF_FILE)
    D_true = read_xyz_disp(X_FILE, Y_FILE, Z_FILE, nodes)

    ndof = min(len(D_true), Phi_all.shape[0], M.shape[0], K.shape[0])
    Phi_all = Phi_all[:ndof, :]
    M = M[:ndof, :ndof]
    K = K[:ndof, :ndof]
    D_true = D_true[:ndof]
    ref_xyz = ref_xyz[:ndof // 3]

    # ⭐ Only the 7th mode
    Phi_vib = Phi_all[:, SKIP_RIGID: SKIP_RIGID + NUM_FLEX]

    specs = [
        ("None", disturb_none, dict()),  # ⭐ baseline
        ("Noise", disturb_noise, dict(ratio=NOISE_LEVEL)),
        ("Sparse", disturb_sparse, dict(keep_ratio=SPARSE_RATIO)),
        ("Scale", disturb_scale, dict(scale=SCALE_FACTOR)),
    ]

    rows = []

    for tag, disturb, kw in specs:
        print(f"\n=== Disturbance: {tag} ===")

        res_raw = run_validation(D_true, "raw", disturb, kw, Phi_vib, M, K, ref_xyz,
                                 MC_RUNS, ITER_STEPS, RAND_SEED)
        res_dir = run_validation(D_true, "direct", disturb, kw, Phi_vib, M, K, ref_xyz,
                                 MC_RUNS, ITER_STEPS, RAND_SEED)
        res_mod = run_validation(D_true, "modal", disturb, kw, Phi_vib, M, K, ref_xyz,
                                 MC_RUNS, ITER_STEPS, RAND_SEED)

        # paired t-test Direct vs Modal
        t_rms, p_rms = ttest_rel(res_dir["RMS_list"], res_mod["RMS_list"])
        t_cur, p_cur = ttest_rel(res_dir["Curv_list"], res_mod["Curv_list"])
        t_gam, p_gam = ttest_rel(res_dir["Gamma_list"], res_mod["Gamma_list"])

        rows.append(dict(
            Disturbance=tag,
            RMS_Raw=res_raw["RMS"],
            Curv_Raw=res_raw["Curv"],
            Gamma_Raw=res_raw["Gamma"],
            Rho_Raw=res_raw["Rho"],

            RMS_Direct=res_dir["RMS"],
            Curv_Direct=res_dir["Curv"],
            Gamma_Direct=res_dir["Gamma"],
            Rho_Direct=res_dir["Rho"],

            RMS_Modal=res_mod["RMS"],
            Curv_Modal=res_mod["Curv"],
            Gamma_Modal=res_mod["Gamma"],
            Rho_Modal=res_mod["Rho"],

            RMS_Improve=100 * (res_dir["RMS"] - res_mod["RMS"]) / res_dir["RMS"],
            Curv_Improve=100 * (res_dir["Curv"] - res_mod["Curv"]) / res_dir["Curv"],
            Gamma_Improve=100 * (res_mod["Gamma"] - res_dir["Gamma"]) / max(1e-9, res_dir["Gamma"]),

            p_RMS=p_rms, p_Curv=p_cur, p_Gamma=p_gam
        ))

    df = pd.DataFrame(rows)
    print("\n📊 FINAL SUMMARY:")
    print(df.round(5).to_string(index=False))

    out = BASE_PATH / "robustness_raw_direct_modal.csv"
    df.to_csv(out, index=False)
    print(f"\n💾 Saved to: {out}\n")


