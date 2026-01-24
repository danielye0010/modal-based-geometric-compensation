import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import trimesh


# ===================== CONFIG ===================== 
STL_FILE  = BASE_PATH / "newplat.stl"
MODES_CSV = BASE_PATH / "mode" / "all_modes.csv"
MASS_FILE = BASE_PATH / "Msparse.txt"

X_FILE = BASE_PATH / "newx.txt"
Y_FILE = BASE_PATH / "newy.txt"
Z_FILE = BASE_PATH / "newz.txt"

MODE_ID = 7  # first physical mode (your convention)

# ---- OPTIONAL MANUAL OVERRIDES ----
AUTO_UNIT = True
STL_SCALE = 1.0
DISP_SCALE = 1.0

# KDTree sanity threshold (in mm after scaling)
MAX_MAP_DIST_WARN = 1.0

# Visualization exaggeration factor
EXAG_SCALE = 5.0


# ===================== IO ===================== #
def read_modes_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df["Direction"] = df["Direction"].astype(str).str.strip().str.upper()
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce").round().astype(int)
    df["Udir"] = pd.to_numeric(df["Udir"], errors="coerce")
    return df


def build_phi_from_modes(df: pd.DataFrame):
    nodes = np.sort(df["Node"].unique())
    N = len(nodes)
    num_modes = int(df["Mode"].max())

    Ux = np.zeros((N, num_modes))
    Uy = np.zeros_like(Ux)
    Uz = np.zeros_like(Ux)

    for m in range(1, num_modes + 1):
        for lab, Umat in (("X", Ux), ("Y", Uy), ("Z", Uz)):
            sub = df[(df["Mode"] == m) & (df["Direction"] == lab)]
            if not sub.empty:
                Umat[:, m-1] = (
                    sub.set_index("Node")
                       .reindex(nodes)["Udir"]
                       .fillna(0.0)
                       .to_numpy()
                )

    Phi = np.zeros((3*N, num_modes))
    Phi[0::3, :], Phi[1::3, :], Phi[2::3, :] = Ux, Uy, Uz
    return Phi, nodes


def read_sparse_matrix_txt(path: Path):
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
        "Deformation(mm)":"Udir",
    })

    need_cols = ["Node", "X Location (mm)", "Y Location (mm)", "Z Location (mm)", "Udir"]
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce").round().astype(int)
    df["Udir"] = pd.to_numeric(df["Udir"], errors="coerce")
    for c in ["X Location (mm)", "Y Location (mm)", "Z Location (mm)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[need_cols]


def build_disp_from_xyz(xf, yf, zf):
    dfx = read_deform_component(xf).rename(columns={"Udir":"UX"})
    dfy = read_deform_component(yf)[["Node","Udir"]].rename(columns={"Udir":"UY"})
    dfz = read_deform_component(zf)[["Node","Udir"]].rename(columns={"Udir":"UZ"})
    return dfx.merge(dfy, on="Node").merge(dfz, on="Node")


# ===================== DEBUG ===================== #
def bbox_stats(name, pts):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rg = mx - mn
    print(f"[DEBUG] {name} bbox range: {rg}, max = {rg.max():.6g}")
    return rg.max()


def disp_stats(name, disp):
    mag = np.linalg.norm(disp, axis=1)
    print(f"[DEBUG] {name} disp |u|: mean = {mag.mean():.6g}, max = {mag.max():.6g}")
    return mag.max()


def auto_choose_scales(fem_len, stl_len, disp_max):
    stl_scale, disp_scale = 1.0, 1.0
    ratio = fem_len / max(stl_len, 1e-12)
    if ratio > 200: stl_scale = 1000.0
    if disp_max < 0.05 and fem_len > 50: disp_scale = 1000.0
    return stl_scale, disp_scale


# ===================== MAIN ===================== #
def main():
    print("="*90)
    print(" Unit-safe Compensation Export (with x10 visualization STL) ")
    print("="*90)

    df_disp = build_disp_from_xyz(X_FILE, Y_FILE, Z_FILE)
    fem_xyz = df_disp[["X Location (mm)", "Y Location (mm)", "Z Location (mm)"]].to_numpy()
    fem_disp = df_disp[["UX","UY","UZ"]].to_numpy()

    fem_len = bbox_stats("FEM", fem_xyz)
    disp_max0 = disp_stats("FEM raw", fem_disp)

    mesh = trimesh.load(STL_FILE)
    stl_len = bbox_stats("STL raw", mesh.vertices)

    global STL_SCALE, DISP_SCALE
    if AUTO_UNIT:
        STL_SCALE, DISP_SCALE = auto_choose_scales(fem_len, stl_len, disp_max0)
        print(f"[DEBUG] AUTO_UNIT → STL_SCALE={STL_SCALE}, DISP_SCALE={DISP_SCALE}")

    mesh.vertices *= STL_SCALE
    fem_disp *= DISP_SCALE

    tree = cKDTree(fem_xyz)
    dist, idx = tree.query(mesh.vertices, k=1)
    if dist.max() > MAX_MAP_DIST_WARN:
        print("[WARNING] Large STL–FEM mapping distance")

    df_modes = read_modes_csv(MODES_CSV)
    Phi_all, _ = build_phi_from_modes(df_modes)
    M = read_sparse_matrix_txt(MASS_FILE)

    ndof = min(M.shape[0], Phi_all.shape[0], fem_disp.size)
    Phi_all = Phi_all[:ndof]
    M = M[:ndof, :ndof]
    D = fem_disp.reshape(-1)[:ndof]

    # ================= Direct Inversion =================
    disp_stl_DI = -fem_disp[idx]
    mesh_DI = mesh.copy()
    mesh_DI.vertices += disp_stl_DI
    mesh_DI.vertices /= STL_SCALE
    out1 = BASE_PATH / "compensation_direct_inversion.stl"
    mesh_DI.export(out1)

    # ---- x10 exaggerated ----
    mesh_DI_ex = mesh.copy()
    mesh_DI_ex.vertices += EXAG_SCALE * disp_stl_DI
    mesh_DI_ex.vertices /= STL_SCALE
    out1x = BASE_PATH / "compensation_direct_inversion_x10.stl"
    mesh_DI_ex.export(out1x)

    # ================= Modal Compensation =================
    Phi7 = Phi_all[:, MODE_ID-1:MODE_ID]
    A7 = Phi7.T @ (M @ Phi7)
    b7 = Phi7.T @ (M @ D)
    c7 = np.linalg.solve(A7 + 1e-12*np.eye(A7.shape[0]), b7)

    D7 = (Phi7 @ c7).reshape(-1, 3)
    disp_stl_modal = -D7[idx]

    mesh_M = mesh.copy()
    mesh_M.vertices += disp_stl_modal
    mesh_M.vertices /= STL_SCALE
    out2 = BASE_PATH / "compensation_modal_mode7.stl"
    mesh_M.export(out2)

    # ---- x10 exaggerated ----
    mesh_M_ex = mesh.copy()
    mesh_M_ex.vertices += EXAG_SCALE * disp_stl_modal
    mesh_M_ex.vertices /= STL_SCALE
    out2x = BASE_PATH / "compensation_modal_mode7_x10.stl"
    mesh_M_ex.export(out2x)

    print("[OK] Exported:")
    print(f"  {out1.name}")
    print(f"  {out2.name}")
    print(f"  {out1x.name} (x10 visualization)")
    print(f"  {out2x.name} (x10 visualization)")
    print("="*90)


if __name__ == "__main__":
    main()


