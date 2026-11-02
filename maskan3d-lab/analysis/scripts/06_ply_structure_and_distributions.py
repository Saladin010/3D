import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree

INPUT_PLY = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\apartment_splat.ply")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
FIG_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\figures")
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("--- TASK 1: Header and Structure ---")
file_size_bytes = INPUT_PLY.stat().st_size
file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

plydata = PlyData.read(str(INPUT_PLY))
vertex = plydata['vertex']
num_gaussians = len(vertex)

props_info = []
dtype_sizes = {
    'float32': 4, 'float64': 8, 'float': 4,
    'uint8': 1, 'int8': 1, 'int32': 4, 'uint32': 4
}

total_vertex_bytes = 0
for prop in vertex.properties:
    dt = str(prop.val_dtype)
    sz = dtype_sizes.get(dt, 4)
    props_info.append({"name": prop.name, "dtype": dt, "size_bytes": sz})
    total_vertex_bytes += sz

# Group breakdown
groups = {
    "position": ["x", "y", "z"],
    "normals": ["nx", "ny", "nz"],
    "dc_colour": ["f_dc_0", "f_dc_1", "f_dc_2"],
    "higher_order_sh": [f"f_rest_{i}" for i in range(45)],
    "opacity": ["opacity"],
    "scale": ["scale_0", "scale_1", "scale_2"],
    "rotation": ["rot_0", "rot_1", "rot_2", "rot_3"]
}

group_summary = {}
for gname, gprops in groups.items():
    g_bytes_per_vertex = sum(p["size_bytes"] for p in props_info if p["name"] in gprops)
    g_total_bytes = g_bytes_per_vertex * num_gaussians
    g_pct = round((g_total_bytes / file_size_bytes) * 100, 2)
    group_summary[gname] = {
        "properties_count": len(gprops),
        "bytes_per_vertex": g_bytes_per_vertex,
        "total_bytes": g_total_bytes,
        "total_mb": round(g_total_bytes / (1024 * 1024), 2),
        "percent_of_file": g_pct
    }

# Check normals
nx = np.array(vertex['nx'], dtype=float)
ny = np.array(vertex['ny'], dtype=float)
nz = np.array(vertex['nz'], dtype=float)
normals_are_all_zero = bool(np.all(nx == 0) and np.all(ny == 0) and np.all(nz == 0))
normals_nonzero_count = int(np.sum((nx != 0) | (ny != 0) | (nz != 0)))

task1_res = {
    "file_size_bytes": file_size_bytes,
    "file_size_mb": file_size_mb,
    "num_gaussians": num_gaussians,
    "per_vertex_bytes": total_vertex_bytes,
    "header_size_bytes": file_size_bytes - (total_vertex_bytes * num_gaussians),
    "normals_are_all_zero": normals_are_all_zero,
    "normals_nonzero_count": normals_nonzero_count,
    "groups": group_summary,
    "all_properties": props_info
}

with open(RAW_DIR / "task1_header_structure.json", "w", encoding="utf-8") as f:
    json.dump(task1_res, f, indent=2)

print(f"Header: {num_gaussians} Gaussians, {total_vertex_bytes} bytes/vertex, all zero normals: {normals_are_all_zero}")

print("--- TASK 2: Distributions ---")
def get_percentiles(arr):
    p = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "p1": float(round(p[0], 6)),
        "p5": float(round(p[1], 6)),
        "p25": float(round(p[2], 6)),
        "p50": float(round(p[3], 6)),
        "p75": float(round(p[4], 6)),
        "p95": float(round(p[5], 6)),
        "p99": float(round(p[6], 6)),
        "mean": float(round(np.mean(arr), 6)),
        "std": float(round(np.std(arr), 6))
    }

# 1. Opacity after sigmoid
raw_opacity = np.array(vertex['opacity'], dtype=float)
sigmoid_opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -20.0, 20.0)))
opacity_dist = get_percentiles(sigmoid_opacity)

# 2. Scale per axis after exp
s0 = np.exp(np.array(vertex['scale_0'], dtype=float))
s1 = np.exp(np.array(vertex['scale_1'], dtype=float))
s2 = np.exp(np.array(vertex['scale_2'], dtype=float))
s0_dist = get_percentiles(s0)
s1_dist = get_percentiles(s1)
s2_dist = get_percentiles(s2)

# 3. Anisotropy ratio (max / min)
scales_stack = np.stack([s0, s1, s2], axis=1) # shape (N, 3)
max_scale = np.max(scales_stack, axis=1)
min_scale = np.min(scales_stack, axis=1)
eps = 1e-8
anisotropy = max_scale / (min_scale + eps)
anisotropy_dist = get_percentiles(anisotropy)

# 4. SH energy per degree
# Degree 0: f_dc_0, 1, 2
f_dc = np.stack([np.array(vertex[f'f_dc_{i}'], dtype=float) for i in range(3)], axis=1)
sh_deg0_energy = np.sum(f_dc ** 2, axis=1)
sh_deg0_dist = get_percentiles(sh_deg0_energy)

# Degree 1: f_rest_0 to 8 (9 coeffs: 3 bands * 3 colors)
f_rest_1 = np.stack([np.array(vertex[f'f_rest_{i}'], dtype=float) for i in range(9)], axis=1)
sh_deg1_energy = np.sum(f_rest_1 ** 2, axis=1)
sh_deg1_dist = get_percentiles(sh_deg1_energy)

# Degree 2: f_rest_9 to 23 (15 coeffs: 5 bands * 3 colors)
f_rest_2 = np.stack([np.array(vertex[f'f_rest_{i}'], dtype=float) for i in range(9, 24)], axis=1)
sh_deg2_energy = np.sum(f_rest_2 ** 2, axis=1)
sh_deg2_dist = get_percentiles(sh_deg2_energy)

# Degree 3: f_rest_24 to 44 (21 coeffs: 7 bands * 3 colors)
f_rest_3 = np.stack([np.array(vertex[f'f_rest_{i}'], dtype=float) for i in range(24, 45)], axis=1)
sh_deg3_energy = np.sum(f_rest_3 ** 2, axis=1)
sh_deg3_dist = get_percentiles(sh_deg3_energy)

task2_res = {
    "opacity_after_sigmoid": opacity_dist,
    "scale_0_exp": s0_dist,
    "scale_1_exp": s1_dist,
    "scale_2_exp": s2_dist,
    "scale_anisotropy_ratio": anisotropy_dist,
    "sh_energy_degree_0": sh_deg0_dist,
    "sh_energy_degree_1": sh_deg1_dist,
    "sh_energy_degree_2": sh_deg2_dist,
    "sh_energy_degree_3": sh_deg3_dist
}

with open(RAW_DIR / "task2_distributions.json", "w", encoding="utf-8") as f:
    json.dump(task2_res, f, indent=2)

# Save distribution plots
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
axs[0, 0].hist(sigmoid_opacity, bins=50, color='#2b5c8f', edgecolor='black', alpha=0.8)
axs[0, 0].set_title("Opacity Distribution (after Sigmoid)", fontsize=11, fontweight='bold')
axs[0, 0].set_xlabel("Active Opacity")
axs[0, 0].set_ylabel("Count")
axs[0, 0].grid(True, linestyle=':', alpha=0.6)

# Clip extreme scale values for visualization
log_s = np.log10(np.clip(max_scale, 1e-6, 1e2))
axs[0, 1].hist(log_s, bins=50, color='#27ae60', edgecolor='black', alpha=0.8)
axs[0, 1].set_title("Max Scale per Gaussian (log10 scale)", fontsize=11, fontweight='bold')
axs[0, 1].set_xlabel("log10(Max Scale)")
axs[0, 1].grid(True, linestyle=':', alpha=0.6)

log_aniso = np.log10(np.clip(anisotropy, 1.0, 1e4))
axs[1, 0].hist(log_aniso, bins=50, color='#e67e22', edgecolor='black', alpha=0.8)
axs[1, 0].set_title("Scale Anisotropy Ratio (max/min) (log10)", fontsize=11, fontweight='bold')
axs[1, 0].set_xlabel("log10(Anisotropy Ratio)")
axs[1, 0].set_ylabel("Count")
axs[1, 0].grid(True, linestyle=':', alpha=0.6)

axs[1, 1].boxplot([sh_deg0_energy, sh_deg1_energy, sh_deg2_energy, sh_deg3_energy],
                  tick_labels=["Deg 0 (DC)", "Deg 1", "Deg 2", "Deg 3"], showfliers=False, patch_artist=True)
axs[1, 1].set_title("Spherical Harmonics Energy by Degree", fontsize=11, fontweight='bold')
axs[1, 1].set_ylabel("Sum of Squared SH Coefficients")
axs[1, 1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig_dist_path = FIG_DIR / "04_ply_distributions.png"
plt.savefig(fig_dist_path)
plt.close()
print(f"Saved distribution plot: {fig_dist_path}")

print("--- TASK 3: Floater / Junk Analysis ---")
# Opacity thresholds
count_op_005 = int(np.sum(sigmoid_opacity < 0.05))
count_op_010 = int(np.sum(sigmoid_opacity < 0.10))
count_op_030 = int(np.sum(sigmoid_opacity < 0.30))

pct_op_005 = round((count_op_005 / num_gaussians) * 100, 2)
pct_op_010 = round((count_op_010 / num_gaussians) * 100, 2)
pct_op_030 = round((count_op_030 / num_gaussians) * 100, 2)

# Giant splats (largest scale > 99th percentile)
p99_max_scale = float(np.percentile(max_scale, 99))
giant_mask = max_scale > p99_max_scale
count_giant = int(np.sum(giant_mask))
# Ellipsoid volume = (4/3) * pi * s0 * s1 * s2
volumes = (4.0 / 3.0) * np.pi * s0 * s1 * s2
giant_total_volume = float(np.sum(volumes[giant_mask]))
all_total_volume = float(np.sum(volumes))
giant_vol_share_pct = round((giant_total_volume / (all_total_volume + eps)) * 100, 2)

# Spatial outliers from percentiles (1st to 99th percentile box)
px = np.array(vertex['x'], dtype=float)
py = np.array(vertex['y'], dtype=float)
pz = np.array(vertex['z'], dtype=float)

bx_min, bx_max = float(np.percentile(px, 1)), float(np.percentile(px, 99))
by_min, by_max = float(np.percentile(py, 1)), float(np.percentile(py, 99))
bz_min, bz_max = float(np.percentile(pz, 1)), float(np.percentile(pz, 99))

outside_robust_bbox = (px < bx_min) | (px > bx_max) | (py < by_min) | (py > by_max) | (pz < bz_min) | (pz > bz_max)
count_bbox_outliers = int(np.sum(outside_robust_bbox))
pct_bbox_outliers = round((count_bbox_outliers / num_gaussians) * 100, 2)

# kNN-distance outlier test on subsample up to 200k
subsample_size = min(200000, num_gaussians)
np.random.seed(42)
sub_idx = np.random.choice(num_gaussians, size=subsample_size, replace=False)
coords_sub = np.stack([px[sub_idx], py[sub_idx], pz[sub_idx]], axis=1)

print(f"Building cKDTree on {subsample_size} points...")
tree = cKDTree(coords_sub)
# k=11 because 1st nearest neighbor is the point itself
dists, _ = tree.query(coords_sub, k=11)
mean_knn_dist = np.mean(dists[:, 1:], axis=1) # average distance to 10 neighbors

knn_med = np.median(mean_knn_dist)
knn_mad = np.median(np.abs(mean_knn_dist - knn_med))
knn_threshold = float(knn_med + 3.0 * (1.4826 * knn_mad))
knn_outliers_sub = np.sum(mean_knn_dist > knn_threshold)
knn_outliers_est_total = int(round((knn_outliers_sub / subsample_size) * num_gaussians))
knn_outliers_pct = round((knn_outliers_sub / subsample_size) * 100, 2)

# Combined junk estimate:
# A Gaussian is junk if: (opacity < 0.05) OR (outside robust bbox) OR (giant splat with opacity < 0.3)
junk_mask = (sigmoid_opacity < 0.05) | outside_robust_bbox | (giant_mask & (sigmoid_opacity < 0.3))
total_junk_count = int(np.sum(junk_mask))
total_junk_pct = round((total_junk_count / num_gaussians) * 100, 2)

task3_res = {
    "opacity_thresholds": {
        "less_than_0_05": {"count": count_op_005, "percent": pct_op_005},
        "less_than_0_10": {"count": count_op_010, "percent": pct_op_010},
        "less_than_0_30": {"count": count_op_030, "percent": pct_op_030}
    },
    "giant_gaussians": {
        "p99_max_scale_threshold": p99_max_scale,
        "count": count_giant,
        "percent": round((count_giant / num_gaussians) * 100, 2),
        "giant_volume": giant_total_volume,
        "all_volume": all_total_volume,
        "volume_share_pct": giant_vol_share_pct
    },
    "spatial_outliers": {
        "robust_bbox_p1_p99": {
            "x": [bx_min, bx_max], "y": [by_min, by_max], "z": [bz_min, bz_max]
        },
        "points_outside_robust_bbox": count_bbox_outliers,
        "percent_outside_robust_bbox": pct_bbox_outliers
    },
    "knn_outlier_test": {
        "subsample_size": subsample_size,
        "k_neighbors": 10,
        "median_dist": float(round(knn_med, 6)),
        "mad_dist": float(round(knn_mad, 6)),
        "threshold": float(round(knn_threshold, 6)),
        "outliers_in_subsample": int(knn_outliers_sub),
        "estimated_total_outliers": knn_outliers_est_total,
        "percent": knn_outliers_pct
    },
    "junk_estimate": {
        "total_junk_gaussians": total_junk_count,
        "junk_percentage": total_junk_pct,
        "confidence": "HIGH (based on direct quantitative opacity (<0.05) and spatial bounding percentiles)"
    }
}

with open(RAW_DIR / "task3_junk_analysis.json", "w", encoding="utf-8") as f:
    json.dump(task3_res, f, indent=2)

print(f"Junk analysis: {total_junk_count} Gaussians ({total_junk_pct}%) classified as junk.")
