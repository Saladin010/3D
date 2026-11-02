import json
import numpy as np
from pathlib import Path
from plyfile import PlyData

PLY_PATH = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\apartment_splat.ply")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading PLY file: {PLY_PATH}...")
plydata = PlyData.read(str(PLY_PATH))

vertex = plydata['vertex']
num_gaussians = len(vertex)
properties = [p.name for p in vertex.properties]

x = np.array(vertex['x'], dtype=float)
y = np.array(vertex['y'], dtype=float)
z = np.array(vertex['z'], dtype=float)
opacity = np.array(vertex['opacity'], dtype=float)

# Sigmoid / active opacity check
# In 3DGS, opacity in ply is often raw logits, active opacity is sigmoid(opacity)
active_opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity, -20, 20)))

# Bounding box of splat
bbox = {
    "x_min": float(round(np.min(x), 4)),
    "x_max": float(round(np.max(x), 4)),
    "y_min": float(round(np.min(y), 4)),
    "y_max": float(round(np.max(y), 4)),
    "z_min": float(round(np.min(z), 4)),
    "z_max": float(round(np.max(z), 4)),
    "span_x": float(round(np.ptp(x), 4)),
    "span_y": float(round(np.ptp(y), 4)),
    "span_z": float(round(np.ptp(z), 4)),
    "volume_proxy": float(round(np.ptp(x) * np.ptp(y) * np.ptp(z), 4))
}

# Centroid and percentiles
percentiles_x = [float(round(p, 4)) for p in np.percentile(x, [1, 5, 50, 95, 99])]
percentiles_y = [float(round(p, 4)) for p in np.percentile(y, [1, 5, 50, 95, 99])]
percentiles_z = [float(round(p, 4)) for p in np.percentile(z, [1, 5, 50, 95, 99])]

# Scales analysis
scales = []
for s_prop in ['scale_0', 'scale_1', 'scale_2']:
    if s_prop in properties:
        s_vals = np.exp(np.array(vertex[s_prop], dtype=float))
        scales.append(s_vals)

scale_stats = {}
if scales:
    scales_arr = np.array(scales) # shape (3, N)
    mean_scale_per_gaussian = np.mean(scales_arr, axis=0)
    scale_stats = {
        "min_scale": float(round(np.min(mean_scale_per_gaussian), 6)),
        "median_scale": float(round(np.median(mean_scale_per_gaussian), 6)),
        "max_scale": float(round(np.max(mean_scale_per_gaussian), 6)),
        "large_gaussians_count": int(np.sum(mean_scale_per_gaussian > 1.0)),
        "tiny_gaussians_count": int(np.sum(mean_scale_per_gaussian < 0.001))
    }

# Android Memory / VRAM calculations
file_size_bytes = PLY_PATH.stat().st_size
file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

# Mobile VRAM estimate:
# Typical WebGL/WebGPU 3DGS renderer (e.g., GaussianSplats3D / SuperSplat):
# Each Gaussian buffer layout: pos (12B) + rot (16B) + scale (12B) + color (12B) + opacity (4B) = 56 bytes.
# Plus sorting buffer / index buffer / depth buffer: ~32-48 bytes per Gaussian.
# Total VRAM footprint = N * (56 + 32) bytes = ~88 bytes / gaussian.
vram_geometry_mb = round((num_gaussians * 56) / (1024 * 1024), 2)
vram_total_estimated_mb = round((num_gaussians * 88) / (1024 * 1024), 2)

ply_summary = {
    "filename": PLY_PATH.name,
    "file_size_bytes": file_size_bytes,
    "file_size_mb": file_size_mb,
    "num_gaussians": num_gaussians,
    "properties_count": len(properties),
    "properties": properties,
    "spatial_bounding_box": bbox,
    "percentiles_1_5_50_95_99": {
        "x": percentiles_x,
        "y": percentiles_y,
        "z": percentiles_z
    },
    "active_opacity_stats": {
        "min": float(round(np.min(active_opacity), 4)),
        "median": float(round(np.median(active_opacity), 4)),
        "mean": float(round(np.mean(active_opacity), 4)),
        "max": float(round(np.max(active_opacity), 4)),
        "transparent_count_under_0_1": int(np.sum(active_opacity < 0.1))
    },
    "scale_stats": scale_stats,
    "android_feasibility": {
        "target_vram_budget_mb": 150.0,
        "vram_geometry_mb": vram_geometry_mb,
        "vram_total_estimated_mb": vram_total_estimated_mb,
        "fits_in_android_budget": bool(vram_total_estimated_mb <= 150.0),
        "verdict": "FEASIBLE_VRAM_BUT_NEEDS_COMPRESSION" if vram_total_estimated_mb <= 150.0 else "EXCEEDS_BUDGET"
    }
}

with open(RAW_DIR / "02_ply_analysis.json", "w", encoding="utf-8") as f:
    json.dump(ply_summary, f, indent=2)

print(f"PLY Analysis done: {num_gaussians:,} Gaussians, {file_size_mb} MB.")
print(f"Bounding Box: X[{bbox['x_min']} to {bbox['x_max']}], Y[{bbox['y_min']} to {bbox['y_max']}], Z[{bbox['z_min']} to {bbox['z_max']}]")

