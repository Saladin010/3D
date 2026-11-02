import os
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

INPUT_PLY = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\apartment_splat.ply")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")

print("Loading original PLY file...")
plydata = PlyData.read(str(INPUT_PLY))
orig_vertex = plydata['vertex']
orig_count = len(orig_vertex)
orig_size_bytes = INPUT_PLY.stat().st_size
orig_size_mb = round(orig_size_bytes / (1024 * 1024), 2)

TARGET_BUDGET_MB = 25.0
TARGET_BUDGET_BYTES = 25.0 * 1024 * 1024

results = []

# Base scenario: Original
results.append({
    "scenario": "Original (exported by splatfacto)",
    "scenario_code": "original",
    "gaussian_count": orig_count,
    "size_bytes": orig_size_bytes,
    "size_mb": orig_size_mb,
    "pct_of_25mb_target": round((orig_size_bytes / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "MEASURED"
})

# Scenario a: Remove normals only (drop nx, ny, nz)
print("Testing Scenario A: Remove normals only...")
props_a = [p for p in orig_vertex.data.dtype.names if p not in ['nx', 'ny', 'nz']]
data_a = np.empty(orig_count, dtype=[(name, orig_vertex.data.dtype[name]) for name in props_a])
for p in props_a:
    data_a[p] = orig_vertex.data[p]

el_a = PlyElement.describe(data_a, 'vertex')
file_a = RAW_DIR / "temp_scenario_a.ply"
PlyData([el_a], text=False).write(str(file_a))
size_a_bytes = file_a.stat().st_size
size_a_mb = round(size_a_bytes / (1024 * 1024), 2)
file_a.unlink() # Delete after measuring

results.append({
    "scenario": "a. Remove normals only",
    "scenario_code": "a_no_normals",
    "gaussian_count": orig_count,
    "size_bytes": size_a_bytes,
    "size_mb": size_a_mb,
    "pct_of_25mb_target": round((size_a_bytes / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "MEASURED"
})

# Scenario b: SH degree 0 only (drop f_rest_0..44) + no normals
print("Testing Scenario B: SH degree 0 only + no normals...")
props_b = [p for p in props_a if not p.startswith('f_rest_')]
data_b = np.empty(orig_count, dtype=[(name, orig_vertex.data.dtype[name]) for name in props_b])
for p in props_b:
    data_b[p] = orig_vertex.data[p]

el_b = PlyElement.describe(data_b, 'vertex')
file_b = RAW_DIR / "temp_scenario_b.ply"
PlyData([el_b], text=False).write(str(file_b))
size_b_bytes = file_b.stat().st_size
size_b_mb = round(size_b_bytes / (1024 * 1024), 2)
file_b.unlink() # Delete after measuring

results.append({
    "scenario": "b. SH Degree 0 only (drop f_rest) + no normals",
    "scenario_code": "b_sh0_no_normals",
    "gaussian_count": orig_count,
    "size_bytes": size_b_bytes,
    "size_mb": size_b_mb,
    "pct_of_25mb_target": round((size_b_bytes / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "MEASURED"
})

# Scenario c: (b) + remove opacity < 0.05 and spatial outliers
print("Testing Scenario C: (b) + filter opacity < 0.05 & spatial outliers...")
raw_opacity = np.array(orig_vertex['opacity'], dtype=float)
sig_opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -20.0, 20.0)))

px = np.array(orig_vertex['x'], dtype=float)
py = np.array(orig_vertex['y'], dtype=float)
pz = np.array(orig_vertex['z'], dtype=float)
bx_min, bx_max = float(np.percentile(px, 1)), float(np.percentile(px, 99))
by_min, by_max = float(np.percentile(py, 1)), float(np.percentile(py, 99))
bz_min, bz_max = float(np.percentile(pz, 1)), float(np.percentile(pz, 99))

valid_mask_c = (sig_opacity >= 0.05) & (px >= bx_min) & (px <= bx_max) & (py >= by_min) & (py <= by_max) & (pz >= bz_min) & (pz <= bz_max)
data_c = data_b[valid_mask_c]
count_c = len(data_c)

el_c = PlyElement.describe(data_c, 'vertex')
file_c = RAW_DIR / "temp_scenario_c.ply"
PlyData([el_c], text=False).write(str(file_c))
size_c_bytes = file_c.stat().st_size
size_c_mb = round(size_c_bytes / (1024 * 1024), 2)
file_c.unlink() # Delete after measuring

results.append({
    "scenario": "c. Scenario (b) + prune opacity < 0.05 & spatial outliers",
    "scenario_code": "c_filter_low_opac_and_outliers",
    "gaussian_count": count_c,
    "size_bytes": size_c_bytes,
    "size_mb": size_c_mb,
    "pct_of_25mb_target": round((size_c_bytes / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "MEASURED"
})

# Scenario d: (c) + prune the lowest-opacity 30%
print("Testing Scenario D: (c) + prune lowest 30% opacity...")
opac_c = 1.0 / (1.0 + np.exp(-np.clip(np.array(data_c['opacity'], dtype=float), -20.0, 20.0)))
p30_thresh = float(np.percentile(opac_c, 30))
keep_mask_d = opac_c >= p30_thresh
data_d = data_c[keep_mask_d]
count_d = len(data_d)

el_d = PlyElement.describe(data_d, 'vertex')
file_d = RAW_DIR / "temp_scenario_d.ply"
PlyData([el_d], text=False).write(str(file_d))
size_d_bytes = file_d.stat().st_size
size_d_mb = round(size_d_bytes / (1024 * 1024), 2)
file_d.unlink() # Delete after measuring

results.append({
    "scenario": "d. Scenario (c) + prune lowest 30% opacity",
    "scenario_code": "d_prune_lowest_30pct",
    "gaussian_count": count_d,
    "size_bytes": size_d_bytes,
    "size_mb": size_d_mb,
    "pct_of_25mb_target": round((size_d_bytes / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "MEASURED"
})

# Scenario e: Quantized / compressed format estimate
# SuperSplat compressed ply or SOG:
# We checked repos/supersplat and confirmed no numerical ratio is documented in README/docs.
# Thus documented ratio is UNKNOWN.
# We infer using standard 3DGS compressed quantization (15-20 bytes / gaussian):
# On scenario (c) count_c: ~18 bytes * count_c:
inferred_compressed_bytes_c = int(count_c * 18)
inferred_compressed_mb_c = round(inferred_compressed_bytes_c / (1024 * 1024), 2)

results.append({
    "scenario": "e. Quantised / Compressed format (SOG / SuperSplat compressed PLY)",
    "scenario_code": "e_compressed_quantized",
    "gaussian_count": count_c,
    "size_bytes": inferred_compressed_bytes_c,
    "size_mb": inferred_compressed_mb_c,
    "pct_of_25mb_target": round((inferred_compressed_bytes_c / TARGET_BUDGET_BYTES) * 100, 1),
    "status": "INFERRED (documented ratio in repos/supersplat is UNKNOWN; inferred based on 18 bytes/gaussian quantization standard: 16-bit pos, 8-bit scale/rot, 8-bit DC)"
})

with open(RAW_DIR / "task5_size_reduction.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n--- Summary of Size-Reduction Scenarios ---")
for r in results:
    print(f"[{r['status'][:8]}] {r['scenario']}: {r['gaussian_count']} Gaussians | {r['size_mb']} MB ({r['pct_of_25mb_target']}% of target)")

