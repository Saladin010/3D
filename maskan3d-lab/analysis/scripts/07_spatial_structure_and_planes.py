import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData

INPUT_PLY = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\apartment_splat.ply")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
FIG_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\figures")

plydata = PlyData.read(str(INPUT_PLY))
vertex = plydata['vertex']

x = np.array(vertex['x'], dtype=float)
y = np.array(vertex['y'], dtype=float)
z = np.array(vertex['z'], dtype=float)
opacity = np.array(vertex['opacity'], dtype=float)
active_opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity, -20.0, 20.0)))

# Filter out low-opacity floaters for structural analysis
solid_mask = (active_opacity > 0.1)
x_s, y_s, z_s = x[solid_mask], y[solid_mask], z[solid_mask]
points = np.stack([x_s, y_s, z_s], axis=1)
total_solid = len(points)
print(f"Total solid points (opacity > 0.1): {total_solid}")

# Subsample up to 50,000 points for robust RANSAC plane fitting
np.random.seed(42)
sub_n = min(50000, total_solid)
sub_idx = np.random.choice(total_solid, size=sub_n, replace=False)
pts_sub = points[sub_idx]

# 1. RANSAC plane fitting function
def fit_plane_ransac(pts, threshold=0.15, max_iters=2000):
    best_inliers = []
    best_plane = None
    N = len(pts)
    
    for _ in range(max_iters):
        # Sample 3 random points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idx]
        
        # Normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p1)
        
        # Distance of all points to plane: |ax + by + cz + d|
        dists = np.abs(np.dot(pts, normal) + d)
        inliers = np.where(dists < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)
            if len(best_inliers) > 0.4 * N:
                break
                
    if best_plane is not None and len(best_inliers) > 3:
        # Refit with least squares on inliers
        inlier_pts = pts[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        u, s, vh = np.linalg.svd(inlier_pts - centroid)
        normal = vh[2] # smallest singular vector
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        dists = np.abs(np.dot(pts, normal) + d)
        best_inliers = np.where(dists < threshold)[0]
        best_plane = (normal, d)
        
    return best_plane, best_inliers

# Detect multiple planes sequentially
remaining_mask = np.ones(len(pts_sub), dtype=bool)
detected_planes = []

for p_i in range(5):
    cur_indices = np.where(remaining_mask)[0]
    if len(cur_indices) < 500:
        break
    plane, inliers_local = fit_plane_ransac(pts_sub[cur_indices], threshold=0.18, max_iters=1500)
    if plane is None or len(inliers_local) < 500:
        break
    
    inliers_global = cur_indices[inliers_local]
    normal, d = plane
    # Orient normal consistently
    if normal[1] < 0: # or largest component positive
        normal = -normal
        d = -d
        
    inlier_ratio = round(len(inliers_global) / sub_n, 4)
    detected_planes.append({
        "plane_id": p_i + 1,
        "normal": [float(round(n, 4)) for n in normal],
        "d": float(round(d, 4)),
        "inliers_count": len(inliers_global),
        "inlier_ratio": inlier_ratio
    })
    
    # Mark inliers as used
    remaining_mask[inliers_global] = False

# Classify planes: floor / ceiling vs walls
# Check which coordinate axis has the strongest alignment with the dominant plane normal
dominant_normal = np.abs(detected_planes[0]["normal"]) if detected_planes else [0, 1, 0]
axis_names = ['X', 'Y', 'Z']
up_axis_idx = int(np.argmax(dominant_normal))
up_axis_name = axis_names[up_axis_idx]

print(f"Dominant plane normal: {detected_planes[0]['normal'] if detected_planes else 'None'}")
print(f"Identified Up-Axis: {up_axis_name} (dominant component: {dominant_normal[up_axis_idx]:.3f})")

# Check perpendicularity between planes
wall_normals = []
floor_normal = None
for p in detected_planes:
    norm = np.array(p["normal"])
    # If normal is aligned with up-axis (>0.7), it's floor/ceiling
    if abs(norm[up_axis_idx]) > 0.7:
        p["type"] = "floor_or_ceiling"
        if floor_normal is None:
            floor_normal = norm
    else:
        p["type"] = "wall"
        wall_normals.append(norm)

orthogonality_checks = []
if floor_normal is not None and wall_normals:
    for i, wn in enumerate(wall_normals):
        dot_floor = float(round(np.dot(wn, floor_normal), 4))
        angle_deg = float(round(np.degrees(np.arccos(np.clip(abs(dot_floor), -1, 1))), 2))
        orthogonality_checks.append({
            "pair": f"wall_{i+1}_vs_floor",
            "dot_product": dot_floor,
            "angle_deg": angle_deg,
            "is_perpendicular": abs(dot_floor) < 0.25 # within ~14 degrees of 90
        })

if len(wall_normals) >= 2:
    for i in range(len(wall_normals)):
        for j in range(i+1, len(wall_normals)):
            dot_walls = float(round(np.dot(wall_normals[i], wall_normals[j]), 4))
            angle_deg = float(round(np.degrees(np.arccos(np.clip(abs(dot_walls), -1, 1))), 2))
            orthogonality_checks.append({
                "pair": f"wall_{i+1}_vs_wall_{j+1}",
                "dot_product": dot_walls,
                "angle_deg": angle_deg,
                "relationship": "perpendicular" if abs(dot_walls) < 0.25 else ("parallel" if abs(dot_walls) > 0.8 else "oblique")
            })

# 2. Floor Plane Hole/Empty Region Detection
# Top-down slice near floor
if up_axis_name == 'Y':
    up_coords = points[:, 1]
    horiz_u = points[:, 0]
    horiz_v = points[:, 2]
    u_label, v_label = "X", "Z"
elif up_axis_name == 'Z':
    up_coords = points[:, 2]
    horiz_u = points[:, 0]
    horiz_v = points[:, 1]
    u_label, v_label = "X", "Y"
else:
    up_coords = points[:, 0]
    horiz_u = points[:, 1]
    horiz_v = points[:, 2]
    u_label, v_label = "Y", "Z"

# Slice points within 10th-30th percentile of height (floor slab level)
h_low = np.percentile(up_coords, 5)
h_high = np.percentile(up_coords, 35)
floor_slab_mask = (up_coords >= h_low) & (up_coords <= h_high)
u_floor = horiz_u[floor_slab_mask]
v_floor = horiz_v[floor_slab_mask]

# 2D Grid density analysis on floor plane
u_min, u_max = np.percentile(u_floor, [2, 98])
v_min, v_max = np.percentile(v_floor, [2, 98])
grid_res = 40
hist_2d, u_edges, v_edges = np.histogram2d(u_floor, v_floor, bins=grid_res, range=[[u_min, u_max], [v_min, v_max]])

# A cell is a "hole" if it has 0 points while being surrounded by occupied cells
occupied_mask = hist_2d > 5
total_cells = occupied_mask.size
# Compute convex bounding mask
occupied_indices = np.argwhere(occupied_mask)
if len(occupied_indices) > 0:
    min_r, min_c = np.min(occupied_indices, axis=0)
    max_r, max_c = np.max(occupied_indices, axis=0)
    interior_mask = np.zeros_like(occupied_mask)
    interior_mask[min_r:max_r+1, min_c:max_c+1] = True
    holes_mask = interior_mask & (~occupied_mask)
    hole_cells_count = int(np.sum(holes_mask))
    hole_ratio_pct = round((hole_cells_count / np.sum(interior_mask)) * 100, 2)
else:
    hole_cells_count = 0
    hole_ratio_pct = 0.0

# 3. Save Figures: Heatmaps
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

# (a) Top-Down Heatmap
h1 = axs[0].hist2d(horiz_u, horiz_v, bins=80, range=[[u_min, u_max], [v_min, v_max]], cmap='hot', cmin=1)
axs[0].set_title(f"Top-Down Density Heatmap ({u_label}-{v_label} Plane, Up={up_axis_name})", fontsize=11, fontweight='bold')
axs[0].set_xlabel(f"{u_label} (normalised)")
axs[0].set_ylabel(f"{v_label} (normalised)")
axs[0].set_aspect('equal')
plt.colorbar(h1[3], ax=axs[0], label="Gaussian Count")

# (b) Side View 1 (Horiz_U vs Up)
up_min, up_max = np.percentile(up_coords, [1, 99])
h2 = axs[1].hist2d(horiz_u, up_coords, bins=80, range=[[u_min, u_max], [up_min, up_max]], cmap='viridis', cmin=1)
axs[1].set_title(f"Side View 1 ({u_label}-{up_axis_name} Plane)", fontsize=11, fontweight='bold')
axs[1].set_xlabel(f"{u_label} (normalised)")
axs[1].set_ylabel(f"{up_axis_name} (Height)")
axs[1].set_aspect('equal')
plt.colorbar(h2[3], ax=axs[1], label="Gaussian Count")

# (c) Side View 2 (Horiz_V vs Up)
h3 = axs[2].hist2d(horiz_v, up_coords, bins=80, range=[[v_min, v_max], [up_min, up_max]], cmap='plasma', cmin=1)
axs[2].set_title(f"Side View 2 ({v_label}-{up_axis_name} Plane)", fontsize=11, fontweight='bold')
axs[2].set_xlabel(f"{v_label} (normalised)")
axs[2].set_ylabel(f"{up_axis_name} (Height)")
axs[2].set_aspect('equal')
plt.colorbar(h3[3], ax=axs[2], label="Gaussian Count")

plt.tight_layout()
heatmap_fig_path = FIG_DIR / "05_spatial_structure_heatmaps.png"
plt.savefig(heatmap_fig_path)
plt.close()
print(f"Saved heatmaps: {heatmap_fig_path}")

task4_res = {
    "up_axis": {
        "determined_axis": up_axis_name,
        "justification": f"RANSAC dominant horizontal plane normal is aligned with {up_axis_name} axis with component magnitude {dominant_normal[up_axis_idx]:.3f}"
    },
    "planes_detected": detected_planes,
    "orthogonality_checks": orthogonality_checks,
    "floor_plane_density": {
        "grid_resolution": f"{grid_res}x{grid_res}",
        "interior_grid_cells": int(np.sum(interior_mask)) if len(occupied_indices) > 0 else 0,
        "empty_hole_cells": hole_cells_count,
        "hole_ratio_pct": hole_ratio_pct,
        "finding": "Substantial floor holes detected in floor plane density grid" if hole_ratio_pct > 15 else "Floor density is largely continuous"
    }
}

with open(RAW_DIR / "task4_spatial_structure.json", "w", encoding="utf-8") as f:
    json.dump(task4_res, f, indent=2)

print("Task 4 completed successfully.")

