import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData

RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
FIG_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Figure 1: Image Quality Distributions
with open(RAW_DIR / "03_image_quality_summary.json", "r", encoding="utf-8") as f:
    img_data = json.load(f)

# Load CSV
csv_path = RAW_DIR / "image_quality.csv"
import csv
blur_vals = []
bright_vals = []
clip_vals = []
fnames = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fnames.append(r["filename"])
        blur_vals.append(float(r["blur_score"]))
        bright_vals.append(float(r["mean_brightness"]))
        clip_vals.append(float(r["clipped_highlights_ratio"]))

plt.style.use('default')
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

# (a) Blur Score
axs[0].hist(blur_vals, bins=8, color='#2b5c8f', edgecolor='white', alpha=0.85)
axs[0].axvline(np.median(blur_vals), color='red', linestyle='--', label=f'Median: {np.median(blur_vals):.1f}')
axs[0].set_title("Blur Score (Laplacian Var)", fontsize=11, fontweight='bold')
axs[0].set_xlabel("Variance of Laplacian (higher = sharper)")
axs[0].set_ylabel("Image Count")
axs[0].legend()
axs[0].grid(True, linestyle=':', alpha=0.6)

# (b) Mean Brightness
axs[1].hist(bright_vals, bins=8, color='#d97724', edgecolor='white', alpha=0.85)
axs[1].axvline(np.median(bright_vals), color='red', linestyle='--', label=f'Median: {np.median(bright_vals):.1f}')
axs[1].set_title("Mean Brightness (Luminance 0-255)", fontsize=11, fontweight='bold')
axs[1].set_xlabel("Mean Grayscale Value")
axs[1].legend()
axs[1].grid(True, linestyle=':', alpha=0.6)

# (c) Clipped Highlights Ratio
axs[2].hist([v * 100 for v in clip_vals], bins=8, color='#c0392b', edgecolor='white', alpha=0.85)
axs[2].axvline(np.median(clip_vals) * 100, color='red', linestyle='--', label=f'Median: {np.median(clip_vals)*100:.2f}%')
axs[2].set_title("Clipped Highlights (% Pixels >= 250)", fontsize=11, fontweight='bold')
axs[2].set_xlabel("Percentage (%)")
axs[2].legend()
axs[2].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig1_path = FIG_DIR / "01_image_quality_distributions.png"
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# 2. Figure 2: COLMAP Incremental Mapping Progression
with open(RAW_DIR / "04_colmap_log_summary.json", "r", encoding="utf-8") as f:
    colmap_data = json.load(f)

reg_table = colmap_data.get("per_image_registration_table", [])
if reg_table:
    steps = [r["cumulative_registered"] for r in reg_table]
    seen = [r["points_seen"] for r in reg_table]
    model_pts = [r["total_model_points_at_step"] for r in reg_table]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    color = '#1f77b4'
    ax1.set_xlabel('Incremental Mapping Step (Registered Camera)', fontsize=11)
    ax1.set_ylabel('Points Seen by Registered Image', color=color, fontsize=11)
    ax1.plot(steps, seen, color=color, marker='o', linewidth=2, label="Observed 3D Points")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Total Model 3D Points in Reconstruction', color=color, fontsize=11)
    ax2.plot(steps, model_pts, color=color, linestyle='--', linewidth=2, label="Cumulative Sparse Points")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("COLMAP Incremental Registration: 26/26 Cameras", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig2_path = FIG_DIR / "02_reconstruction_progression.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

# 3. Figure 3: Gaussian Splat Top-Down Density (from apartment_splat.ply)
ply_path = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\apartment_splat.ply")
plydata = PlyData.read(str(ply_path))
vx = np.array(plydata['vertex']['x'], dtype=float)
vy = np.array(plydata['vertex']['y'], dtype=float)
vz = np.array(plydata['vertex']['z'], dtype=float)
opac = np.array(plydata['vertex']['opacity'], dtype=float)
active_op = 1.0 / (1.0 + np.exp(-np.clip(opac, -20, 20)))

# Filter extreme outliers (>99th percentile) for clean visual projection
p1, p99 = np.percentile(vx, [1, 99])
p1_z, p99_z = np.percentile(vz, [1, 99])
mask = (vx >= p1) & (vx <= p99) & (vz >= p1_z) & (vz <= p99_z) & (active_op > 0.1)

# Sample 50,000 points for scatter
idx = np.random.choice(np.where(mask)[0], size=min(50000, np.sum(mask)), replace=False)

fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# (a) Top-Down (XZ plane)
sc1 = axs[0].scatter(vx[idx], vz[idx], c=active_op[idx], cmap='viridis', s=0.8, alpha=0.6)
axs[0].set_title("3D Gaussian Splatting: Top-Down Orthographic View (X-Z plane)", fontsize=11, fontweight='bold')
axs[0].set_xlabel("X (arbitrary units)")
axs[0].set_ylabel("Z (arbitrary units)")
axs[0].grid(True, linestyle=':', alpha=0.5)
axs[0].axis('equal')
plt.colorbar(sc1, ax=axs[0], label="Active Opacity")

# (b) Side View (XY plane)
sc2 = axs[1].scatter(vx[idx], vy[idx], c=active_op[idx], cmap='plasma', s=0.8, alpha=0.6)
axs[1].set_title("3D Gaussian Splatting: Side View (X-Y plane)", fontsize=11, fontweight='bold')
axs[1].set_xlabel("X (arbitrary units)")
axs[1].set_ylabel("Y (arbitrary units)")
axs[1].grid(True, linestyle=':', alpha=0.5)
axs[1].axis('equal')
plt.colorbar(sc2, ax=axs[1], label="Active Opacity")

plt.tight_layout()
fig3_path = FIG_DIR / "03_gaussian_splat_geometry.png"
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

