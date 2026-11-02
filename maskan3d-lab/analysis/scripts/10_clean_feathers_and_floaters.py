#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_clean_feathers_and_floaters.py
=================================
Deterministic 3DGS De-Feathering & Floater Removal Pipeline.

Removes feathery needle artifacts, giant background blur blobs, low-opacity haze,
and isolated floaters from 3D Gaussian Splatting PLY point clouds, and cleans out
overfitted spherical harmonics (SH=0 export) to eliminate rainbow view-dependent noise.

Pipeline Filters:
1. Binary PLY Ingestion (strict float32)
2. Filter 1: Opacity Thresholding (linear alpha rejection)
3. Filter 2: Scale & Anisotropy Spike Removal (De-Feathering & Scale Clamping)
4. Filter 3: Spatial Core Bounding Box & k-NN Density Filtering (cKDTree outlier pruning)
5. Filter 4: SH=0 Clean Color Export (dropping normals & f_rest_*, <20 MB footprint)

Author: Principal 3D Computer Vision Engineer
Project: Maskan 3D Lab (maskan-lab)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent cp1252 encoding crashes
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic 3DGS De-Feathering & Floater Removal Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input 3DGS .ply file (default: auto-detect from notes or analysis/input)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output cleaned .ply file (default: notes/splat_cleaned.ply)"
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.15,
        help="Minimum linear opacity threshold alpha in [0.0, 1.0] (default: 0.15)"
    )
    parser.add_argument(
        "--max-anisotropy",
        type=float,
        default=8.0,
        help="Maximum anisotropy aspect ratio threshold r = s_max / (s_min + 1e-6) (default: 8.0)"
    )
    parser.add_argument(
        "--core-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for spatial core bounding box [P_core, P_(100-core)] (default: 2.0)"
    )
    parser.add_argument(
        "--blob-percentile",
        type=float,
        default=97.0,
        help="Percentile threshold for giant background blob rejection (default: 97.0)"
    )
    parser.add_argument(
        "--needle-scale-multiplier",
        type=float,
        default=2.5,
        help="Scale multiplier above median for needle spike rejection (default: 2.5)"
    )
    parser.add_argument(
        "--scale-clamp-ratio",
        type=float,
        default=6.0,
        help="Maximum scale clamp multiplier relative to median axis scale (default: 6.0)"
    )
    parser.add_argument(
        "--clamp-anisotropy",
        action="store_true",
        help="Enforce strict aspect ratio cap on remaining splats (clamps s_max / s_min <= max_anisotropy)"
    )
    parser.add_argument(
        "--target-max-mb",
        type=float,
        default=None,
        help="Optional target maximum PLY file size in MB (adaptively prunes lowest opacity if exceeded)"
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=16,
        help="Number of nearest neighbors for density filtering (default: 16)"
    )
    parser.add_argument(
        "--knn-std-ratio",
        type=float,
        default=2.0,
        help="Outlier rejection threshold in standard deviations above mean NN distance (default: 2.0)"
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Optional path to write full structured JSON execution report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose per-step debug logs"
    )
    return parser.parse_args()


def find_default_input_path() -> Path:
    candidates = [
        Path("notes/splat_E0.ply"),
        Path("maskan3d-lab/analysis/input/apartment_splat.ply"),
        Path("apartment_splat.ply"),
        Path("splat_maskan3d_sh0.ply")
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1024:
            return c
    raise FileNotFoundError(
        "Could not auto-detect input .ply file. Please specify --input /path/to/model.ply"
    )


def compute_anisotropy_stats(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Compute aspect ratio statistics for diagnostics."""
    if len(s0) == 0:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "pct_above_8": 0.0, "pct_above_10": 0.0}
    scales = np.stack([s0, s1, s2], axis=1)
    s_max = np.max(scales, axis=1)
    s_min = np.min(scales, axis=1)
    ratio = s_max / (s_min + 1e-6)
    p50, p90, p99 = np.percentile(ratio, [50.0, 90.0, 99.0])
    pct_above_8 = float(np.mean(ratio > 8.0) * 100.0)
    pct_above_10 = float(np.mean(ratio > 10.0) * 100.0)
    return {
        "count": int(len(s0)),
        "p50_ratio": float(p50),
        "p90_ratio": float(p90),
        "p99_ratio": float(p99),
        "pct_above_8": pct_above_8,
        "pct_above_10": pct_above_10,
        "median_max_scale": float(np.median(s_max)),
        "p99_max_scale": float(np.percentile(s_max, 99.0))
    }


def clean_feathers_and_floaters(
    input_path: Path,
    output_path: Path,
    min_opacity: float = 0.15,
    max_anisotropy: float = 8.0,
    core_percentile: float = 2.0,
    blob_percentile: float = 97.0,
    needle_scale_multiplier: float = 2.5,
    scale_clamp_ratio: float = 6.0,
    clamp_anisotropy: bool = False,
    target_max_mb: float = None,
    knn_k: int = 16,
    knn_std_ratio: float = 2.0,
    verbose: bool = False
) -> dict:
    t_start = time.time()
    input_size_bytes = input_path.stat().st_size
    input_size_mb = input_size_bytes / (1024.0 * 1024.0)

    print("=" * 80)
    print("🧹 3DGS De-Feathering & Floater Removal Pipeline")
    print("=" * 80)
    print(f"📥 Input PLY:  {input_path.resolve()} ({input_size_mb:.2f} MB)")
    print(f"📤 Output PLY: {output_path.resolve()}")
    print("⚙️ Configuration Parameters:")
    print(f"   • Min Opacity (alpha):          {min_opacity:.3f}")
    print(f"   • Max Anisotropy Ratio (r):    {max_anisotropy:.1f}x")
    print(f"   • Core Bounding Box Percentile: [{core_percentile:.1f}%, {100.0 - core_percentile:.1f}%]")
    print(f"   • Blob Percentile (s_max):     P{blob_percentile:.1f}")
    print(f"   • Needle Scale Multiplier:     {needle_scale_multiplier:.1f}x median")
    print(f"   • Scale Axis Clamp Multiplier: {scale_clamp_ratio:.1f}x median axis")
    print(f"   • k-NN Outlier Rejection:      k={knn_k}, threshold=mu + {knn_std_ratio:.1f}*sigma")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Binary PLY Ingestion (strict float32)
    # -------------------------------------------------------------------------
    print("\n[1/5] Ingesting Binary PLY...")
    t0 = time.time()
    plydata = PlyData.read(str(input_path))
    vertex = plydata['vertex']
    n_initial = len(vertex)
    available_props = [p.name for p in vertex.properties]

    req_props = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    missing = [p for p in req_props if p not in available_props]
    if missing:
        raise ValueError(f"Input PLY is missing mandatory Gaussian properties: {missing}")

    has_sh_rest = any(p.startswith('f_rest_') for p in available_props)
    has_normals = any(p in available_props for p in ['nx', 'ny', 'nz'])

    x = np.asarray(vertex['x'], dtype=np.float32)
    y = np.asarray(vertex['y'], dtype=np.float32)
    z = np.asarray(vertex['z'], dtype=np.float32)
    opacity = np.asarray(vertex['opacity'], dtype=np.float32)
    scale_0 = np.asarray(vertex['scale_0'], dtype=np.float32)
    scale_1 = np.asarray(vertex['scale_1'], dtype=np.float32)
    scale_2 = np.asarray(vertex['scale_2'], dtype=np.float32)
    rot_0 = np.asarray(vertex['rot_0'], dtype=np.float32)
    rot_1 = np.asarray(vertex['rot_1'], dtype=np.float32)
    rot_2 = np.asarray(vertex['rot_2'], dtype=np.float32)
    rot_3 = np.asarray(vertex['rot_3'], dtype=np.float32)
    f_dc_0 = np.asarray(vertex['f_dc_0'], dtype=np.float32)
    f_dc_1 = np.asarray(vertex['f_dc_1'], dtype=np.float32)
    f_dc_2 = np.asarray(vertex['f_dc_2'], dtype=np.float32)

    # Initial baseline anisotropy stats
    s0_init = np.exp(scale_0)
    s1_init = np.exp(scale_1)
    s2_init = np.exp(scale_2)
    initial_stats = compute_anisotropy_stats(s0_init, s1_init, s2_init)
    print(f"   ✅ Loaded {n_initial:,} Gaussians in {time.time() - t0:.2f}s")
    print(f"   • Properties in file: {len(available_props)} (Normals: {has_normals}, Higher SH Rest: {has_sh_rest})")
    print(f"   • Initial Anisotropy: Median {initial_stats['p50_ratio']:.2f}x | P90 {initial_stats['p90_ratio']:.2f}x | P99 {initial_stats['p99_ratio']:.2f}x")
    print(f"   • Needles (ratio > 10.0): {initial_stats['pct_above_10']:.2f}% ({int(initial_stats['pct_above_10']*n_initial/100):,} splats)")

    # -------------------------------------------------------------------------
    # Step 2: Filter 1 - Opacity Thresholding
    # -------------------------------------------------------------------------
    print("\n[2/5] Filter 1: Opacity Thresholding...")
    t0 = time.time()
    # Compute active linear opacity via numerically stable sigmoid
    linear_alpha = 1.0 / (1.0 + np.exp(-np.clip(opacity, -20.0, 20.0)))
    mask_opacity = linear_alpha >= min_opacity

    n_low_opacity_pruned = int(np.sum(~mask_opacity))
    x = x[mask_opacity]
    y = y[mask_opacity]
    z = z[mask_opacity]
    opacity = opacity[mask_opacity]
    scale_0 = scale_0[mask_opacity]
    scale_1 = scale_1[mask_opacity]
    scale_2 = scale_2[mask_opacity]
    rot_0 = rot_0[mask_opacity]
    rot_1 = rot_1[mask_opacity]
    rot_2 = rot_2[mask_opacity]
    rot_3 = rot_3[mask_opacity]
    f_dc_0 = f_dc_0[mask_opacity]
    f_dc_1 = f_dc_1[mask_opacity]
    f_dc_2 = f_dc_2[mask_opacity]
    linear_alpha = linear_alpha[mask_opacity]

    n_after_f1 = len(x)
    print(f"   • Pruned {n_low_opacity_pruned:,} hazy splats with alpha < {min_opacity:.2f}")
    print(f"   • Remaining: {n_after_f1:,} ({n_after_f1 / n_initial * 100:.1f}%) in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Step 3: Filter 2 - Scale & Anisotropy Spike Removal (De-Feathering)
    # -------------------------------------------------------------------------
    print("\n[3/5] Filter 2: Scale & Anisotropy Spike Removal (De-Feathering)...")
    t0 = time.time()
    s0 = np.exp(scale_0)
    s1 = np.exp(scale_1)
    s2 = np.exp(scale_2)
    scales_stacked = np.stack([s0, s1, s2], axis=1)

    s_max = np.max(scales_stacked, axis=1)
    s_min = np.min(scales_stacked, axis=1)
    anisotropy_ratio = s_max / (s_min + 1e-6)

    # 1. Reject giant background blobs (s_max > P97)
    blob_threshold = float(np.percentile(s_max, blob_percentile))
    mask_giant_blobs = s_max > blob_threshold

    # 2. Reject needle / feather spikes: r > max_anisotropy AND s_max > 2.5 * median(s_max)
    median_s_max = float(np.median(s_max))
    needle_scale_threshold = median_s_max * needle_scale_multiplier
    mask_needle_spikes = (anisotropy_ratio > max_anisotropy) & (s_max > needle_scale_threshold)

    # Combined scale rejection mask
    mask_scale_reject = mask_giant_blobs | mask_needle_spikes
    mask_keep_f2 = ~mask_scale_reject

    n_giant_blobs_pruned = int(np.sum(mask_giant_blobs))
    n_needle_spikes_pruned = int(np.sum(mask_needle_spikes & ~mask_giant_blobs))

    x = x[mask_keep_f2]
    y = y[mask_keep_f2]
    z = z[mask_keep_f2]
    opacity = opacity[mask_keep_f2]
    s0 = s0[mask_keep_f2]
    s1 = s1[mask_keep_f2]
    s2 = s2[mask_keep_f2]
    rot_0 = rot_0[mask_keep_f2]
    rot_1 = rot_1[mask_keep_f2]
    rot_2 = rot_2[mask_keep_f2]
    rot_3 = rot_3[mask_keep_f2]
    f_dc_0 = f_dc_0[mask_keep_f2]
    f_dc_1 = f_dc_1[mask_keep_f2]
    f_dc_2 = f_dc_2[mask_keep_f2]
    linear_alpha = linear_alpha[mask_keep_f2]

    # 3. Scale Axis Clamping on remaining splats
    all_axes = np.concatenate([s0, s1, s2])
    median_axis_scale = float(np.median(all_axes))
    s_cap = scale_clamp_ratio * median_axis_scale

    clamp_mask = (s0 > s_cap) | (s1 > s_cap) | (s2 > s_cap)
    n_clamped_splats = int(np.sum(clamp_mask))

    s0 = np.minimum(s0, s_cap)
    s1 = np.minimum(s1, s_cap)
    s2 = np.minimum(s2, s_cap)

    # 4. Optional Strict Aspect Ratio Clamping
    n_ratio_clamped = 0
    if clamp_anisotropy:
        s_stacked = np.stack([s0, s1, s2], axis=1)
        cur_min = np.min(s_stacked, axis=1, keepdims=True)
        max_allowed = max_anisotropy * (cur_min + 1e-6)
        ratio_clamp_mask = np.any(s_stacked > max_allowed, axis=1)
        n_ratio_clamped = int(np.sum(ratio_clamp_mask))
        s0 = np.minimum(s0, max_allowed[:, 0])
        s1 = np.minimum(s1, max_allowed[:, 0])
        s2 = np.minimum(s2, max_allowed[:, 0])

    scale_0 = np.log(np.maximum(s0, 1e-12)).astype(np.float32)
    scale_1 = np.log(np.maximum(s1, 1e-12)).astype(np.float32)
    scale_2 = np.log(np.maximum(s2, 1e-12)).astype(np.float32)

    n_after_f2 = len(x)
    print(f"   • Pruned {n_giant_blobs_pruned:,} giant background blobs (> P{blob_percentile:.1f}: {blob_threshold:.4f})")
    print(f"   • Pruned {n_needle_spikes_pruned:,} feathery needle spikes (ratio > {max_anisotropy:.1f}x & scale > {needle_scale_threshold:.4f})")
    print(f"   • Clamped scales on {n_clamped_splats:,} splats exceeding {scale_clamp_ratio:.1f}x median axis scale ({s_cap:.4f})")
    if clamp_anisotropy:
        print(f"   • Clamped aspect ratios on {n_ratio_clamped:,} splats to enforce r <= {max_anisotropy:.1f}x")
    print(f"   • Remaining: {n_after_f2:,} ({n_after_f2 / n_initial * 100:.1f}%) in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Step 4: Filter 3 - Spatial Core Bounding & k-NN Density Filtering
    # -------------------------------------------------------------------------
    print("\n[4/5] Filter 3: Spatial Core Bounding & k-NN Density Filtering...")
    t0 = time.time()

    # Part A: Robust Core Bounding Box computed from high-opacity splats
    high_op_mask = linear_alpha > 0.30
    if np.sum(high_op_mask) >= 100:
        x_core = x[high_op_mask]
        y_core = y[high_op_mask]
        z_core = z[high_op_mask]
    else:
        x_core, y_core, z_core = x, y, z

    p_low = core_percentile
    p_high = 100.0 - core_percentile

    x_min, x_max = np.percentile(x_core, [p_low, p_high])
    y_min, y_max = np.percentile(y_core, [p_low, p_high])
    z_min, z_max = np.percentile(z_core, [p_low, p_high])

    mask_in_core = (
        (x >= x_min) & (x <= x_max) &
        (y >= y_min) & (y <= y_max) &
        (z >= z_min) & (z <= z_max)
    )
    n_outside_bbox = int(np.sum(~mask_in_core))

    x = x[mask_in_core]
    y = y[mask_in_core]
    z = z[mask_in_core]
    opacity = opacity[mask_in_core]
    scale_0 = scale_0[mask_in_core]
    scale_1 = scale_1[mask_in_core]
    scale_2 = scale_2[mask_in_core]
    s0 = s0[mask_in_core]
    s1 = s1[mask_in_core]
    s2 = s2[mask_in_core]
    rot_0 = rot_0[mask_in_core]
    rot_1 = rot_1[mask_in_core]
    rot_2 = rot_2[mask_in_core]
    rot_3 = rot_3[mask_in_core]
    f_dc_0 = f_dc_0[mask_in_core]
    f_dc_1 = f_dc_1[mask_in_core]
    f_dc_2 = f_dc_2[mask_in_core]

    print(f"   • Spatial Core BBox [P{p_low:.1f}, P{p_high:.1f}]:")
    print(f"     X: [{x_min:.3f}, {x_max:.3f}] | Y: [{y_min:.3f}, {y_max:.3f}] | Z: [{z_min:.3f}, {z_max:.3f}]")
    print(f"   • Pruned {n_outside_bbox:,} external perimeter floaters outside core room")

    # Part B: k-NN Density Filtering via cKDTree
    points = np.column_stack([x, y, z])
    tree = cKDTree(points)
    # Query k neighbors (col 0 is self-distance 0.0)
    dists, _ = tree.query(points, k=knn_k, workers=-1)
    mean_nn_dist = np.mean(dists[:, 1:], axis=1)

    dist_mu = float(np.mean(mean_nn_dist))
    dist_sigma = float(np.std(mean_nn_dist))
    knn_cutoff = dist_mu + (knn_std_ratio * dist_sigma)

    mask_knn_valid = mean_nn_dist <= knn_cutoff
    n_knn_outliers = int(np.sum(~mask_knn_valid))

    x = x[mask_knn_valid]
    y = y[mask_knn_valid]
    z = z[mask_knn_valid]
    opacity = opacity[mask_knn_valid]
    scale_0 = scale_0[mask_knn_valid]
    scale_1 = scale_1[mask_knn_valid]
    scale_2 = scale_2[mask_knn_valid]
    s0 = s0[mask_knn_valid]
    s1 = s1[mask_knn_valid]
    s2 = s2[mask_knn_valid]
    rot_0 = rot_0[mask_knn_valid]
    rot_1 = rot_1[mask_knn_valid]
    rot_2 = rot_2[mask_knn_valid]
    rot_3 = rot_3[mask_knn_valid]
    f_dc_0 = f_dc_0[mask_knn_valid]
    f_dc_1 = f_dc_1[mask_knn_valid]
    f_dc_2 = f_dc_2[mask_knn_valid]

    # Part C: Optional Adaptive Target Size Capping
    n_budget_pruned = 0
    if target_max_mb is not None:
        target_bytes = target_max_mb * 1024.0 * 1024.0
        max_allowed_splats = int(target_bytes / 56.0)
        if len(x) > max_allowed_splats:
            n_budget_pruned = len(x) - max_allowed_splats
            # Keep splats with highest opacity
            top_indices = np.argsort(opacity)[-max_allowed_splats:]
            x = x[top_indices]
            y = y[top_indices]
            z = z[top_indices]
            opacity = opacity[top_indices]
            scale_0 = scale_0[top_indices]
            scale_1 = scale_1[top_indices]
            scale_2 = scale_2[top_indices]
            s0 = s0[top_indices]
            s1 = s1[top_indices]
            s2 = s2[top_indices]
            rot_0 = rot_0[top_indices]
            rot_1 = rot_1[top_indices]
            rot_2 = rot_2[top_indices]
            rot_3 = rot_3[top_indices]
            f_dc_0 = f_dc_0[top_indices]
            f_dc_1 = f_dc_1[top_indices]
            f_dc_2 = f_dc_2[top_indices]
            print(f"   • Adaptively pruned {n_budget_pruned:,} lowest-opacity splats to meet {target_max_mb:.1f} MB target budget")

    n_after_f3 = len(x)
    print(f"   • k-NN Statistics (k={knn_k}): Mean dist = {dist_mu:.4f}, Std = {dist_sigma:.4f}, Cutoff = {knn_cutoff:.4f}")
    print(f"   • Pruned {n_knn_outliers:,} isolated floaters with sparse neighborhood")
    print(f"   • Remaining: {n_after_f3:,} ({n_after_f3 / n_initial * 100:.1f}%) in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Step 5: Filter 4 - SH=0 Clean Color Export (Standard Binary PLY)
    # -------------------------------------------------------------------------
    print("\n[5/5] Filter 4: SH=0 Clean Color Export...")
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 14 standard 3DGS attributes (float32 strict)
    out_dtype = [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4'),
        ('opacity', '<f4'),
        ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
        ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')
    ]

    clean_vertex = np.empty(n_after_f3, dtype=out_dtype)
    clean_vertex['x'] = x
    clean_vertex['y'] = y
    clean_vertex['z'] = z
    clean_vertex['f_dc_0'] = f_dc_0
    clean_vertex['f_dc_1'] = f_dc_1
    clean_vertex['f_dc_2'] = f_dc_2
    clean_vertex['opacity'] = opacity
    clean_vertex['scale_0'] = scale_0
    clean_vertex['scale_1'] = scale_1
    clean_vertex['scale_2'] = scale_2
    clean_vertex['rot_0'] = rot_0
    clean_vertex['rot_1'] = rot_1
    clean_vertex['rot_2'] = rot_2
    clean_vertex['rot_3'] = rot_3

    element = PlyElement.describe(clean_vertex, 'vertex')
    PlyData([element], text=False).write(str(output_path))

    output_size_bytes = output_path.stat().st_size
    output_size_mb = output_size_bytes / (1024.0 * 1024.0)

    # Final Anisotropy Statistics
    final_stats = compute_anisotropy_stats(s0, s1, s2)
    t_total = time.time() - t_start

    # Build Structured Report
    total_pruned = n_initial - n_after_f3
    pct_pruned = (total_pruned / n_initial) * 100.0
    size_reduction_pct = ((input_size_bytes - output_size_bytes) / input_size_bytes) * 100.0

    report = {
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "execution_time_seconds": round(t_total, 2),
        "counts": {
            "initial_gaussians": n_initial,
            "final_gaussians": n_after_f3,
            "total_pruned": total_pruned,
            "pct_pruned": round(pct_pruned, 2),
            "breakdown": {
                "low_opacity_haze": n_low_opacity_pruned,
                "giant_background_blobs": n_giant_blobs_pruned,
                "feather_needle_spikes": n_needle_spikes_pruned,
                "scales_clamped": n_clamped_splats,
                "aspect_ratio_clamped": n_ratio_clamped,
                "target_budget_pruned": n_budget_pruned,
                "outside_core_bbox": n_outside_bbox,
                "isolated_knn_floaters": n_knn_outliers
            }
        },
        "file_size": {
            "initial_mb": round(input_size_mb, 2),
            "final_mb": round(output_size_mb, 2),
            "reduction_mb": round(input_size_mb - output_size_mb, 2),
            "reduction_pct": round(size_reduction_pct, 2),
            "under_20mb_target": bool(round(output_size_mb, 2) <= 20.0)
        },
        "anisotropy_evaluation": {
            "initial": initial_stats,
            "final": final_stats,
            "needle_reduction_pct": round(
                ((initial_stats["pct_above_10"] - final_stats["pct_above_10"]) / (initial_stats["pct_above_10"] + 1e-9)) * 100.0, 2
            )
        }
    }

    # Summary Console Table
    print("\n" + "=" * 80)
    print("🏆 SUMMARY REPORT: 3DGS De-Feathering & Floater Removal")
    print("=" * 80)
    print(f"• Total Gaussians:         {n_initial:,} ➔ {n_after_f3:,} (Removed {total_pruned:,} | {pct_pruned:.1f}%)")
    print(f"• File Size:               {input_size_mb:.2f} MB ➔ {output_size_mb:.2f} MB (Saved {input_size_mb - output_size_mb:.2f} MB | {size_reduction_pct:.1f}%)")
    print(f"• Needles (Ratio > 10.0):  {initial_stats['pct_above_10']:.2f}% ➔ {final_stats['pct_above_10']:.2f}% (Reduced by {report['anisotropy_evaluation']['needle_reduction_pct']:.1f}%)")
    print(f"• Median Anisotropy Ratio: {initial_stats['p50_ratio']:.2f}x ➔ {final_stats['p50_ratio']:.2f}x")
    print(f"• 99th Percentile Ratio:   {initial_stats['p99_ratio']:.2f}x ➔ {final_stats['p99_ratio']:.2f}x")
    print(f"• Execution Wall Time:     {t_total:.2f}s")
    print("=" * 80)
    print(f"✅ Clean model saved: {output_path} (Target <= 20 MB: {'PASS ✅' if round(output_size_mb, 2) <= 20.0 else 'FAIL ❌'})")
    print("=" * 80 + "\n")

    return report


def main():
    args = parse_args()

    input_path = Path(args.input) if args.input else find_default_input_path()
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.ply"

    report = clean_feathers_and_floaters(
        input_path=input_path,
        output_path=output_path,
        min_opacity=args.min_opacity,
        max_anisotropy=args.max_anisotropy,
        core_percentile=args.core_percentile,
        blob_percentile=args.blob_percentile,
        needle_scale_multiplier=args.needle_scale_multiplier,
        scale_clamp_ratio=args.scale_clamp_ratio,
        clamp_anisotropy=args.clamp_anisotropy,
        target_max_mb=args.target_max_mb,
        knn_k=args.knn_k,
        knn_std_ratio=args.knn_std_ratio,
        verbose=args.verbose
    )

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"📝 Structured JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
