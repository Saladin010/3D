import os
import json
import hashlib
from pathlib import Path

INPUT_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def get_file_info(file_path: Path):
    stat = file_path.stat()
    hasher = hashlib.md5()
    # compute md5 for small/medium files, or first 1MB for large
    with open(file_path, "rb") as f:
        chunk = f.read(1024 * 1024)
        hasher.update(chunk)
    return {
        "filename": file_path.name,
        "relative_path": str(file_path.relative_to(INPUT_DIR)),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "extension": file_path.suffix.lower(),
        "md5_prefix": hasher.hexdigest()
    }

inventory = []
total_size = 0
for root, dirs, files in os.walk(INPUT_DIR):
    for f in sorted(files):
        fp = Path(root) / f
        info = get_file_info(fp)
        inventory.append(info)
        total_size += info["size_bytes"]

# Classify inventory
images = [f for f in inventory if f["extension"] in [".jpg", ".jpeg", ".png"]]
models = [f for f in inventory if f["extension"] in [".ply", ".splat"]]
logs = [f for f in inventory if f["extension"] in [".log", ".txt", ".md"]]
configs = [f for f in inventory if f["extension"] in [".json", ".yml", ".yaml"]]

# Check what is missing
missing_items = [
    {
        "item": "transforms.json",
        "description": "Nerfstudio camera poses, intrinsics, and coordinate transform bounding box",
        "impact_if_missing": "Camera trajectory, baseline distances, and co-visibility proxy cannot be MEASURED; requires INFERRED / UNKNOWN status"
    },
    {
        "item": "colmap/sparse/0 (cameras.bin, images.bin, points3D.bin)",
        "description": "COLMAP sparse reconstruction bundle containing 3D tie points and camera extrinsics/intrinsics",
        "impact_if_missing": "Exact mean track length, point cloud density, and per-camera reprojection error cannot be directly parsed from binary"
    },
    {
        "item": "4 visual screenshots",
        "description": "Visual captures of the rendered splat from multiple perspectives (top-down, eye-level, corner, zoom)",
        "impact_if_missing": "Direct visual evaluation is constrained to 'NOT PROVIDED' (Rule 3 restricts evaluation to exactly 4 provided screenshots)"
    },
    {
        "item": "train.log / event logs",
        "description": "Tensorboard/Rich event logs for 5000 iterations",
        "impact_if_missing": "Training loss convergence curve and per-step PSNR metrics cannot be plotted across time"
    }
]

output_data = {
    "summary": {
        "total_files": len(inventory),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "image_count": len(images),
        "model_count": len(models),
        "log_count": len(logs),
        "config_count": len(configs)
    },
    "files": inventory,
    "missing_items": missing_items
}

# Save raw output
with open(RAW_DIR / "01_inventory.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

with open(RAW_DIR / "01_inventory.txt", "w", encoding="utf-8") as f:
    f.write(f"=== INPUT INVENTORY REPORT ===\n")
    f.write(f"Total files: {len(inventory)}\n")
    f.write(f"Total size: {round(total_size / (1024 * 1024), 2)} MB\n\n")
    f.write(f"Files Present:\n")
    for item in inventory:
        f.write(f"  - {item['relative_path']} | {item['size_mb']} MB | {item['extension']}\n")
    f.write(f"\nMissing Critical Artifacts:\n")
    for m in missing_items:
        f.write(f"  [MISSING] {m['item']}: {m['description']}\n")

print(f"Task 1 Inventory completed successfully: {len(inventory)} files cataloged.")

