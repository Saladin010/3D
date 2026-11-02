import os
import glob
import json
import csv
import numpy as np
import cv2
from PIL import Image, ExifTags
from pathlib import Path

INPUT_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\input\images")
RAW_DIR = Path(r"d:\Projects\maskan-lab\maskan3d-lab\analysis\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def robust_z_score(values):
    vals = np.array(values, dtype=float)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        return np.zeros_like(vals)
    return (vals - med) / (1.4826 * mad)

image_paths = sorted(glob.glob(str(INPUT_DIR / "*.jpg")))
print(f"Analyzing {len(image_paths)} images...")

records = []
for p in image_paths:
    fname = os.path.basename(p)
    file_size_kb = round(os.path.getsize(p) / 1024, 2)
    
    # 1. EXIF analysis via PIL
    has_exif = False
    make = None
    model = None
    focal_length = None
    datetime_str = None
    orientation = None
    
    with Image.open(p) as img:
        width, height = img.size
        raw_exif = img.getexif()
        if raw_exif:
            has_exif = True
            for tag_id, val in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "Make":
                    make = str(val).strip()
                elif tag == "Model":
                    model = str(val).strip()
                elif tag == "FocalLength":
                    try:
                        focal_length = float(val)
                    except:
                        focal_length = str(val)
                elif tag == "DateTime":
                    datetime_str = str(val).strip()
                elif tag == "Orientation":
                    orientation = val
        
        # Also check exif IFD for FocalLength if not in 0th IFD
        if has_exif and focal_length is None:
            try:
                exif_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
                if exif_ifd and 37386 in exif_ifd: # 37386 is FocalLength
                    focal_length = float(exif_ifd[37386])
            except:
                pass

        # Convert to numpy array safely without OpenCV unicode path bugs
        rgb = np.array(img.convert('RGB'))
    
    # Grayscale for metrics
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Blur score: variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    
    # Mean brightness
    mean_brightness = float(np.mean(gray))
    
    # Clipped highlights ratio (pixels >= 250)
    clipped_ratio = float(np.mean(gray >= 250))
    
    # Clipped shadows ratio (pixels <= 5)
    shadow_ratio = float(np.mean(gray <= 5))

    records.append({
        "filename": fname,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "file_size_kb": file_size_kb,
        "has_exif": has_exif,
        "make": make,
        "model": model,
        "focal_length": focal_length,
        "datetime": datetime_str,
        "orientation": orientation,
        "blur_score": blur_score,
        "mean_brightness": mean_brightness,
        "clipped_highlights_ratio": clipped_ratio,
        "shadow_ratio": shadow_ratio
    })

# Compute robust z-scores across population
blur_scores = [r["blur_score"] for r in records]
brightness_scores = [r["mean_brightness"] for r in records]
clipped_scores = [r["clipped_highlights_ratio"] for r in records]

blur_z = robust_z_score(blur_scores)
bright_z = robust_z_score(brightness_scores)
clip_z = robust_z_score(clipped_scores)

for i, r in enumerate(records):
    r["blur_z_score"] = float(round(blur_z[i], 3))
    r["brightness_z_score"] = float(round(bright_z[i], 3))
    r["clipped_z_score"] = float(round(clip_z[i], 3))
    
    # Flag outliers (threshold |z| > 2.5)
    outliers = []
    if abs(r["blur_z_score"]) > 2.5:
        outliers.append(f"blur(z={r['blur_z_score']})")
    if abs(r["brightness_z_score"]) > 2.5:
        outliers.append(f"brightness(z={r['brightness_z_score']})")
    if abs(r["clipped_z_score"]) > 2.5:
        outliers.append(f"clipped_highlights(z={r['clipped_z_score']})")
    
    r["is_outlier"] = len(outliers) > 0
    r["outlier_flags"] = "; ".join(outliers) if outliers else "None"

# Save CSV
csv_path = RAW_DIR / "image_quality.csv"
fieldnames = [
    "filename", "resolution", "file_size_kb", "has_exif", "make", "model", "focal_length",
    "blur_score", "mean_brightness", "clipped_highlights_ratio",
    "blur_z_score", "brightness_z_score", "clipped_z_score",
    "is_outlier", "outlier_flags"
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

# Summary statistics
summary = {
    "total_images": len(records),
    "resolutions": list(set(r["resolution"] for r in records)),
    "exif_presence_count": sum(1 for r in records if r["has_exif"]),
    "cameras_identified": list(set(f"{r['make']} {r['model']}" for r in records)),
    "blur_score": {
        "min": float(round(np.min(blur_scores), 2)),
        "median": float(round(np.median(blur_scores), 2)),
        "mean": float(round(np.mean(blur_scores), 2)),
        "max": float(round(np.max(blur_scores), 2)),
        "mad": float(round(np.median(np.abs(blur_scores - np.median(blur_scores))), 2))
    },
    "mean_brightness": {
        "min": float(round(np.min(brightness_scores), 2)),
        "median": float(round(np.median(brightness_scores), 2)),
        "mean": float(round(np.mean(brightness_scores), 2)),
        "max": float(round(np.max(brightness_scores), 2)),
        "mad": float(round(np.median(np.abs(brightness_scores - np.median(brightness_scores))), 2))
    },
    "clipped_highlights_ratio": {
        "min": float(round(np.min(clipped_scores), 4)),
        "median": float(round(np.median(clipped_scores), 4)),
        "mean": float(round(np.mean(clipped_scores), 4)),
        "max": float(round(np.max(clipped_scores), 4)),
        "mad": float(round(np.median(np.abs(clipped_scores - np.median(clipped_scores))), 4))
    },
    "outliers": [r for r in records if r["is_outlier"]]
}

with open(RAW_DIR / "03_image_quality_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Saved CSV to {csv_path}")
print(f"Summary: {summary['total_images']} images, Resolutions: {summary['resolutions']}, Outliers: {len(summary['outliers'])}")
