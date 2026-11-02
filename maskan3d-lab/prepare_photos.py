#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أداة فحص وتجهيز صور العقارات لمسار Maskan3D Lab
Maskan3D Photo Preparation & Preflight Tool

الوظائف:
1. تقرير شامل لعدد الصور، نطاق الدقة (Resolution Range)، وإجمالي الحجم.
2. كشف الصور الضبابية باستخدام تباين لابلاس (Laplacian Variance) وترتيب أكثر الصور ضبابية.
3. فحص وتصحيح تدوير الصور المفقود أو المقروء خطأ عبر بيانات EXIF.
4. تحذير ذكي إذا كان عدد الصور أقل من 60 أو أكثر من 250.
5. ضغط الصور داخل photos.zip بشكل مسطح (Flat) وجاهز للرفع إلى Google Colab.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ضبط ترميز الإخراج ليدعم الحروف العربية في جميع أنظمة التشغيل
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
except ImportError as e:
    print(f"❌ خطأ: الحزم المطلوبة غير متوفرة: {e}")
    print("يرجى تثبيتها عبر: pip install -r requirements.txt")
    sys.exit(1)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def generate_dummy_photos(photos_dir: Path, count: int = 5) -> None:
    """توليد صور اختبار تجريبية في حال كان المجلد فارغاً."""
    photos_dir.mkdir(parents=True, exist_ok=True)
    print("⚠️ مجلد الصور فارغ! جاري توليد صور اختبار تجريبية (Dummy Images)...")

    for i in range(1, count + 1):
        # إنشاء صورة بها أشكال هندسية وخطوط واضحة
        width, height = 1280, 720
        img_np = np.zeros((height, width, 3), dtype=np.uint8)
        
        # خلفية متدرجة
        for y in range(height):
            img_np[y, :, 0] = int(255 * (y / height))
            img_np[y, :, 1] = int(200 * (1 - y / height))
            img_np[y, :, 2] = 120

        # أشكال هندسية بنقاط حادة لاختبار التباين
        cv2.circle(img_np, (300, 360), 150, (255, 255, 255), 4)
        cv2.rectangle(img_np, (500, 200), (800, 500), (0, 255, 255), 4)
        cv2.line(img_np, (100, 100), (1180, 620), (255, 0, 0), 3)
        cv2.putText(
            img_np,
            f"Test Apartment Photo #{i}",
            (150, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # جعل الصورة الأخيرة ضبابية عمداً لاختبار كاشف الضبابية
        if i == count:
            img_np = cv2.GaussianBlur(img_np, (45, 45), 0)
        elif i == count - 1:
            img_np = cv2.GaussianBlur(img_np, (21, 21), 0)

        file_path = photos_dir / f"test_dummy_photo_{i}.jpg"
        pil_img = Image.fromarray(img_np)
        pil_img.save(file_path, "JPEG", quality=90)

    print(f"✅ تم توليد {count} صور تجريبية في {photos_dir}\n")


def calculate_blur_variance(pil_img: Image.Image) -> float:
    """حساب تباين لابلاس (Laplacian Variance) لقياس حدة الصورة ووضوحها."""
    rgb_arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def check_and_fix_exif_orientation(img_path: Path, fix: bool = True) -> Tuple[str, bool]:
    """
    فحص اتجاه EXIF وتصحيحه إذا كانت الصورة مقلوبة أو مائلة.
    يرجع (حالة الاتجاه, هل تم التصحيح).
    """
    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            orientation = exif.get(0x0112, None)

            if orientation is None:
                return "مفقود (افتراضي سليم)", False
            elif orientation == 1:
                return "صحيح (أفقي قياسي)", False
            else:
                # الصورة تحتاج تدوير/تصحيح
                if fix:
                    transposed = ImageOps.exif_transpose(img)
                    # حفظ الصورة بالاتجاه الفيزيائي الصحيح
                    transposed.save(img_path, quality=95)
                    return f"تم تعديله (كود {orientation} ➔ 1)", True
                else:
                    return f"مقلوب (كود {orientation})", False
    except Exception as e:
        return f"خطأ قراءة: {e}", False


def analyze_photos(photos_dir: Path, blur_threshold: float = 100.0, fix_orientation: bool = True) -> Dict:
    """تحليل شامل لمجلد الصور."""
    photos_dir.mkdir(parents=True, exist_ok=True)
    photo_files = [
        f for f in photos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not photo_files:
        generate_dummy_photos(photos_dir)
        photo_files = [
            f for f in photos_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    total_count = len(photo_files)
    total_size_bytes = 0
    widths = []
    heights = []
    file_details = []
    fixed_orientation_count = 0
    missing_exif_count = 0

    for path in photo_files:
        size_bytes = path.stat().st_size
        total_size_bytes += size_bytes

        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            
            # فحص الاتجاه وتصحيحه
            orientation_status, was_fixed = check_and_fix_exif_orientation(path, fix=fix_orientation)
            if was_fixed:
                fixed_orientation_count += 1
            if "مفقود" in orientation_status:
                missing_exif_count += 1

            # حساب الحدة / الضبابية
            blur_score = calculate_blur_variance(img)

        file_details.append({
            "name": path.name,
            "path": path,
            "width": w,
            "height": h,
            "size_kb": size_bytes / 1024,
            "blur_score": blur_score,
            "orientation_status": orientation_status,
            "is_blurry": blur_score < blur_threshold,
        })

    # ترتيب الصور من الأكثر ضبابية (الأقل سكور) إلى الأكثر حدة
    file_details.sort(key=lambda x: x["blur_score"])

    min_res = (min(widths), min(heights)) if widths else (0, 0)
    max_res = (max(widths), max(heights)) if widths else (0, 0)
    total_size_mb = total_size_bytes / (1024 * 1024)

    return {
        "count": total_count,
        "total_size_mb": total_size_mb,
        "min_res": min_res,
        "max_res": max_res,
        "fixed_orientation_count": fixed_orientation_count,
        "missing_exif_count": missing_exif_count,
        "details": file_details,
        "photo_files": photo_files,
    }


def create_flat_zip(photo_files: List[Path], output_zip_path: Path) -> float:
    """إنشاء ملف zip مسطح بالكامل بدون أي مجلدات فرعية."""
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in photo_files:
            # اسم الملف فقط دون المسار لضمان أن يكون الأرشيف Flat
            zf.write(file_path, arcname=file_path.name)
    
    zip_size_mb = output_zip_path.stat().st_size / (1024 * 1024)
    return zip_size_mb


def print_table(analysis: Dict, zip_path: Path, zip_size_mb: float, blur_threshold: float = 100.0) -> None:
    """طباعة النتائج باللغة العربية في جدول أنيق وسهل القراءة."""
    count = analysis["count"]
    total_size_mb = analysis["total_size_mb"]
    min_w, min_h = analysis["min_res"]
    max_w, max_h = analysis["max_res"]
    fixed_ori = analysis["fixed_orientation_count"]
    missing_ori = analysis["missing_exif_count"]
    details = analysis["details"]

    print("\n" + "=" * 70)
    print("          📊 تقرير فحص وتجهيز صور مسكن ثلاثية الأبعاد (Maskan3D)")
    print("=" * 70)

    # جدول المؤشرات العامة
    summary_rows = [
        ("إجمالي عدد الصور (Image Count)", f"{count} صورة"),
        ("نطاق الدقة (Resolution Range)", f"من {min_w}x{min_h} إلى {max_w}x{max_h} بكسل"),
        ("إجمالي حجم الصور غير المضغوطة", f"{total_size_mb:.2f} MB"),
        ("صور تم تصحيح تدويرها (Fixed EXIF)", f"{fixed_ori} صورة"),
        ("صور بدون بيانات تدوير (Missing EXIF)", f"{missing_ori} صورة"),
        ("حالة ملف الأرشيف المضغوط (Flat ZIP)", f"{zip_path.name} ({zip_size_mb:.2f} MB)"),
    ]

    col1_w = 40
    col2_w = 26
    print(f"┌{'─' * col1_w}┬{'─' * col2_w}┐")
    print(f"│ {'المؤشر':<{col1_w - 2}} │ {'القيمة':<{col2_w - 2}} │")
    print(f"├{'─' * col1_w}┼{'─' * col2_w}┤")
    for k, v in summary_rows:
        print(f"│ {k:<{col1_w - 2}} │ {v:<{col2_w - 2}} │")
    print(f"└{'─' * col1_w}┴{'─' * col2_w}┘")

    # فحص القيود والتحذيرات الذكية للعدد
    print("\n🔍 تقييم جاهزية عدد الصور للتدريب (Density Check):")
    if count < 60:
        print(f"  ⚠️ تحذير: عدد الصور ({count}) أقل من 60 صورة!")
        print("     قد تعاني عملية COLMAP من فجوات في التغطية، ولن يتمكن النموذج من إعادة بناء كافة زوايا الشقة.")
        print("     💡 التوصية: التقط ما بين 80 إلى 150 صورة مع تداخل (Overlap) بنسبة 70% على الأقل بين كل صورتين.")
    elif count > 250:
        print(f"  ⚠️ تحذير: عدد الصور ({count}) يتجاوز 250 صورة!")
        print("     معالجة هذا العدد على معالج Colab (CPU COLMAP) قد تستغرق ساعات طويلة.")
        print("     💡 التوصية: تقليص الصور إلى أفضل 120-180 صورة واستبعاد المتشابهة جداً أو الضبابية.")
    else:
        print(f"  ✅ ممتاز: عدد الصور ({count}) مثالي جداً لإنشاء جولة ثلاثية الأبعاد متكاملة وسريعة.")

    # قائمة أكثر الصور ضبابية
    print(f"\n🔍 تحليل حدة ووضوح الصور (Blur Detection - عتبة التباين: {blur_threshold}):")
    blurry_images = [d for d in details if d["is_blurry"]]
    
    worst_count = min(5, len(details))
    worst_images = details[:worst_count]

    print(f"┌{'─' * 36}┬{'─' * 16}┬{'─' * 14}┐")
    print(f"│ {'اسم الصورة (الأكثر ضبابية)':<34} │ {'سكور الحدة':<14} │ {'الحالة':<12} │")
    print(f"├{'─' * 36}┼{'─' * 16}┼{'─' * 14}┤")
    for d in worst_images:
        status_text = "⚠️ ضبابية جداً" if d["is_blurry"] else "✅ مقبولة"
        print(f"│ {d['name'][:34]:<34} │ {d['blur_score']:<14.1f} │ {status_text:<12} │")
    print(f"└{'─' * 36}┴{'─' * 16}┴{'─' * 14}┘")

    if blurry_images:
        print(f"  ⚠️ تنبيه: تم رصد {len(blurry_images)} صورة ذات حدة منخفضة (أقل من {blur_threshold}).")
        print("     يُفضل مراجعتها وحذف غير الواضحة منها لأنها قد تسبب ضباباً في الـ Gaussian Splats.")
    else:
        print("  🎉 رائع! جميع الصور واضحة وحادة ولا توجد صور ضبابية حرجة.")

    print("\n" + "=" * 70)
    print(f"📁 تم حفظ ملف الأرشيف بنجاح في: {zip_path.resolve()}")
    print("🚀 يمكنك الآن سحب ملف photos.zip مباشرة إلى لوحة ملفات Google Colab للبدء!")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maskan3D Photos Preflight & Preparation Tool"
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=Path(__file__).parent / "photos",
        help="مسار مجلد الصور (الافتراضي: maskan3d-lab/photos)",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path(__file__).parent / "photos.zip",
        help="مسار ملف zip الناتج (الافتراضي: maskan3d-lab/photos.zip)",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="عتبة التباين لاكتشاف الضبابية (الافتراضي: 100.0)",
    )
    parser.add_argument(
        "--no-fix-orientation",
        action="store_true",
        help="تعطيل تصحيح اتجاه EXIF التلقائي",
    )

    args = parser.parse_args()

    # 1. تحليل الصور وتصحيح الاتجاه
    analysis = analyze_photos(
        photos_dir=args.photos_dir,
        blur_threshold=args.blur_threshold,
        fix_orientation=not args.no_fix_orientation,
    )

    # 2. إنشاء أرشيف zip مسطح
    zip_size_mb = create_flat_zip(
        photo_files=analysis["photo_files"],
        output_zip_path=args.output_zip,
    )

    # 3. طباعة الجدول والتقرير باللغة العربية
    print_table(
        analysis=analysis,
        zip_path=args.output_zip,
        zip_size_mb=zip_size_mb,
        blur_threshold=args.blur_threshold,
    )


if __name__ == "__main__":
    main()
