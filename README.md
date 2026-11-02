# 🏠 Maskan 3D - 3D Gaussian Splatting Indoor Reconstruction

مشروع متكامل لإعادة البناء ثلاثي الأبعاد للشقق السكنية والبيئات الداخلية باستخدام تقنيات **3D Gaussian Splatting** و **COLMAP (SfM)** ونظام **Nerfstudio (splatfacto)**.

---

## 🌟 أبرز مميزات المشروع (Key Highlights)

- **دقة هندسية عالية:** معالجة مسار حلقي كامل (360° Loop Closure) مسجل عبر **131 كاميرا** مع خطأ إعادة إسقاط أقل من بكسل واحد (**0.889 px**).
- **تدريب فائق السرعة:** تدريب موديل `splatfacto` لـ **15,000 خطوة** في **8 دقائق فقط** على بطاقة **Tesla T4 GPU**.
- **مجسم فائق الخفة للويب:** إنتاج مجسم 3D Gaussian Splatting خفيف الوزن (**37.73 MB** فقط بفضل تفعيل `--pipeline.model.sh-degree 0`) بدلاً من 144 MB، مما يتيح عرضه بسلاسة فائقة عبر المتصفحات والموبايل.
- **جاهزية كاملة لكاجل (Self-Contained Kaggle Pipeline):** نوت بوك مستقل يقوم بسحب الفيديو، واستخراج الإطارات، وحساب الكاميرات، والتدريب والتصدير ذاتياً.

---

## 📁 هيكلية المخرجات والملفات (Deliverables)

| الملف | الحجم | الوصف الهندسي |
| :--- | :--- | :--- |
| **`splat_maskan3d_sh0.ply`** | `37.73 MB` | مجسم الـ 3D Gaussian Splatting النهائي ويحتوي على **581,803** جاوسيان |
| **`transforms.json`** | `0.11 MB` | مصفوفات تحويل الكاميرات (131 كاميرا) بنموذج `OPENCV` مع معاملات التشوه |
| **`sparse_points.ply`** | `0.32 MB` | سحابة النقاط المرجعية الهيكلية من COLMAP تضم **22,667** نقطة ثلاثية الأبعاد |
| **`camera_trajectory.png`** | `0.22 MB` | رسم بياني هندسي لمسار حركة الكاميرا ودورانها داخل الغرفة |
| **`train.log`** | `1.93 MB` | سجل التدريب الكامل (15,000 خطوة) ومعدلات معالجة الأشعة (16.6M rays/sec) |
| **`config.yml`** | `7.7 KB` | ملف إعدادات الموديل وتكوين شبكة Splatfacto |

---

## 🚀 كيفية تشغيل واستعراض المجسم ثلاثي الأبعاد

يمكنك استعراض مجسم الشقة **`splat_maskan3d_sh0.ply`** مباشرة دون الحاجة لتثبيت أي برامج:
1. افتح عارض الجاوسيان السحابي: [PlayCanvas SuperSplat](https://playcanvas.com/supersplat/editor).
2. اسحب ملف `splat_maskan3d_sh0.ply` وأفلته في المتصفح.
3. استمتع بالتجول داخل الغرفة بزاوية 360 درجة.

---

## 📓 دفاتر العمل (Notebooks)

* **`notebooks/maskan3d_kaggle_train.ipynb`**: الدفتر الشامل المتكامل للتدريب والتصدير على Kaggle T4 GPU.
* **`notebooks/maskan3d_kaggle_diag.ipynb`**: دفتر التشخيص والمقارنة بين خوارزميات COLMAP المختلفة.

---

## 🛠️ المتطلبات البرمجية (Requirements)

```bash
pip install nerfstudio
sudo apt-get install -y colmap ffmpeg
```

---

## 📜 الترخيص (License)
MIT License
