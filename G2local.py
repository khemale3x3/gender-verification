# """
# gender_detect_final.py
# ══════════════════════════════════════════════════════════════════════════════
# Detects gender (and age) from profile image URLs using:
#   • YOLOv8n       – person detection (confirms a human is in the photo)
#   • FairFace      – gender classification  (~93% accuracy, via HuggingFace)
#   • FairFace      – age classification     (~59% accuracy, via HuggingFace)

# NO API KEY REQUIRED.  Models are free and download automatically on first run.

# ── Setup ────────────────────────────────────────────────────────────────────
#   pip install ultralytics transformers torch pillow requests

# ── Run ──────────────────────────────────────────────────────────────────────
#   python gender_detect_final.py

#   Input  : hardcoded URLS list below  OR  create image_urls.csv with column "image_url"
#   Output : gender_results.csv   (appends; safe to resume after interruption)

# ── First-run downloads (one time only) ─────────────────────────────────────
#   yolov8n.pt                               ~6 MB   (Ultralytics CDN)
#   dima806/fairface_gender_image_detection  ~100 MB (HuggingFace)
#   dima806/fairface_age_image_detection     ~100 MB (HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════
# """

# import os
# import csv
# import time
# import warnings
# import logging
# import tempfile
# import requests
# from io import BytesIO
# from threading import Lock
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ── Silence noisy log output ──────────────────────────────────────────────────
# warnings.filterwarnings("ignore")
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("ultralytics").setLevel(logging.ERROR)

# from PIL import Image
# from transformers import pipeline
# from ultralytics import YOLO

# # ── Configuration ─────────────────────────────────────────────────────────────

# OUTPUT_CSV  = "gender_results.csv"
# INPUT_CSV   = "image_urls.csv"      # optional: CSV with "image_url" column

# # # Paste your image URLs here (used when image_urls.csv is not found):
# # URLS = [
# #     "https://assets.veelapp.com/focusonmore.jpg",
# #     "https://assets.veelapp.com/carolynjarrettjones.jpg",
# #     "https://assets.veelapp.com/baileyjst.jpg",
# #     "https://assets.veelapp.com/esteelbridal.jpg",
# #     "https://assets.veelapp.com/theosbornfour.jpg",
# #     "https://assets.veelapp.com/effiespaper.jpg",
# #     "https://assets.veelapp.com/bingeon.travel.jpg",
# #     "https://assets.veelapp.com/babynamesunday.jpg",
# #     "https://assets.veelapp.com/thestumpshop.jpg",
# #     "https://assets.veelapp.com/freysbaking.jpg",
# #     "https://assets.veelapp.com/hairbybiancarose.jpg",
# #     "https://assets.veelapp.com/andru.model.jpg",
# #     "https://assets.veelapp.com/mylifeinmotherhood.jpg",
# #     "https://assets.veelapp.com/the_google_pro.jpg",
# #     "https://assets.veelapp.com/_notjustjess.jpg",
# #     "https://assets.veelapp.com/robsmopolitan.jpg",
# #     "https://assets.veelapp.com/staycsmart.jpg",
# #     "https://assets.veelapp.com/oscarbravo.jpg",
# #     "https://assets.veelapp.com/maluvoss.jpg",
# #     "https://assets.veelapp.com/mypracticehome.jpg",
# # ]

# MAX_WORKERS             = 4     # parallel download+inference threads
# IMAGE_TIMEOUT           = 15    # seconds to wait for image download
# FACE_CONF_THRESHOLD     = 0.50  # min YOLO confidence to accept a "person"
# GENDER_CONF_THRESHOLD   = 0.50  # min FairFace confidence to report a gender

# # ── Age label → readable group ────────────────────────────────────────────────

# AGE_MAP = {
#     "0-2":         "(0-2)",
#     "3-9":         "(4-6)",
#     "10-19":       "(8-12)",
#     "20-29":       "(25-32)",
#     "30-39":       "(38-43)",
#     "40-49":       "(48-53)",
#     "50-59":       "(48-53)",
#     "60-69":       "(60-100)",
#     "more than 70":"(60-100)",
# }

# def map_age(label: str) -> str:
#     return AGE_MAP.get(label, "Unknown")

# # ── CSV helpers ───────────────────────────────────────────────────────────────

# FIELDNAMES = ["url", "name", "gender", "age_group",
#               "gender_confidence", "age_confidence", "status"]

# csv_lock = Lock()

# def write_row(row: dict) -> None:
#     with csv_lock:
#         with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
#             csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

# def load_processed_urls() -> set:
#     processed = set()
#     if not os.path.exists(OUTPUT_CSV):
#         return processed
#     with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             u = row.get("url", "").strip()
#             if u:
#                 processed.add(u)
#     return processed

# def url_to_name(url: str) -> str:
#     return url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]

# # ── Model loader ──────────────────────────────────────────────────────────────

# class Detector:
#     """Loads YOLOv8 + FairFace models once; then processes images."""

#     def __init__(self):
#         print("─" * 60)
#         print("Loading models (first run downloads ~200 MB total)")
#         print("─" * 60)

#         print("  [1/3] YOLOv8n  (person detection) …", end=" ", flush=True)
#         self.yolo = YOLO("yolov8n.pt")
#         print("✓")

#         print("  [2/3] FairFace gender model …", end=" ", flush=True)
#         self.gender = pipeline(
#             "image-classification",
#             model="dima806/fairface_gender_image_detection",
#             device=-1,
#         )
#         print("✓")

#         print("  [3/3] FairFace age model …", end=" ", flush=True)
#         self.age = pipeline(
#             "image-classification",
#             model="dima806/fairface_age_image_detection",
#             device=-1,
#         )
#         print("✓")
#         print()

#     # ── person detection ──────────────────────────────────────────────────────

#     def has_person(self, image_path: str) -> tuple[bool, float]:
#         """Return (found, best_confidence) for COCO class 0 = person."""
#         results = self.yolo(image_path, verbose=False)
#         best = 0.0
#         for r in results:
#             for box in r.boxes:
#                 if int(box.cls[0]) == 0:
#                     conf = float(box.conf[0])
#                     if conf > best:
#                         best = conf
#         return best >= FACE_CONF_THRESHOLD, best

#     # ── main predict ─────────────────────────────────────────────────────────

#     def predict(self, img: Image.Image, tmp_path: str) -> dict:
#         """
#         Run person detection then gender/age classification.
#         Returns a result dict with keys:
#             gender, age_group, gender_confidence, age_confidence, status
#         """
#         # 1. Person detection
#         found, conf = self.has_person(tmp_path)
#         if not found:
#             return {
#                 "gender": "Unknown",
#                 "age_group": "Unknown",
#                 "gender_confidence": 0.0,
#                 "age_confidence": 0.0,
#                 "status": f"No person detected (YOLO conf {conf:.1%})",
#             }

#         # 2. Gender
#         g_results = self.gender(img)
#         g_label   = g_results[0]["label"].capitalize()   # 'Male' / 'Female'
#         g_conf    = g_results[0]["score"] * 100

#         if g_conf < GENDER_CONF_THRESHOLD * 100:
#             return {
#                 "gender": "Unknown",
#                 "age_group": "Unknown",
#                 "gender_confidence": round(g_conf, 1),
#                 "age_confidence": 0.0,
#                 "status": f"Low gender confidence ({g_conf:.1f}%)",
#             }

#         # 3. Age
#         a_results = self.age(img)
#         a_label   = a_results[0]["label"]
#         a_conf    = a_results[0]["score"] * 100

#         return {
#             "gender": g_label,
#             "age_group": map_age(a_label),
#             "gender_confidence": round(g_conf, 1),
#             "age_confidence": round(a_conf, 1),
#             "status": "Success",
#         }

# # ── Per-URL worker ────────────────────────────────────────────────────────────

# def process_url(url: str, detector: Detector, idx: int, total: int) -> dict:
#     name = url_to_name(url)
#     prefix = f"[{idx:>3}/{total}]"

#     # Download image
#     try:
#         resp = requests.get(url, timeout=IMAGE_TIMEOUT)
#         resp.raise_for_status()
#         img = Image.open(BytesIO(resp.content)).convert("RGB")
#     except Exception as e:
#         print(f"{prefix} ✗ {name}: download failed — {e}")
#         return {"url": url, "name": name,
#                 "gender": "Unknown", "age_group": "Unknown",
#                 "gender_confidence": "", "age_confidence": "",
#                 "status": f"Download error: {e}"}

#     # Save to temp file (YOLO needs a path)
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
#             img.save(f.name, format="JPEG")
#             tmp = f.name

#         result = detector.predict(img, tmp)
#     except Exception as e:
#         print(f"{prefix} ✗ {name}: inference error — {e}")
#         result = {"gender": "Unknown", "age_group": "Unknown",
#                   "gender_confidence": "", "age_confidence": "",
#                   "status": f"Inference error: {e}"}
#     finally:
#         try:
#             os.unlink(tmp)
#         except Exception:
#             pass

#     # Print progress line
#     if result["status"] == "Success":
#         print(f"{prefix} ✓ {name}: {result['gender']} "
#               f"({result['gender_confidence']}%)  "
#               f"age {result['age_group']} ({result['age_confidence']}%)")
#     else:
#         sym = "○" if "No person" in result["status"] or "Low" in result["status"] else "✗"
#         print(f"{prefix} {sym} {name}: {result['status']}")

#     return {"url": url, "name": name, **result}

# # ── URL loading ───────────────────────────────────────────────────────────────

# def load_urls() -> list[str]:
#     if os.path.exists(INPUT_CSV):
#         urls = []
#         with open(INPUT_CSV, newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             if "image_url" in (reader.fieldnames or []):
#                 for row in reader:
#                     u = row["image_url"].strip()
#                     if u:
#                         urls.append(u)
#                 print(f"✓ Loaded {len(urls)} URLs from {INPUT_CSV}")
#                 return urls
#         print(f"Warning: '{INPUT_CSV}' has no 'image_url' column — using hardcoded list.")
#     return list(URLS)

# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     t0 = time.time()

#     # 1. URLs
#     all_urls = load_urls()

#     # Deduplicate
#     seen, unique = set(), []
#     for u in all_urls:
#         if u not in seen:
#             unique.append(u); seen.add(u)
#     if len(unique) < len(all_urls):
#         print(f"✓ Removed {len(all_urls) - len(unique)} duplicate URLs")
#     all_urls = unique

#     # 2. Resume: skip already processed
#     processed     = load_processed_urls()
#     to_process    = [u for u in all_urls if u not in processed]
#     skipped       = len(all_urls) - len(to_process)
#     if skipped:
#         print(f"✓ Skipping {skipped} already-processed URLs")
#     if not to_process:
#         print("Nothing to process — all URLs are in the output CSV already.")
#         return

#     print(f"URLs to process: {len(to_process)}\n")

#     # 3. Init output CSV
#     if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
#         with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
#             csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
#         print(f"✓ Created {OUTPUT_CSV}")
#     else:
#         print(f"✓ Appending to existing {OUTPUT_CSV}")

#     # 4. Load models
#     detector = Detector()

#     # 5. Process in parallel
#     print(f"{'─'*60}")
#     print(f"Processing {len(to_process)} images  |  {MAX_WORKERS} threads")
#     print(f"{'─'*60}")

#     summary = {"Female": 0, "Male": 0, "Unknown": 0, "Error": 0}
#     total   = len(to_process)

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#         futures = {
#             pool.submit(process_url, url, detector, i + 1, total): url
#             for i, url in enumerate(to_process)
#         }
#         for future in as_completed(futures):
#             try:
#                 row = future.result()
#                 write_row(row)
#                 g = row["gender"]
#                 if "error" in row["status"].lower():
#                     summary["Error"] += 1
#                 elif g in summary:
#                     summary[g] += 1
#                 else:
#                     summary["Unknown"] += 1
#             except Exception as e:
#                 print(f"✗ Unexpected thread error: {e}")
#                 summary["Error"] += 1

#     # 6. Summary
#     elapsed = time.time() - t0
#     avg     = elapsed / total if total else 0
#     print(f"\n{'═'*60}")
#     print("SUMMARY")
#     print(f"{'═'*60}")
#     print(f"  Female   : {summary['Female']}")
#     print(f"  Male     : {summary['Male']}")
#     print(f"  Unknown  : {summary['Unknown']}")
#     print(f"  Error    : {summary['Error']}")
#     print(f"  ─────────")
#     print(f"  Total    : {total}")
#     print(f"\n  Time     : {elapsed:.1f}s  (avg {avg:.1f}s / image)")
#     print(f"  Output   : {OUTPUT_CSV}")
#     print(f"{'═'*60}\n")


# if __name__ == "__main__":
#     main()


"""
gender_detect_final.py
══════════════════════════════════════════════════════════════════════════════
Detects gender (and age) from profile image URLs using:
  • YOLOv8n       – person detection (confirms a human is in the photo)
  • FairFace      – gender classification  (~93% accuracy, via HuggingFace)
  • FairFace      – age classification     (~59% accuracy, via HuggingFace)

NO API KEY REQUIRED.  Models are free and download automatically on first run.

── Setup ────────────────────────────────────────────────────────────────────
  pip install ultralytics transformers torch pillow requests

── Run ──────────────────────────────────────────────────────────────────────
  python gender_detect_final.py

  Input  : hardcoded URLS list below  OR  create image_urls.csv with column "image_url"
  Output : gender_results.csv   (appends; safe to resume after interruption)

── First-run downloads (one time only) ─────────────────────────────────────
  yolov8n.pt                               ~6 MB   (Ultralytics CDN)
  dima806/fairface_gender_image_detection  ~100 MB (HuggingFace)
  dima806/fairface_age_image_detection     ~100 MB (HuggingFace)
══════════════════════════════════════════════════════════════════════════════
"""

import os
import csv
import time
import warnings
import logging
import tempfile
import requests
from io import BytesIO
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Silence noisy log output ──────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_CSV  = "gender_results.csv"
INPUT_CSV   = "image_urls.csv"      # optional: CSV with "image_url" column

# Paste your image URLs here (used when image_urls.csv is not found):
# URLS = [
#     "https://assets.veelapp.com/focusonmore.jpg",
#     "https://assets.veelapp.com/carolynjarrettjones.jpg",
#     "https://assets.veelapp.com/baileyjst.jpg",
#     "https://assets.veelapp.com/esteelbridal.jpg",
#     "https://assets.veelapp.com/theosbornfour.jpg",
#     "https://assets.veelapp.com/effiespaper.jpg",
#     "https://assets.veelapp.com/bingeon.travel.jpg",
#     "https://assets.veelapp.com/babynamesunday.jpg",
#     "https://assets.veelapp.com/thestumpshop.jpg",
#     "https://assets.veelapp.com/freysbaking.jpg",
#     "https://assets.veelapp.com/hairbybiancarose.jpg",
#     "https://assets.veelapp.com/andru.model.jpg",
#     "https://assets.veelapp.com/mylifeinmotherhood.jpg",
#     "https://assets.veelapp.com/the_google_pro.jpg",
#     "https://assets.veelapp.com/_notjustjess.jpg",
#     "https://assets.veelapp.com/robsmopolitan.jpg",
#     "https://assets.veelapp.com/staycsmart.jpg",
#     "https://assets.veelapp.com/oscarbravo.jpg",
#     "https://assets.veelapp.com/maluvoss.jpg",
#     "https://assets.veelapp.com/mypracticehome.jpg",
# ]

MAX_WORKERS             = 4     # parallel download+inference threads
IMAGE_TIMEOUT           = 15    # seconds to wait for image download
FACE_CONF_THRESHOLD     = 0.50  # min YOLO confidence to accept a "person"
GENDER_CONF_THRESHOLD   = 0.50  # min FairFace confidence to report a gender

# ── Age label → readable group ────────────────────────────────────────────────

AGE_MAP = {
    "0-2":         "(0-2)",
    "3-9":         "(4-6)",
    "10-19":       "(8-12)",
    "20-29":       "(25-32)",
    "30-39":       "(38-43)",
    "40-49":       "(48-53)",
    "50-59":       "(48-53)",
    "60-69":       "(60-100)",
    "more than 70":"(60-100)",
}

def map_age(label: str) -> str:
    return AGE_MAP.get(label, "Unknown")

# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDNAMES = ["url", "name", "gender", "age_group",
              "gender_confidence", "age_confidence", "status"]

csv_lock = Lock()

def write_row(row: dict) -> None:
    with csv_lock:
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

def load_processed_urls() -> set:
    processed = set()
    if not os.path.exists(OUTPUT_CSV):
        return processed
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = row.get("url", "").strip()
            if u:
                processed.add(u)
    return processed

def url_to_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]

# ── Model loader ──────────────────────────────────────────────────────────────

class Detector:
    """Loads YOLOv8 + FairFace models once; then processes images."""

    def __init__(self):
        print("─" * 60)
        print("Loading models (first run downloads ~200 MB total)")
        print("─" * 60)

        print("  [1/3] YOLOv8n  (person detection) …", end=" ", flush=True)
        self.yolo = YOLO("yolov8n.pt")
        print("✓")

        print("  [2/3] FairFace gender model …", end=" ", flush=True)
        self.gender = pipeline(
            "image-classification",
            model="dima806/fairface_gender_image_detection",
            device=-1,
        )
        print("✓")

        print("  [3/3] FairFace age model …", end=" ", flush=True)
        self.age = pipeline(
            "image-classification",
            model="dima806/fairface_age_image_detection",
            device=-1,
        )
        print("✓")
        print()

    # ── person detection ──────────────────────────────────────────────────────

    def has_person(self, image_path: str) -> tuple[bool, float]:
        """Return (found, best_confidence) for COCO class 0 = person."""
        results = self.yolo(image_path, verbose=False)
        best = 0.0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    conf = float(box.conf[0])
                    if conf > best:
                        best = conf
        return best >= FACE_CONF_THRESHOLD, best

    # ── main predict ─────────────────────────────────────────────────────────

    def predict(self, img: Image.Image, tmp_path: str) -> dict:
        """
        Run person detection then gender/age classification.
        Returns a result dict with keys:
            gender, age_group, gender_confidence, age_confidence, status
        """
        # 1. Person detection
        found, conf = self.has_person(tmp_path)
        if not found:
            return {
                "gender": "Unknown",
                "age_group": "Unknown",
                "gender_confidence": 0.0,
                "age_confidence": 0.0,
                "status": f"No person detected (YOLO conf {conf:.1%})",
            }

        # 2. Gender
        g_results = self.gender(img)
        g_label   = g_results[0]["label"].capitalize()   # 'Male' / 'Female'
        g_conf    = g_results[0]["score"] * 100

        if g_conf < GENDER_CONF_THRESHOLD * 100:
            return {
                "gender": "Unknown",
                "age_group": "Unknown",
                "gender_confidence": round(g_conf, 1),
                "age_confidence": 0.0,
                "status": f"Low gender confidence ({g_conf:.1f}%)",
            }

        # 3. Age
        a_results = self.age(img)
        a_label   = a_results[0]["label"]
        a_conf    = a_results[0]["score"] * 100

        return {
            "gender": g_label,
            "age_group": map_age(a_label),
            "gender_confidence": round(g_conf, 1),
            "age_confidence": round(a_conf, 1),
            "status": "Success",
        }

# ── Per-URL worker ────────────────────────────────────────────────────────────

def process_url(url: str, detector: Detector, idx: int, total: int) -> dict:
    name = url_to_name(url)
    prefix = f"[{idx:>3}/{total}]"

    # Download image
    try:
        resp = requests.get(url, timeout=IMAGE_TIMEOUT)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"{prefix} ✗ {name}: download failed — {e}")
        return {"url": url, "name": name,
                "gender": "Unknown", "age_group": "Unknown",
                "gender_confidence": "", "age_confidence": "",
                "status": f"Download error: {e}"}

    # Save to temp file (YOLO needs a path)
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, format="JPEG")
            tmp = f.name

        result = detector.predict(img, tmp)
    except Exception as e:
        print(f"{prefix} ✗ {name}: inference error — {e}")
        result = {"gender": "Unknown", "age_group": "Unknown",
                  "gender_confidence": "", "age_confidence": "",
                  "status": f"Inference error: {e}"}
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

    # Print progress line
    if result["status"] == "Success":
        print(f"{prefix} ✓ {name}: {result['gender']} "
              f"({result['gender_confidence']}%)  "
              f"age {result['age_group']} ({result['age_confidence']}%)")
    else:
        sym = "○" if "No person" in result["status"] or "Low" in result["status"] else "✗"
        print(f"{prefix} {sym} {name}: {result['status']}")

    return {"url": url, "name": name, **result}

# ── URL loading ───────────────────────────────────────────────────────────────

def load_urls() -> list[str]:
    if os.path.exists(INPUT_CSV):
        urls = []
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "image_url" in (reader.fieldnames or []):
                for row in reader:
                    u = row["image_url"].strip()
                    if u:
                        urls.append(u)
                print(f"✓ Loaded {len(urls)} URLs from {INPUT_CSV}")
                return urls
        print(f"Warning: '{INPUT_CSV}' has no 'image_url' column — using hardcoded list.")
    return list(URLS)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. URLs
    all_urls = load_urls()

    # Deduplicate
    seen, unique = set(), []
    for u in all_urls:
        if u not in seen:
            unique.append(u); seen.add(u)
    if len(unique) < len(all_urls):
        print(f"✓ Removed {len(all_urls) - len(unique)} duplicate URLs")
    all_urls = unique

    # 2. Resume: skip already processed
    processed     = load_processed_urls()
    to_process    = [u for u in all_urls if u not in processed]
    skipped       = len(all_urls) - len(to_process)
    if skipped:
        print(f"✓ Skipping {skipped} already-processed URLs")
    if not to_process:
        print("Nothing to process — all URLs are in the output CSV already.")
        return

    print(f"URLs to process: {len(to_process)}\n")

    # 3. Init output CSV
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print(f"✓ Created {OUTPUT_CSV}")
    else:
        print(f"✓ Appending to existing {OUTPUT_CSV}")

    # 4. Load models
    detector = Detector()

    # 5. Process in parallel
    print(f"{'─'*60}")
    print(f"Processing {len(to_process)} images  |  {MAX_WORKERS} threads")
    print(f"{'─'*60}")

    summary = {"Female": 0, "Male": 0, "Unknown": 0, "Error": 0}
    total   = len(to_process)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_url, url, detector, i + 1, total): url
            for i, url in enumerate(to_process)
        }
        for future in as_completed(futures):
            try:
                row = future.result()
                write_row(row)
                g = row["gender"]
                if "error" in row["status"].lower():
                    summary["Error"] += 1
                elif g in summary:
                    summary[g] += 1
                else:
                    summary["Unknown"] += 1
            except Exception as e:
                print(f"✗ Unexpected thread error: {e}")
                summary["Error"] += 1

    # 6. Summary
    elapsed = time.time() - t0
    avg     = elapsed / total if total else 0
    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    print(f"  Female   : {summary['Female']}")
    print(f"  Male     : {summary['Male']}")
    print(f"  Unknown  : {summary['Unknown']}")
    print(f"  Error    : {summary['Error']}")
    print(f"  ─────────")
    print(f"  Total    : {total}")
    print(f"\n  Time     : {elapsed:.1f}s  (avg {avg:.1f}s / image)")
    print(f"  Output   : {OUTPUT_CSV}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()