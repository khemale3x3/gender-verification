import os
import csv
from PIL import Image
import warnings
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress transformers warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from transformers import pipeline
from ultralytics import YOLO

# Configuration
PROFILE_FOLDER = '/var/www/photos'
INPUT_CSV = '/var/www/G2/usernames.csv'  # CSV with usernames to process
OUTPUT_CSV = '/var/www/G2/user_demographics.csv'
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Performance settings
MAX_WORKERS = 4  # Number of parallel threads (adjust based on CPU cores)
BATCH_SIZE = 10  # Number of images to process before saving checkpoint

# Thresholds
FACE_CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence for face detection
GENDER_CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence for gender prediction

# Thread-safe lock for CSV writing
csv_lock = Lock()

# Age group mapping for FairFace output
def map_fairface_age_to_group(age_label):
    """Map FairFace age labels to our age groups"""
    age_mapping = {
        '0-2': '(0-2)',
        '3-9': '(4-6)',
        '10-19': '(8-12)',
        '20-29': '(25-32)',
        '30-39': '(38-43)',
        '40-49': '(48-53)',
        '50-59': '(48-53)',
        '60-69': '(60-100)',
        'more than 70': '(60-100)'
    }
    return age_mapping.get(age_label, '(60-100)')

class GenderAgeDetector:
    """Gender and Age detector using YOLOv8 and FairFace models"""
    
    def __init__(self):
        print("Loading models...")
        print("This may take a moment on first run (downloading models)...\n")
        
        try:
            # Load YOLOv8 nano model (fastest, smallest)
            print("Loading YOLOv8n model for object detection...")
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLOv8n model loaded")
            
            # Load FairFace gender classification model
            print("Loading FairFace gender classification model...")
            self.gender_classifier = pipeline(
                "image-classification",
                model="dima806/fairface_gender_image_detection",
                device=-1  # CPU
            )
            print("✓ Gender classification model loaded (93.4% accuracy)")
            
            # Load FairFace age classification model
            print("Loading FairFace age classification model...")
            self.age_classifier = pipeline(
                "image-classification",
                model="dima806/fairface_age_image_detection",
                device=-1  # CPU
            )
            print("✓ Age classification model loaded (59% accuracy)\n")
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            print("Make sure you have:")
            print("  1. Internet connection for first-time model download")
            print("  2. ultralytics package installed: pip install ultralytics")
            raise
    
    def is_human_face(self, image_path):
        """Check if image contains a human face/person using YOLOv8"""
        try:
            # Run YOLOv8 inference
            results = self.yolo_model(image_path, verbose=False)
            
            # Check if any person is detected with good confidence
            person_detected = False
            max_confidence = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Class 0 is 'person' in COCO dataset
                    if int(box.cls[0]) == 0:
                        confidence = float(box.conf[0])
                        max_confidence = max(max_confidence, confidence)
                        if confidence >= FACE_CONFIDENCE_THRESHOLD:
                            person_detected = True
                            break
                
                if person_detected:
                    break
            
            return person_detected, max_confidence
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return False, 0
    
    def predict(self, image_path):
        """Predict gender and age from image using FairFace models"""
        try:
            # First check if image contains a person
            is_person, person_confidence = self.is_human_face(image_path)
            
            if not is_person:
                return {
                    'gender': 'Unknown',
                    'age_group': 'Unknown',
                    'gender_confidence': 0,
                    'age_confidence': 0,
                    'age_label': '',
                    'success': True,
                    'error': None,
                    'reason': f'No human detected (confidence: {person_confidence:.1%})'
                }
            
            # Load image for classification
            image = Image.open(image_path).convert('RGB')
            
            # Predict gender
            gender_results = self.gender_classifier(image)
            gender_pred = gender_results[0]
            gender = gender_pred['label'].capitalize()
            gender_confidence = gender_pred['score'] * 100
            
            # Check if gender confidence is too low
            if gender_confidence < GENDER_CONFIDENCE_THRESHOLD * 100:
                return {
                    'gender': 'Unknown',
                    'age_group': 'Unknown',
                    'gender_confidence': gender_confidence,
                    'age_confidence': 0,
                    'age_label': '',
                    'success': True,
                    'error': None,
                    'reason': f'Low gender confidence ({gender_confidence:.1f}%)'
                }
            
            # Predict age
            age_results = self.age_classifier(image)
            age_pred = age_results[0]
            age_label = age_pred['label']
            age_confidence = age_pred['score'] * 100
            
            # Map to our age groups
            age_group = map_fairface_age_to_group(age_label)
            
            return {
                'gender': gender,
                'age_group': age_group,
                'gender_confidence': gender_confidence,
                'age_confidence': age_confidence,
                'age_label': age_label,
                'success': True,
                'error': None,
                'reason': 'Success'
            }
            
        except Exception as e:
            return {
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'gender_confidence': 0,
                'age_confidence': 0,
                'age_label': '',
                'success': False,
                'error': str(e),
                'reason': f'Error: {str(e)}'
            }

def load_usernames_from_csv(csv_path):
    """Load usernames from CSV file"""
    usernames = []
    
    if not os.path.exists(csv_path):
        print(f"Warning: Input CSV '{csv_path}' not found!")
        print("Expected CSV format:")
        print("  username")
        print("  john_doe")
        print("  jane_smith")
        return usernames
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if 'username' column exists
            if 'username' not in reader.fieldnames:
                print(f"Error: CSV must have 'username' column!")
                print(f"Found columns: {reader.fieldnames}")
                return usernames
            
            for row in reader:
                username = row['username'].strip()
                if username:  # Skip empty usernames
                    usernames.append(username)
        
        print(f"✓ Loaded {len(usernames)} usernames from CSV")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return usernames

def get_already_processed_usernames(output_csv):
    """Get set of usernames already processed from output CSV"""
    processed = set()
    
    if not os.path.exists(output_csv):
        return processed
    
    try:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = row.get('username', '').strip()
                if username:
                    processed.add(username)
        
        print(f"✓ Found {len(processed)} already processed usernames")
        
    except Exception as e:
        print(f"Warning: Could not read existing results: {e}")
    
    return processed

def find_image_for_username(username, profile_folder):
    """Find image file for a given username"""
    # Try different extensions
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(profile_folder, f"{username}{ext}")
        if os.path.exists(image_path):
            return image_path
    
    return None

def process_image(username, image_path, detector, idx, total):
    """Process a single image"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"[{idx}/{total}] ✗ {username}: Image not found")
            return {
                'username': username,
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': '',
                'status': 'Image not found'
            }
        
        # Predict
        result = detector.predict(image_path)
        
        if result['success'] and result['gender'] != 'Unknown':
            print(f"[{idx}/{total}] ✓ {username}: {result['gender']} ({result['gender_confidence']:.1f}%), "
                  f"Age: {result['age_group']} ({result['age_confidence']:.1f}%)")
            
            return {
                'username': username,
                'gender': result['gender'],
                'age_group': result['age_group'],
                'confidence': f"Gender: {result['gender_confidence']:.1f}%, Age: {result['age_confidence']:.1f}%",
                'status': 'Success'
            }
        elif result['gender'] == 'Unknown':
            print(f"[{idx}/{total}] ○ {username}: Unknown - {result['reason']}")
            return {
                'username': username,
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': '',
                'status': result['reason']
            }
        else:
            print(f"[{idx}/{total}] ✗ {username}: {result['error']}")
            return {
                'username': username,
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': '',
                'status': f"Error: {result['error']}"
            }
            
    except Exception as e:
        print(f"[{idx}/{total}] ✗ {username}: Unexpected error - {str(e)}")
        return {
            'username': username,
            'gender': 'Unknown',
            'age_group': 'Unknown',
            'confidence': '',
            'status': f'Error: {str(e)}'
        }

def write_result_to_csv(result, output_csv, fieldnames):
    """Thread-safe CSV writing"""
    with csv_lock:
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

def main():
    """Main function with parallel processing"""
    
    start_time = time.time()
    
    # Check if profile folder exists
    if not os.path.exists(PROFILE_FOLDER):
        print(f"Error: '{PROFILE_FOLDER}' folder not found!")
        print(f"Please create a '{PROFILE_FOLDER}' folder and add images to it.")
        return
    
    # Load usernames from CSV
    print(f"{'='*70}")
    print(f"STEP 1: Loading usernames from CSV")
    print(f"{'='*70}\n")
    
    usernames = load_usernames_from_csv(INPUT_CSV)
    
    if not usernames:
        print(f"\nNo usernames found in '{INPUT_CSV}'!")
        print("Please create a CSV file with format:")
        print("  username")
        print("  john_doe")
        print("  jane_smith")
        return
    
    # Remove duplicates while preserving order
    unique_usernames = []
    seen = set()
    for username in usernames:
        if username not in seen:
            unique_usernames.append(username)
            seen.add(username)
    
    duplicates_removed = len(usernames) - len(unique_usernames)
    if duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed} duplicate entries")
    
    usernames = unique_usernames
    
    # Get already processed usernames
    print(f"\n{'='*70}")
    print(f"STEP 2: Checking for already processed usernames")
    print(f"{'='*70}\n")
    
    already_processed = get_already_processed_usernames(OUTPUT_CSV)
    
    # Filter out already processed usernames
    usernames_to_process = [u for u in usernames if u not in already_processed]
    skipped_count = len(usernames) - len(usernames_to_process)
    
    if skipped_count > 0:
        print(f"✓ Skipping {skipped_count} already processed usernames")
    
    if not usernames_to_process:
        print(f"\n✓ All usernames have already been processed!")
        print(f"Results are in: {OUTPUT_CSV}")
        return
    
    # Find images for usernames
    print(f"\n{'='*70}")
    print(f"STEP 3: Finding images for usernames")
    print(f"{'='*70}\n")
    
    username_image_pairs = []
    missing_images = []
    
    for username in usernames_to_process:
        image_path = find_image_for_username(username, PROFILE_FOLDER)
        if image_path:
            username_image_pairs.append((username, image_path))
        else:
            missing_images.append(username)
    
    print(f"✓ Found images for {len(username_image_pairs)} usernames")
    if missing_images:
        print(f"○ Missing images for {len(missing_images)} usernames")
    
    if not username_image_pairs:
        print(f"\nNo images found for any usernames!")
        return
    
    # Initialize detector
    print(f"\n{'='*70}")
    print(f"STEP 4: Loading AI models")
    print(f"{'='*70}\n")
    
    try:
        detector = GenderAgeDetector()
    except Exception as e:
        print(f"\nFailed to initialize detector. Please check your internet connection.")
        return
    
    # Initialize CSV file with headers if it doesn't exist
    fieldnames = ['username', 'gender', 'age_group', 'confidence', 'status']
    
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"✓ CSV file initialized: {OUTPUT_CSV}")
    else:
        print(f"✓ Appending to existing CSV: {OUTPUT_CSV}")
    
    # Process images in parallel
    print(f"\n{'='*70}")
    print(f"STEP 5: Processing images (Parallel: {MAX_WORKERS} threads)")
    print(f"{'='*70}\n")
    print(f"Processing {len(username_image_pairs)} images...")
    print(f"Real-time output: {OUTPUT_CSV}\n")
    
    results_summary = {'successful': 0, 'unknown': 0, 'failed': 0}
    
    # Process with ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_username = {
            executor.submit(
                process_image, 
                username, 
                image_path, 
                detector, 
                idx + 1, 
                len(username_image_pairs)
            ): username
            for idx, (username, image_path) in enumerate(username_image_pairs)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_username):
            username = future_to_username[future]
            try:
                result = future.result()
                
                # Write result to CSV immediately
                write_result_to_csv(result, OUTPUT_CSV, fieldnames)
                
                # Update summary counts
                if result['status'] == 'Success':
                    results_summary['successful'] += 1
                elif result['gender'] == 'Unknown' and 'Error' not in result['status']:
                    results_summary['unknown'] += 1
                else:
                    results_summary['failed'] += 1
                
            except Exception as e:
                print(f"✗ {username}: Task failed with exception: {e}")
                results_summary['failed'] += 1
    
    # Process missing images (write to CSV)
    if missing_images:
        print(f"\nRecording {len(missing_images)} missing images...")
        for username in missing_images:
            result = {
                'username': username,
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': '',
                'status': 'Image file not found'
            }
            write_result_to_csv(result, OUTPUT_CSV, fieldnames)
            results_summary['failed'] += 1
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total usernames in CSV: {len(usernames) + skipped_count}")
    print(f"Already processed (skipped): {skipped_count}")
    print(f"Newly processed: {len(username_image_pairs) + len(missing_images)}")
    print(f"\nResults breakdown:")
    print(f"  ✓ Successful detections: {results_summary['successful']}")
    print(f"  ○ Unknown (non-human/low confidence): {results_summary['unknown']}")
    print(f"  ✗ Failed/Missing: {results_summary['failed']}")
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/(len(username_image_pairs) + len(missing_images)):.2f} seconds")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()