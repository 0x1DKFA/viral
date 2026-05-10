import os
import sys
import shutil

# Add project root to sys.path to allow running as 'python3 src/main.py'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.processor import VideoProcessor
from src.analyzer import VLMAnalyzer
from tqdm import tqdm

load_dotenv()

def main():
    # Load config
    raw_dir = os.getenv("RAW_RECORDINGS_DIR", "./raw_recordings")
    processed_dir = os.getenv("PROCESSED_DIR", "./processed")
    segments_dir = os.getenv("SEGMENTS_DIR", "./segments")
    output_dir = os.getenv("OUTPUT_DIR", "./output/clips")
    threshold = int(os.getenv("VIRAL_THRESHOLD", "7"))
    model_id = os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-7B-Instruct")

    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    processor = VideoProcessor(segments_dir=segments_dir)
    # Lazy load analyzer only if we find files to process
    analyzer = None

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".mp4")]
    if not raw_files:
        print(f"No recordings found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} recordings. Starting pipeline...")

    for file in raw_files:
        input_path = os.path.join(raw_dir, file)
        print(f"\nProcessing {file}...")
        
        # 1. Segment the video
        print("Slicing video into segments...")
        segments = processor.segment(input_path)
        
        if not segments:
            print("No segments created.")
            continue

        # 2. Analyze segments
        if analyzer is None:
            analyzer = VLMAnalyzer(model_id=model_id)

        print(f"Analyzing {len(segments)} segments with {model_id}...")
        for seg_path in tqdm(segments, desc="Analyzing"):
            result = analyzer.analyze(seg_path)
            score = result.get('score', 0)
            center_x = result.get('action_center_x', 0.5)

            # 3. Filter and Crop
            if score >= threshold:
                print(f"\nFound highlight! Score: {score}. Cropping...")
                output_filename = f"highlight_{score}_{os.path.basename(seg_path)}"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    processor.crop_9_16(seg_path, output_path, center_x)
                    print(f"Saved to {output_path}")
                except Exception as e:
                    print(f"Failed to crop {seg_path}: {e}")

        # 4. Cleanup and Archive
        print(f"Archiving {file}...")
        shutil.move(input_path, os.path.join(processed_dir, file))
        
        # Clear segments for next file
        for s in segments:
            if os.path.exists(s):
                os.remove(s)

    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
