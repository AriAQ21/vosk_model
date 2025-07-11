import sys
import os
import glob
from jiwer import wer

def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip().lower()

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <clean_transcripts_folder> <vosk_transcripts_folder>")
        sys.exit(1)

    clean_folder = sys.argv[1]
    vosk_folder = sys.argv[2]

    clean_files = glob.glob(os.path.join(clean_folder, "*.txt"))
    if not clean_files:
        print(f"No transcript files found in {clean_folder}")
        sys.exit(1)

    total_wer = 0.0
    count = 0

    for clean_file in clean_files:
        filename = os.path.basename(clean_file)
        vosk_file = os.path.join(vosk_folder, filename)

        if not os.path.exists(vosk_file):
            print(f"Warning: {filename} not found in Vosk transcripts folder, skipping.")
            continue

        clean_text = load_transcript(clean_file)
        vosk_text = load_transcript(vosk_file)

        error = wer(clean_text, vosk_text)
        print(f"{filename}: WER = {error:.3f}")

        total_wer += error
        count += 1

    if count > 0:
        print(f"\nAverage WER over {count} files: {total_wer/count:.3f}")
    else:
        print("No matching transcript files found to evaluate.")

if __name__ == "__main__":
    main()
