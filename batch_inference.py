import sys
import os
import time
import csv
from inference import transcribe
import glob

def main():
    if len(sys.argv) != 5:
        print("Usage: python batch_inference.py <model_path> <audio_dir> <output_dir> <num_files>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_dir = sys.argv[2]
    output_dir = sys.argv[3]
    num_files = int(sys.argv[4])

    os.makedirs(output_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if len(audio_files) < num_files:
        print(f"Warning: only {len(audio_files)} WAV files found, less than requested {num_files}")

    selected_files = audio_files[:num_files]

    metrics_file = os.path.join(output_dir, "metrics.csv")
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'time_taken_s', 'word_count', 'audio_duration_s', 'words_per_second']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_start = time.time()
        total_words = 0
        total_audio_duration = 0.0

        for audio_path in selected_files:
            start = time.time()
            try:
                transcript = transcribe(model_path, audio_path)
            except Exception as e:
                print(f"Failed on {audio_path}: {e}")
                continue

            duration = get_wav_duration(audio_path)
            total_audio_duration += duration

            word_count = len(transcript.split())
            total_words += word_count

            filename = os.path.basename(audio_path).replace(".wav", ".txt")
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(transcript)

            elapsed = time.time() - start
            print(f"Processed {filename} in {elapsed:.2f} s, {word_count} words, audio length: {duration:.2f} s")

            writer.writerow({
                'filename': filename,
                'time_taken_s': f"{elapsed:.2f}",
                'word_count': word_count,
                'audio_duration_s': f"{duration:.2f}",
                'words_per_second': f"{(word_count / duration) if duration > 0 else 0:.2f}"
            })

        total_elapsed = time.time() - total_start
        print(f"\nTotal time: {total_elapsed:.2f} s for {len(selected_files)} files")
        print(f"Total words: {total_words}, total audio duration: {total_audio_duration:.2f} s")
        if total_audio_duration > 0:
            print(f"Words per second of audio: {total_words / total_audio_duration:.2f}")

def get_wav_duration(wav_path):
    import wave
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

if __name__ == "__main__":
    main()
