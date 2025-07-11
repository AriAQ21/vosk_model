import sys
import os
import time
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

        # Count words
        word_count = len(transcript.split())
        total_words += word_count

        filename = os.path.basename(audio_path).replace(".wav", ".txt")
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(transcript)

        elapsed = time.time() - start
        print(f"Processed {filename} in {elapsed:.2f} s, {word_count} words, audio length: {duration:.2f} s")

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
