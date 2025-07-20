import sys
import wave
import json
import os
import time
import csv
from vosk import Model, KaldiRecognizer

def transcribe(model_path, audio_file):
    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    full_text = []
    start = time.time()
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            full_text.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    full_text.append(final_result.get("text", ""))
    elapsed = time.time() - start

    text = " ".join(full_text)
    return text, elapsed, wf.getnframes() / wf.getframerate()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python single_inference.py <model_path> <audio_file> <output_folder>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]
    output_folder = sys.argv[3]

    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    txt_path = os.path.join(output_folder, f"{base_name}.txt")
    metrics_path = os.path.join(output_folder, "metrics.csv")

    try:
        transcript, elapsed, duration = transcribe(model_path, audio_file)
        with open(txt_path, "w") as f:
            f.write(transcript)

        word_count = len(transcript.split())
        wps = word_count / elapsed if elapsed > 0 else 0

        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["file", "duration_sec", "words", "time_sec", "wps"])
            writer.writerow([f"{base_name}.wav", round(duration, 2), word_count, round(elapsed, 2), round(wps, 2)])

        print(f"Transcribed {base_name}.wav")
        print(f"Transcript saved to: {txt_path}")
        print(f"Metrics appended to: {metrics_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
