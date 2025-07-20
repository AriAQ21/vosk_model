import sys
import wave
import json
import os
import time
import torchaudio
from vosk import Model, KaldiRecognizer

def transcribe(model_path, audio_file):
    # === Load and convert audio using torchaudio ===
    waveform, sr = torchaudio.load(audio_file)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16000 Hz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Save as PCM 16-bit WAV file for compatibility with wave module
    temp_wav = "/tmp/temp_vosk_input.wav"
    waveform = waveform.mul(32767).clamp(-32768, 32767).to(torch.int16)
    torchaudio.save(temp_wav, waveform, 16000, encoding="PCM_S", bits_per_sample=16)

    # Open with wave module (Vosk requirement)
    wf = wave.open(temp_wav, "rb")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

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
    transcript = " ".join(full_text)

    return transcript, elapsed, wf.getnframes() / wf.getframerate()

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
            import csv
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["file", "duration_sec", "words", "time_sec", "wps"])
            writer.writerow([f"{base_name}.wav", round(duration, 2), word_count, round(elapsed, 2), round(wps, 2)])

        print(f"✅ Transcribed {base_name}.wav")
        print(f"📄 Transcript saved to: {txt_path}")
        print(f"📝 Metrics appended to: {metrics_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
