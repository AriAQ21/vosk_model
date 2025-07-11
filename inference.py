import sys
import wave
import json
from vosk import Model, KaldiRecognizer

def main():
    if len(sys.argv) != 4:
        print("Usage: python inference.py <model_path> <audio_file> <output_text_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]
    output_file = sys.argv[3]

    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    full_text = []

    print(f"Running inference on {audio_file} using model at {model_path}...\n")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            full_text.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    full_text.append(final_result.get("text", ""))

    transcript = " ".join(full_text).strip()
    print(transcript)

    with open(output_file, "w") as f:
        f.write(transcript + "\n")

if __name__ == "__main__":
    main()
