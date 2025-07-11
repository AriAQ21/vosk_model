import sys
import wave
import json
from vosk import Model, KaldiRecognizer

def transcribe(model_path, audio_file):
    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    full_text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            full_text.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    full_text.append(final_result.get("text", ""))
    return " ".join(full_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model_path> <audio_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]
    try:
        transcript = transcribe(model_path, audio_file)
        print(transcript)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
