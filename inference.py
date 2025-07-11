import sys
import wave
import json
from vosk import Model, KaldiRecognizer

def main():
    if len(sys.argv) != 4:
        print("Usage: python inference.py <model_path> <audio_file> <transcript_output_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]
    transcript_file = sys.argv[3]

    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))

    # Join all partial results into full transcript
    full_transcript = " ".join(results).strip()

    # Write transcript to file
    with open(transcript_file, "w") as f:
        f.write(full_transcript)

    print(f"Transcription saved to {transcript_file}")

if __name__ == "__main__":
    main()
