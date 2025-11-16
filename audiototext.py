import sys
import soundfile as sf
from vosk import Model, KaldiRecognizer

def transcribe(audio_file):
    model = Model("vosk-model-small-en-us-0.15")  # download from vosk website first
    rec = KaldiRecognizer(model, 16000)

    with sf.SoundFile(audio_file) as f:
        for block in f.blocks(blocksize=8000):
            rec.AcceptWaveform(block.tobytes())
    return rec.FinalResult()

if __name__ == "__main__":
    file_path = "sample.wav"  # must be PCM WAV 16kHz mono
    print(transcribe(file_path))
