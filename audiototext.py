import os
import sys
import time
import json
import shutil
import logging
import subprocess
from pathlib import Path

import yaml
import soundfile as sf
from vosk import Model, KaldiRecognizer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

INPUT_DIR = BASE_DIR / "input"
TEMP_DIR = BASE_DIR / "temp"
PROCESSED_DIR = BASE_DIR / "processed"
LOG_PATH = BASE_DIR / "audiototext.log"

MODEL_PATH = BASE_DIR / "models" / "vosk-model-small-en-us-0.15"

PACKAGE_NAME = "audiototext"

POLL_INTERVAL = 5  # seconds
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def setup_dirs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def package_sleeping() -> bool:
    """
    Check package-level sleep flag: config["packages"][PACKAGE_NAME]["sleep"]
    Default is 'y' (sleep).
    """
    cfg = load_config()
    pkg = cfg.get("packages", {}).get(PACKAGE_NAME, {})
    return str(pkg.get("sleep", "y")).lower() == "y"


def to_wav_16k_mono(input_path: Path) -> Path:
    """
    Convert any ffmpeg-supported audio to 16 kHz mono s16 wav.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEMP_DIR / f"{input_path.stem}_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path


def transcribe_file(audio_path: Path) -> Path:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Vosk model folder not found: {MODEL_PATH}")

    wav_path = to_wav_16k_mono(audio_path)

    model = Model(str(MODEL_PATH))
    with sf.SoundFile(str(wav_path)) as f:
        if f.samplerate != 16000 or f.channels != 1:
            raise ValueError(
                f"Expected 16kHz mono after ffmpeg, got {f.samplerate} Hz / {f.channels} ch"
            )

        rec = KaldiRecognizer(model, f.samplerate)
        for block in f.blocks(blocksize=8000, dtype="int16"):
            rec.AcceptWaveform(block.tobytes())

        result = json.loads(rec.FinalResult())
        text = result.get("text", "")

    transcript_path = TEMP_DIR / f"{audio_path.stem}.txt"
    with open(transcript_path, "w", encoding="utf-8") as out_f:
        out_f.write(text)

    logging.info("Transcribed %s -> %s", audio_path.name, transcript_path.name)
    print(f"[audiototext] Transcribed {audio_path.name} -> {transcript_path.name}")
    return transcript_path


def main_loop():
    setup_dirs()
    logging.info("audiototext worker started.")
    print("[audiototext] worker started, watching input/")

    while True:
        try:
            if package_sleeping():
                logging.info("Package sleep flag is 'y'; worker sleeping.")
                # light console heartbeat so you see it's alive
                print("[audiototext] sleeping (package sleep = y)")
                time.sleep(POLL_INTERVAL)
                continue

            logging.info("Package awake; worker running.")
            print("[audiototext] awake; checking for audio files...")

            audio_files = [
                p
                for p in INPUT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            ]

            if not audio_files:
                time.sleep(POLL_INTERVAL)
                continue

            for audio_path in audio_files:
                try:
                    transcribe_file(audio_path)
                    # move original audio to processed
                    dest = PROCESSED_DIR / audio_path.name
                    shutil.move(str(audio_path), str(dest))
                except Exception as e:
                    logging.exception("Error processing %s: %s", audio_path, e)
                    print(f"[audiototext] Error processing {audio_path.name}: {e}")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            logging.exception("Main loop error: %s", e)
            print(f"[audiototext] Main loop error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
