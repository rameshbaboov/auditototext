import time
import logging
import subprocess
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
DONE_DIR = BASE_DIR / "temp_done"
LOG_PATH = BASE_DIR / "ollama_summarizer.log"

PACKAGE_NAME = "audiototext"

POLL_INTERVAL = 5  # seconds
OLLAMA_MODEL = "gemma2:2b"


def setup_dirs():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)


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
    cfg = load_config()
    pkg = cfg.get("packages", {}).get(PACKAGE_NAME, {})
    return str(pkg.get("sleep", "y")).lower() == "y"


def summarize_text(text: str) -> str:
    prompt = (
        "Summarize this transcript in clear bullet points. "
        "Focus on key decisions, action items, and important details:\n\n"
        + text
    )
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ollama failed: {err}")
    return result.stdout.decode("utf-8", errors="ignore")


def summarize_file(transcript_path: Path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logging.info("Empty transcript in %s, skipping", transcript_path.name)
        print(f"[ollama] Empty transcript in {transcript_path.name}, skipping")
        return

    summary = summarize_text(text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{transcript_path.stem}_summary.txt"
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(summary)

    logging.info("Summarized %s -> %s", transcript_path.name, out_path.name)
    print(f"[ollama] Summarized {transcript_path.name} -> {out_path.name}")


def main_loop():
    setup_dirs()
    logging.info("ollama summarizer started.")
    print("[ollama] summarizer started, watching temp/")

    while True:
        try:
            if package_sleeping():
                logging.info("Package sleep flag is 'y'; summarizer sleeping.")
                print("[ollama] sleeping (package sleep = y)")
                time.sleep(POLL_INTERVAL)
                continue

            logging.info("Package awake; summarizer running.")
            print("[ollama] awake; checking for transcripts...")

            txt_files = [
                p for p in TEMP_DIR.iterdir() if p.is_file() and p.suffix == ".txt"
            ]
            if not txt_files:
                time.sleep(POLL_INTERVAL)
                continue

            for transcript_path in txt_files:
                try:
                    summarize_file(transcript_path)
                    done_path = DONE_DIR / transcript_path.name
                    transcript_path.rename(done_path)
                except Exception as e:
                    logging.exception("Error summarizing %s: %s", transcript_path, e)
                    print(f"[ollama] Error summarizing {transcript_path.name}: {e}")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            logging.exception("Main loop error: %s", e)
            print(f"[ollama] Main loop error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
