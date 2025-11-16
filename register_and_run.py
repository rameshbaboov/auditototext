import subprocess
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

PACKAGE_NAME = "audiototext"

CHECK_INTERVAL = 5      # seconds between checks
GRACE_TIMEOUT = 10      # seconds before force kill after terminate()


def get_package_definition():
    """
    main.py imports this to know what this package is.
    main.py only cares about 'name' and 'description'.
    """
    return {
        "name": PACKAGE_NAME,
        "description": "Audio to text + summary pipeline",
    }


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_package_cfg():
    cfg = load_config()
    return cfg.get("packages", {}).get(PACKAGE_NAME, {})


def get_python_cmd_for_package() -> str:
    """Read venv_path.txt if present, otherwise fall back to python3."""
    venv_file = BASE_DIR / "venv_path.txt"
    if venv_file.exists():
        try:
            cmd = venv_file.read_text(encoding="utf-8").strip()
            if cmd:
                return cmd
        except Exception:
            pass
    return "python3"


def main():
    python_cmd = get_python_cmd_for_package()

    workers = {
        "audiototext": {
            "script": "audiototext.py",
            "proc": None,
            "start_time": None,
            "stop_requested_at": None,
        },
        "ollama": {
            "script": "sumwithollama.py",
            "proc": None,
            "start_time": None,
            "stop_requested_at": None,
        },
    }

    print("[register_and_run] supervisor started.")

    while True:
        pkg_cfg = get_package_cfg()
        sleep_flag = str(pkg_cfg.get("sleep", "y")).lower()
        max_run_seconds = int(pkg_cfg.get("max_run_seconds", 0) or 0)

        now = time.time()

        if sleep_flag == "n":
            print("[register_and_run] package sleep='n' (awake).")

            for name, info in workers.items():
                proc = info["proc"]

                # start if not running
                if proc is None or proc.poll() is not None:
                    cmd = [python_cmd, info["script"]]
                    print(f"[register_and_run] starting worker {name}: {cmd}")
                    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
                    info["proc"] = proc
                    info["start_time"] = now
                    info["stop_requested_at"] = None
                    continue

                # enforce max run time
                if max_run_seconds > 0 and info["start_time"]:
                    runtime = now - info["start_time"]
                    if runtime > max_run_seconds:
                        if info["stop_requested_at"] is None:
                            print(
                                f"[register_and_run] worker {name} exceeded "
                                f"max_run_seconds={max_run_seconds}, terminating."
                            )
                            proc.terminate()
                            info["stop_requested_at"] = now
                        else:
                            if now - info["stop_requested_at"] > GRACE_TIMEOUT:
                                if proc.poll() is None:
                                    print(
                                        f"[register_and_run] worker {name} did not exit, killing."
                                    )
                                    proc.kill()
                                info["proc"] = None
                                info["start_time"] = None
                                info["stop_requested_at"] = None

        else:
            # sleep = 'y' -> ensure workers are stopped
            print("[register_and_run] package sleep='y' (should stop workers).")
            for name, info in workers.items():
                proc = info["proc"]
                if proc is None or proc.poll() is not None:
                    # already stopped
                    info["proc"] = None
                    info["start_time"] = None
                    info["stop_requested_at"] = None
                    continue

                if info["stop_requested_at"] is None:
                    print(f"[register_and_run] terminating worker {name}.")
                    proc.terminate()
                    info["stop_requested_at"] = now
                else:
                    if now - info["stop_requested_at"] > GRACE_TIMEOUT:
                        if proc.poll() is None:
                            print(
                                f"[register_and_run] worker {name} still alive, killing."
                            )
                            proc.kill()
                        info["proc"] = None
                        info["start_time"] = None
                        info["stop_requested_at"] = None

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
