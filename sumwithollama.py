import subprocess
import json

def summarize_with_ollama(text):
    prompt = f"Summarize this transcript in a few bullet points:\n\n{text}"
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

if __name__ == "__main__":
    transcript = "This is the text you got from Whisper or Vosk..."
    summary = summarize_with_ollama(transcript)
    print("Summary:\n", summary)
