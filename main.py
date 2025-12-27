from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from audiototext import INPUT_DIR, AUDIO_EXTS, transcribe_file
from sumwithollama import summarize_file, OUTPUT_DIR

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Audio to Text + Summary API")


@app.get("/")
def root():
    return {"app": "audiototext", "status": "ok"}


async def _process_uploaded_file(file: UploadFile) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix}. Allowed: {sorted(AUDIO_EXTS)}",
        )

    # Save uploaded file into existing input/ directory
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = INPUT_DIR / file.filename

    try:
        data = await file.read()
        with open(audio_path, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Transcribe using existing worker function
    try:
        transcript_path = transcribe_file(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Summarize using existing worker function
    try:
        summarize_file(transcript_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    # Read transcript and summary
    try:
        transcript_text = transcript_path.read_text(encoding="utf-8")
    except Exception:
        transcript_text = ""

    summary_path = OUTPUT_DIR / f"{transcript_path.stem}_summary.txt"
    try:
        summary_text = summary_path.read_text(encoding="utf-8")
    except Exception:
        summary_text = ""

    return {
        "filename": file.filename,
        "transcript": transcript_text,
        "summary": summary_text,
        "transcript_path": str(transcript_path),
        "summary_path": str(summary_path),
    }


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    JSON API endpoint – returns transcript and summary as JSON.
    """
    result = await _process_uploaded_file(file)
    return result


@app.get("/ui", response_class=HTMLResponse)
async def ui_form(request: Request):
    """
    HTML UI – initial page with upload form.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None},
    )


@app.post("/ui", response_class=HTMLResponse)
async def ui_submit(request: Request, file: UploadFile = File(...)):
    """
    HTML UI – handle upload and show transcript + summary.
    """
    result = await _process_uploaded_file(file)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result},
    )
