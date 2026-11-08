"""REST server. Contract: POST /faces/count (multipart `file`) -> {"count": int}."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from detector import FaceDetector, InvalidImageError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = FaceDetector()  # load the model once at startup
    yield


app = FastAPI(title="Face count REST server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/faces/count")
def count_faces(file: UploadFile):
    # Plain `def`: FastAPI runs it in a worker thread so inference does not block the event loop.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    try:
        return {"count": app.state.detector.count(data)}
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc))
