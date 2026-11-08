"""Face detection with MediaPipe BlazeFace. Independent of HTTP so it can be tested alone."""
import io
import os
import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))
MIN_CONFIDENCE = float(os.environ.get("MIN_DETECTION_CONFIDENCE", "0.5"))


class InvalidImageError(ValueError):
    """The uploaded bytes are not a decodable image."""


class FaceDetector:
    def __init__(self, model_path: Path = MODEL_PATH, min_confidence: float = MIN_CONFIDENCE):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"face model not found: {model_path}")
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._lock = threading.Lock()  # one detector instance is not safe for concurrent use

    def count(self, image_bytes: bytes) -> int:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img).convert("RGB")  # phones store rotation in EXIF
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("not a valid image") from exc
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.array(img)))
        with self._lock:
            return len(self._detector.detect(frame).detections)
