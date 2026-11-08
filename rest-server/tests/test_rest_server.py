import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads the model once
        yield c


def post(client, data: bytes, name="x.jpg"):
    return client.post("/faces/count", files={"file": (name, data)})


def png_bytes(color=(128, 128, 128), size=(300, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_one_face(client):
    r = post(client, (FIXTURES / "one_face.jpg").read_bytes())
    assert r.status_code == 200
    assert r.json() == {"count": 1}


def test_no_face(client):
    r = post(client, png_bytes(), "blank.png")
    assert r.status_code == 200
    assert r.json() == {"count": 0}


def upside_down_jpeg(exif_orientation=None) -> bytes:
    img = Image.open(FIXTURES / "one_face.jpg").convert("RGB").rotate(180)
    exif = Image.Exif()
    if exif_orientation:
        exif[0x0112] = exif_orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_upside_down_without_exif_is_not_detected(client):
    # Control: proves the next test really depends on EXIF handling.
    assert post(client, upside_down_jpeg()).json() == {"count": 0}


def test_exif_orientation_is_applied(client):
    # Pixels are stored upside down; EXIF tag 3 (rotate 180) says how to display them,
    # the way phones do. Without exif_transpose the face would not be found.
    assert post(client, upside_down_jpeg(exif_orientation=3)).json() == {"count": 1}


def test_not_an_image_is_400(client):
    r = post(client, b"this is not an image", "x.jpg")
    assert r.status_code == 400


def test_oversized_upload_is_413(client, monkeypatch):
    monkeypatch.setattr("main.MAX_UPLOAD_BYTES", 10)
    r = post(client, b"x" * 100)
    assert r.status_code == 413


def test_missing_file_field_is_422(client):
    assert client.post("/faces/count").status_code == 422
