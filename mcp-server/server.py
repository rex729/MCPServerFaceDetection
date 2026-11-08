"""MCP server: exposes `detect_face` to MCP clients and calls the REST server directly.

The client sends the image itself (base64) in the tool arguments, so the server never touches
the local file system and works the same locally and when deployed.

REST contract: POST /faces/count (multipart `file`) -> {"count": int}
"""
import base64
import binascii
import os
from typing import TypedDict

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

REST_SERVER_URL = os.environ.get("REST_SERVER_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 30.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # decoded image, same limit as the REST server
MAX_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
# The SDK's default HTTP body limit is 4 MB, which would reject any image over ~3 MB.
MAX_REQUEST_BODY_BYTES = MAX_BASE64_CHARS + 1024 * 1024

_transport: httpx.AsyncBaseTransport | None = None  # tests inject a fake REST server here
_client: httpx.AsyncClient | None = None

mcp = MCPServer("face-detect")


def _get_client() -> httpx.AsyncClient:
    """One shared client so connections to the REST server are reused across calls."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=REST_SERVER_URL, timeout=TIMEOUT_SECONDS, transport=_transport
        )
    return _client


def decode_image(image_base64: str) -> bytes:
    data = image_base64.strip()
    if data.startswith("data:"):  # tolerate "data:image/png;base64,...." URLs
        data = data.partition(",")[2]
    if len(data) > MAX_BASE64_CHARS:  # check before decoding so huge input is not expanded
        raise ToolError(f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        raw = base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ToolError("image_base64 is not valid base64") from exc
    if not raw:
        raise ToolError("image is empty")
    return raw


class FaceResult(TypedDict):
    hasface: bool
    count: int


@mcp.tool()
async def detect_face(image_base64: str) -> FaceResult:
    """Detect whether an image contains any human face.

    Args:
        image_base64: The image file's bytes (jpg, png, webp or bmp, up to 20 MB), base64-encoded.
    """
    # ToolError text is shown to the model as-is; any other exception becomes a generic message.
    raw = decode_image(image_base64)

    try:
        resp = await _get_client().post("/faces/count", files={"file": ("image", raw)})
    except httpx.HTTPError as exc:
        raise ToolError(f"REST server unreachable: {type(exc).__name__}") from exc
    if resp.status_code == 400:
        raise ToolError("not a valid image")
    if resp.status_code == 413:
        raise ToolError(f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    if resp.status_code != 200:
        raise ToolError(f"REST server returned HTTP {resp.status_code}")
    try:
        count = int(resp.json()["count"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ToolError("REST server returned an unexpected response") from exc

    return {"hasface": count > 0, "count": count}


if __name__ == "__main__":
    # stdio (default): the MCP client launches this process itself.
    # streamable-http: run it yourself as a long-lived service at http://HOST:PORT/mcp
    if os.environ.get("MCP_TRANSPORT", "stdio") == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8001")),
            max_request_body_size=MAX_REQUEST_BODY_BYTES,
        )
    else:
        mcp.run(transport="stdio")
