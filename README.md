# Face detection: MCP server + REST server

```
MCP client --(image as base64)--> MCP server --(image as multipart upload)--> REST server (FastAPI + MediaPipe)
```

The client sends the image **data** itself; the MCP server never reads the local file system, so the
same setup works on your machine and when deployed elsewhere.

## Project layout

```
models/blaze_face_short_range.tflite   MediaPipe BlazeFace (short-range) model
rest-server/    FastAPI + MediaPipe; POST /faces/count, GET /health
mcp-server/     MCP server exposing the detect_face tool
examples/       detect_face_client.py, a minimal MCP client
```

## Contract

MCP tool `detect_face`
- in:  `{ "image_base64": "<base64 of the image file bytes>" }` (jpg/png/webp/bmp, up to 20 MB; a `data:...;base64,` prefix is tolerated)
- out: `{ "hasface": true, "count": 3 }`
- failure: `is_error: true` with a message (invalid base64, empty, too large, not a valid image, REST server unreachable)

REST server
- `POST /faces/count` (multipart field `file`) -> `{ "count": 3 }`
- `GET /health`

## Run (conda)

Two services, two conda environments (`restserver`, `mcpserver`):

```bash
# 1) REST server on :8000
conda activate restserver
cd rest-server && python -m uvicorn main:app --port 8000

# 2) MCP server on :8001, endpoint http://127.0.0.1:8001/mcp
conda activate mcpserver
cd mcp-server && MCP_TRANSPORT=streamable-http REST_SERVER_URL=http://127.0.0.1:8000 python server.py
```

Try it with the example client:

```bash
python examples/detect_face_client.py /path/to/photo.jpg
```

Or register the HTTP server with Claude Code: `claude mcp add --transport http face-detect http://127.0.0.1:8001/mcp`.
Note a language model has to write the base64 itself, so this is practical for programmatic clients and
small images, not for pasting large photos through a chat.

Without `MCP_TRANSPORT=streamable-http` the server runs over stdio (the MCP client launches it, and
environment variables must be passed by the client, e.g. `claude mcp add -e REST_SERVER_URL=...`).

| Variable | Default | Used by |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | MCP server (`stdio` or `streamable-http`) |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8001` | MCP server (HTTP mode) |
| `REST_SERVER_URL` | `http://127.0.0.1:8000` | MCP server |
| `MODEL_PATH` | `models/blaze_face_short_range.tflite` | REST server |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | REST server |

HTTP mode has no authentication. Keep it on `127.0.0.1`; add auth before binding to a public address.
The MCP server accepts request bodies up to ~28 MB (the SDK default of 4 MB would reject images over ~3 MB).

## Size limits

An image has to pass three limits on its way in. They are sized so that an outer limit never
rejects an image the inner ones would accept.

| # | Where | Limit | Set in | If exceeded |
|---|---|---|---|---|
| 1 | HTTP request body (HTTP mode only) | 29,010,604 bytes (about 27.7 MiB) | `max_request_body_size=MAX_REQUEST_BODY_BYTES` in the `mcp.run(...)` call at the bottom of `mcp-server/server.py` | HTTP 413 `Request body too large`; the tool is never called |
| 2 | Length of the `image_base64` text | 27,962,028 characters (a 20 MB image) | `MAX_BASE64_CHARS` in `mcp-server/server.py`, checked in `decode_image()` before decoding | tool error `image larger than 20 MB` |
| 3 | Image uploaded to the REST server | 20 MB | `MAX_UPLOAD_BYTES` in `rest-server/main.py` | REST answers 413, the tool reports `image larger than 20 MB` |

### Where the numbers come from

```python
MAX_IMAGE_BYTES = 20 * 1024 * 1024                  # 20 MB, the biggest image we accept
MAX_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4  # base64 turns every 3 bytes into 4 characters
MAX_REQUEST_BODY_BYTES = MAX_BASE64_CHARS + 1024 * 1024  # plus 1 MiB for the JSON-RPC wrapper around it
```

### Why the SDK limit is set explicitly

The MCP SDK rejects HTTP request bodies over **4 MB** by default. Base64 makes an image about 33%
bigger, so by default only images up to roughly 3 MB could be sent, and a 6 MB photo would get
HTTP 413 before reaching the tool. `server.py` therefore raises the SDK limit when it starts in
HTTP mode:

```python
mcp.run(
    transport="streamable-http",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8001")),
    max_request_body_size=MAX_REQUEST_BODY_BYTES,  # raise the SDK's 4 MB default
)
```

Limit 1 is an HTTP limit, so it does not apply in stdio mode. Limits 2 and 3 apply in both modes.

### Example

A 6 MB photo becomes about 8 MB of base64 text. That is under limit 1 (27.7 MiB) and limit 2
(about 26.7 MiB), and the decoded image is under limit 3 (20 MB), so it is accepted.

### Changing the maximum image size

Set `MAX_IMAGE_BYTES` in `mcp-server/server.py` and `MAX_UPLOAD_BYTES` in `rest-server/main.py` to
the same value. Limits 1 and 2 are calculated from `MAX_IMAGE_BYTES`, so they follow automatically.

## Tests

```bash
cd mcp-server  && python -m pytest   # env: mcpserver
cd rest-server && python -m pytest   # env: restserver
```
