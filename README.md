# 人臉偵測:MCP server + REST server

```
MCP client --(圖片以 base64 傳送)--> MCP server --(圖片以 multipart 上傳)--> REST server (FastAPI + MediaPipe)
```

Client 直接傳送圖片**資料**;MCP server 不會讀取本機檔案系統,所以同一套做法在你的電腦上和部署到其他地方都能用。

## 專案結構

```
models/blaze_face_short_range.tflite   MediaPipe BlazeFace(short-range)模型
rest-server/    FastAPI + MediaPipe;POST /faces/count、GET /health
mcp-server/     提供 detect_face 工具的 MCP server
examples/       detect_face_client.py,最精簡的 MCP client 範例
```

## 介面約定

MCP 工具 `detect_face`
- 輸入:`{ "image_base64": "<圖片檔案位元組的 base64>" }`(jpg/png/webp/bmp,最大 20 MiB;可帶 `data:...;base64,` 前綴)
- 輸出:`{ "hasface": true, "count": 3 }`
- 失敗:`is_error: true` 並附上訊息(base64 無效、空圖片、太大、不是有效圖片、連不上 REST server)

REST server
- `POST /faces/count`(multipart 欄位 `file`)-> `{ "count": 3 }`
- `GET /health`

## 執行方式(conda)

兩個服務、兩個 conda 環境(`restserver`、`mcpserver`):

```bash
# 1) REST server,埠 8000
conda activate restserver
cd rest-server && python -m uvicorn main:app --port 8000

# 2) MCP server,埠 8001,端點 http://127.0.0.1:8001/mcp
conda activate mcpserver
cd mcp-server && MCP_TRANSPORT=streamable-http REST_SERVER_URL=http://127.0.0.1:8000 python server.py
```

用範例 client 試試看:

```bash
python examples/detect_face_client.py /path/to/photo.jpg
```

也可以把這個 HTTP server 註冊到 Claude Code:`claude mcp add --transport http face-detect http://127.0.0.1:8001/mcp`。
注意:base64 必須由語言模型自己寫出來,所以這個做法適合程式化的 client 和小圖片,不適合在聊天中貼大張照片。

沒有設定 `MCP_TRANSPORT=streamable-http` 時,server 會以 stdio 執行(由 MCP client 負責啟動,環境變數也必須由 client 傳入,例如 `claude mcp add -e REST_SERVER_URL=...`)。

| 變數 | 預設值 | 使用者 |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | MCP server(`stdio` 或 `streamable-http`) |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8001` | MCP server(HTTP 模式) |
| `REST_SERVER_URL` | `http://127.0.0.1:8000` | MCP server |
| `MODEL_PATH` | `models/blaze_face_short_range.tflite` | REST server |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | REST server |

HTTP 模式沒有認證。請讓它只綁在 `127.0.0.1`;要綁到公開位址之前,先加上認證。
MCP server 接受最大 27.7 MiB 的請求內容(SDK 預設的 4 MiB 會擋掉大於約 3 MiB 的圖片)。詳見[大小限制](#大小限制)。

## 大小限制

圖片進來時要通過三道限制。它們的大小是這樣安排的:外層的限制不會擋掉內層本來能接受的圖片。

| # | 位置 | 限制 | 設定在 | 超過時 |
|---|---|---|---|---|
| 1 | HTTP 請求內容(僅 HTTP 模式) | `MAX_REQUEST_BODY_BYTES` = 29,010,604 位元組(27.7 MiB) | `mcp-server/server.py` 最下方 `mcp.run(...)` 裡的 `max_request_body_size=MAX_REQUEST_BODY_BYTES` | HTTP 413 `Request body too large`;工具不會被呼叫 |
| 2 | `image_base64` 文字的長度 | `MAX_BASE64_CHARS` = 27,962,028 字元(20 MiB 圖片的 base64 文字) | `mcp-server/server.py` 的 `MAX_BASE64_CHARS`,在 `decode_image()` 解碼前檢查 | 工具錯誤 `image larger than 20 MB` |
| 3 | 上傳到 REST server 的圖片 | `MAX_UPLOAD_BYTES` = 20 MiB | `rest-server/main.py` 的 `MAX_UPLOAD_BYTES` | REST 回 413,工具回報 `image larger than 20 MB` |

本節的大小單位都是 MiB(1 MiB = 1,048,576 位元組)。工具的錯誤文字仍然寫「20 MB」,因為那是程式碼實際回傳的原文。

### 這些數字怎麼來的

```python
MAX_IMAGE_BYTES = 20 * 1024 * 1024                  # 20 MiB, the biggest image we accept
MAX_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4  # base64 turns every 3 bytes into 4 characters
MAX_REQUEST_BODY_BYTES = MAX_BASE64_CHARS + 1024 * 1024  # plus 1 MiB for the JSON-RPC wrapper around it
```

- `MAX_IMAGE_BYTES`:我們接受的最大圖片,20 MiB。
- `MAX_BASE64_CHARS`:base64 每 3 個位元組變成 4 個字元,所以 20 MiB 的圖片轉成文字後是 27,962,028 個字元。
- `MAX_REQUEST_BODY_BYTES`:再加 1 MiB,留給包在外面的 JSON-RPC 訊息。

### 為什麼要明確設定 SDK 的上限

MCP SDK 預設會拒絕超過 **4 MiB** 的 HTTP 請求內容。Base64 會讓圖片變大約 33%,所以預設只能傳大約 3 MiB 以內的圖片,6 MiB 的照片還沒到達工具就會收到 HTTP 413。因此 `server.py` 在 HTTP 模式啟動時調高了 SDK 的上限:

```python
mcp.run(
    transport="streamable-http",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8001")),
    max_request_body_size=MAX_REQUEST_BODY_BYTES,  # raise the SDK's 4 MiB default
)
```

第 1 道限制是 HTTP 的限制,所以 stdio 模式不適用。第 2、3 道限制在兩種模式都適用。

### 範例

一張 6 MiB 的照片轉成 base64 後約 8 MiB 的文字。它低於第 1 道限制(27.7 MiB)和第 2 道限制(約 26.7 MiB),解碼後的圖片也低於第 3 道限制(20 MiB),所以會被接受。

### 修改最大圖片大小

把 `mcp-server/server.py` 的 `MAX_IMAGE_BYTES` 和 `rest-server/main.py` 的 `MAX_UPLOAD_BYTES` 改成相同的值。第 1、2 道限制是由 `MAX_IMAGE_BYTES` 算出來的,會自動跟著變。

## 測試

```bash
cd mcp-server  && python -m pytest   # 環境:mcpserver
cd rest-server && python -m pytest   # 環境:restserver
```
