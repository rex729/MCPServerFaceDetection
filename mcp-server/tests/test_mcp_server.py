import base64

import httpx
import pytest
from mcp import Client

import server
from server import mcp

IMAGE = b"\x89PNG-fake-image-bytes"
IMAGE_B64 = base64.b64encode(IMAGE).decode()


@pytest.fixture(autouse=True)
def fresh_client(monkeypatch):
    # Each test gets its own shared client (and its own fake REST transport).
    monkeypatch.setattr(server, "_client", None)


def fake_rest(monkeypatch, handler):
    """Replace the REST server with `handler(request) -> httpx.Response`."""
    monkeypatch.setattr(server, "_transport", httpx.MockTransport(handler))


def rest_returns(monkeypatch, count):
    fake_rest(monkeypatch, lambda request: httpx.Response(200, json={"count": count}))


async def call(image_base64):
    async with Client(mcp) as client:
        return await client.call_tool("detect_face", {"image_base64": image_base64})


async def test_lists_tool_with_base64_parameter():
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools
    assert [t.name for t in tools] == ["detect_face"]
    assert list(tools[0].input_schema["properties"]) == ["image_base64"]


async def test_has_face(monkeypatch):
    rest_returns(monkeypatch, 3)
    result = await call(IMAGE_B64)
    assert not result.is_error
    assert result.structured_content == {"hasface": True, "count": 3}


async def test_no_face(monkeypatch):
    rest_returns(monkeypatch, 0)
    result = await call(IMAGE_B64)
    assert result.structured_content == {"hasface": False, "count": 0}


async def test_sends_decoded_bytes_as_multipart_file(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["multipart"] = b'name="file"' in request.content
        seen["has_image_bytes"] = IMAGE in request.content
        return httpx.Response(200, json={"count": 2})

    fake_rest(monkeypatch, handler)
    result = await call(IMAGE_B64)
    assert result.structured_content == {"hasface": True, "count": 2}
    assert seen == {
        "method": "POST", "path": "/faces/count", "multipart": True, "has_image_bytes": True,
    }


async def test_accepts_data_url_and_line_wrapped_base64(monkeypatch):
    rest_returns(monkeypatch, 1)
    wrapped = "\n".join(IMAGE_B64[i:i + 8] for i in range(0, len(IMAGE_B64), 8))
    assert (await call("data:image/png;base64," + wrapped)).structured_content["count"] == 1


@pytest.mark.parametrize("bad", ["not base64 !!!", "abc"])
async def test_invalid_base64_is_rejected(monkeypatch, bad):
    rest_returns(monkeypatch, 1)
    result = await call(bad)
    assert result.is_error
    assert "not valid base64" in result.content[0].text


async def test_empty_image_is_rejected(monkeypatch):
    rest_returns(monkeypatch, 1)
    result = await call("")
    assert result.is_error
    assert "empty" in result.content[0].text


async def test_oversized_input_is_rejected_before_calling_rest(monkeypatch):
    called = []
    fake_rest(monkeypatch, lambda r: called.append(r) or httpx.Response(200, json={"count": 1}))
    monkeypatch.setattr(server, "MAX_BASE64_CHARS", 8)
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "larger than" in result.content[0].text
    assert called == []


async def test_rest_says_not_an_image(monkeypatch):
    fake_rest(monkeypatch, lambda request: httpx.Response(400, json={"detail": "not a valid image"}))
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "not a valid image" in result.content[0].text


async def test_rest_says_too_large(monkeypatch):
    fake_rest(monkeypatch, lambda request: httpx.Response(413))
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "larger than" in result.content[0].text


async def test_rest_server_unreachable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    fake_rest(monkeypatch, handler)
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "unreachable" in result.content[0].text


async def test_rest_server_error_status(monkeypatch):
    fake_rest(monkeypatch, lambda request: httpx.Response(500))
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "HTTP 500" in result.content[0].text


@pytest.mark.parametrize("bad", [{"nope": 1}, {"count": "abc"}, {"count": None}])
async def test_rest_server_unexpected_body(monkeypatch, bad):
    fake_rest(monkeypatch, lambda request: httpx.Response(200, json=bad))
    result = await call(IMAGE_B64)
    assert result.is_error
    assert "unexpected response" in result.content[0].text


async def test_http_client_is_shared_across_calls(monkeypatch):
    rest_returns(monkeypatch, 1)
    await call(IMAGE_B64)
    first = server._client
    await call(IMAGE_B64)
    assert first is not None and server._client is first
