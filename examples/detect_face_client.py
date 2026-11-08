"""Minimal MCP client: reads an image file, sends it as base64 to `detect_face`.

Usage (HTTP server already running):  python detect_face_client.py photo.jpg
Other server:                         python detect_face_client.py photo.jpg --url http://host:8001/mcp
"""
import argparse
import asyncio
import base64
from pathlib import Path

from mcp import Client


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001/mcp")
    args = parser.parse_args()

    image_base64 = base64.b64encode(args.image.read_bytes()).decode()
    async with Client(args.url) as client:
        result = await client.call_tool("detect_face", {"image_base64": image_base64})
    if result.is_error:
        raise SystemExit(f"error: {result.content[0].text}")
    print(result.structured_content)


asyncio.run(main())
