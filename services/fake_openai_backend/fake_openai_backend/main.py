from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from fake_openai_backend.settings import Settings

settings = Settings()
app = FastAPI(title="fake OpenAI-compatible backend", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "fake-openai-backend", "status": "healthy"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "fake",
            }
            for model_id in settings.models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    failure = fake_failure(request)
    if failure is not None:
        return failure

    await fake_delay(request)
    payload = await request.json()
    model = str(payload.get("model") or settings.models[0])
    text = response_text(request, payload)

    if bool(payload.get("stream")):
        return StreamingResponse(
            chat_completion_stream(model, text),
            media_type="text/event-stream",
        )

    return JSONResponse(chat_completion_response(model, text, payload))


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    failure = fake_failure(request)
    if failure is not None:
        return failure

    await fake_delay(request)
    payload = await request.json()
    model = str(payload.get("model") or settings.models[0])
    text = response_text(request, payload)
    return JSONResponse(
        {
            "id": "resp_fake",
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": usage(payload, text),
        }
    )


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    failure = fake_failure(request)
    if failure is not None:
        return failure

    await fake_delay(request)
    payload = await request.json()
    model = str(payload.get("model") or settings.models[0])
    text = response_text(request, payload)
    return JSONResponse(
        {
            "id": "cmpl_fake",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": usage(payload, text),
        }
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    failure = fake_failure(request)
    if failure is not None:
        return failure

    await fake_delay(request)
    payload = await request.json()
    model = str(payload.get("model") or settings.models[0])
    return JSONResponse(
        {
            "object": "list",
            "model": model,
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "usage": {"prompt_tokens": estimated_tokens(payload), "total_tokens": 3},
        }
    )


def fake_failure(request: Request) -> JSONResponse | None:
    raw_status = request.headers.get("x-fake-status-code")
    if raw_status is None:
        return None

    status_code = int(raw_status)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "fake_backend_error",
                "message": f"Forced fake backend status {status_code}.",
            }
        },
    )


async def fake_delay(request: Request) -> None:
    raw_delay = request.headers.get("x-fake-delay-ms")
    if raw_delay is None:
        return
    await asyncio.sleep(max(0, int(raw_delay)) / 1000)


def response_text(request: Request, payload: dict[str, Any]) -> str:
    header_text = request.headers.get("x-fake-response-text")
    if header_text:
        return header_text
    if payload.get("fake_response_text"):
        return str(payload["fake_response_text"])
    return settings.response_text


async def chat_completion_stream(model: str, text: str) -> AsyncIterator[bytes]:
    created = int(time.time())
    for index, chunk in enumerate(split_text(text)):
        payload = {
            "id": "chatcmpl_fake",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        if index == 0:
            payload["choices"][0]["delta"]["role"] = "assistant"
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
        await asyncio.sleep(0.01)

    done_payload = {
        "id": "chatcmpl_fake",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload, separators=(',', ':'))}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def chat_completion_response(
    model: str,
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": "chatcmpl_fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage(payload, text),
    }


def usage(payload: dict[str, Any], output_text: str) -> dict[str, int]:
    prompt_tokens = estimated_tokens(payload)
    completion_tokens = max(1, round(len(output_text) / 4))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimated_tokens(value: Any) -> int:
    text = collect_text(value)
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def collect_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(collect_text(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("messages", "input", "prompt", "content", "text"):
            if key in value:
                parts.append(collect_text(value[key]))
        return "\n".join(parts)
    return ""


def split_text(text: str) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + 2] for index in range(0, len(text), 2)]
