"""OpenAI-compatible /v1/chat/completions shim around local Northstar.

Loads the model once via cua_shared.load_northstar() and serves Qwen-style
tool-call responses for chat-completions requests that include a single image.

Run:
    python3 scripts/northstar_openai_shim.py --port 8000

Test:
    curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \\
      -d '{"model":"northstar-cua-fast","messages":[{"role":"user","content":[
            {"type":"text","text":"sign in"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}'

The response shape mirrors OpenAI chat completions: choices[0].message.content
contains the raw model text (which itself contains a <tool_call>...</tool_call>
block). The Lightcone harness or any OpenAI client can consume this.
"""

from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from cua_shared import (
    QWEN_TOOLS,
    SYS_PROMPT,
    LoadedModel,
    load_northstar,
)

LM: LoadedModel | None = None
SERVED_MODEL_NAME = "northstar-cua-fast"


# --- request/response shapes (loose; we tolerate extra fields) -----------


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = 256
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    tools: list[dict[str, Any]] | None = None
    # Allow other fields silently
    class Config:
        extra = "allow"


# --- helpers --------------------------------------------------------------


def _decode_data_url(data_url: str) -> Image.Image:
    if data_url.startswith("data:"):
        _, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
    else:
        # Assume bare base64
        raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_text_and_image(messages: list[dict[str, Any]]) -> tuple[str, Image.Image | None]:
    """Pull the latest user task text and the latest image from the messages."""
    task_parts: list[str] = []
    image: Image.Image | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            continue  # we use our own system prompt
        if isinstance(content, str):
            if role == "user":
                task_parts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    if role == "user":
                        task_parts.append(part.get("text", ""))
                elif ptype in ("image_url", "input_image"):
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url", "")
                    else:
                        url = iu or part.get("url") or ""
                    if url:
                        try:
                            image = _decode_data_url(url)
                        except Exception as exc:
                            raise HTTPException(400, f"bad image_url: {exc}")
    return ("\n".join(p for p in task_parts if p).strip() or "Continue the task."), image


@torch.no_grad()
def _run_inference(task: str, image: Image.Image, max_new_tokens: int = 256) -> str:
    assert LM is not None
    # Use the project's standard prompt + tools for consistency with eval.py.
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"User task: {task}\nLook at the screenshot and emit ONE tool call for the next action."},
            ],
        },
    ]
    prompt_text = LM.processor.apply_chat_template(
        messages,
        tools=QWEN_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = LM.processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(LM.model.device)
    gen = LM.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    text = LM.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


# --- app ------------------------------------------------------------------


app = FastAPI()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": LM is not None}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": SERVED_MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if LM is None:
        raise HTTPException(503, "model not loaded yet")
    task, image = _extract_text_and_image(req.messages)
    if image is None:
        raise HTTPException(400, "expected at least one image_url part in messages")
    raw_text = _run_inference(task, image, max_new_tokens=req.max_tokens or 256)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": raw_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --- entrypoint -----------------------------------------------------------


def main() -> None:
    global LM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    print(f"[shim] loading Northstar (adapter={args.adapter})", flush=True)
    LM = load_northstar(adapter_path=args.adapter)
    print(f"[shim] ready, serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
