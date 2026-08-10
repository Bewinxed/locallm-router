#!/usr/bin/env python3
"""
OpenAI-compatible ASR server for nvidia/parakeet-tdt-0.6b-v3.

The router speaks the OpenAI audio API, and vLLM has no FastConformer-TDT
architecture, so this sidecar wraps the Transformers implementation
(`AutoModelForTDT`, the path NVIDIA documents on the model card) behind the two
endpoints the router needs: /health for readiness and /v1/audio/transcriptions
for the work.

Launched by the `parakeet-asr` backend descriptor in src/process.ts; the ready
line it matches on is printed once the weights are resident.

Needs ffmpeg on PATH and its own venv, which src/process.ts looks for at
~/parakeet-env (override with PARAKEET_PYTHON):

    uv venv ~/parakeet-env --python 3.12
    uv pip install --python ~/parakeet-env/bin/python \
        torch "transformers>=5.14" librosa soundfile numpy \
        fastapi "uvicorn[standard]" python-multipart
"""

import argparse
import asyncio
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from transformers import AutoModelForTDT, AutoProcessor

SAMPLE_RATE = 16000

app = FastAPI()

asr_model = None
asr_processor = None
# One set of weights on one device: overlapping generate() calls would race over
# both. Requests queue here instead.
inference_lock = threading.Lock()


def decode_audio(data: bytes, suffix: str) -> np.ndarray:
    """Any container ffmpeg understands -> mono float32 at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error",
                "-i", tmp.name,
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "pipe:1",
            ],
            capture_output=True,
        )
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {result.stderr.decode(errors='replace').strip()}",
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


def transcribe(audio: np.ndarray) -> str:
    inputs = asr_processor(audio, sampling_rate=SAMPLE_RATE)
    inputs.to(asr_model.device, dtype=asr_model.dtype)
    with inference_lock, torch.inference_mode():
        output = asr_model.generate(**inputs, return_dict_in_generate=True)
    return asr_processor.decode(output.sequences[0], skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    # Both are accepted and ignored, so OpenAI clients need no special case:
    # the router has already routed on `model`, and parakeet-tdt-0.6b-v3
    # detects the language itself and takes no language conditioning.
    model: str = Form("parakeet"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    if response_format not in ("json", "text", "verbose_json"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}' (json, text, verbose_json)",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio = await asyncio.to_thread(decode_audio, data, suffix)
    text = await asyncio.to_thread(transcribe, audio)

    if response_format == "text":
        return PlainTextResponse(text + "\n")
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "duration": round(len(audio) / SAMPLE_RATE, 3),
            "text": text,
        }
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    global asr_model, asr_processor
    asr_processor = AutoProcessor.from_pretrained(args.model)
    asr_model = AutoModelForTDT.from_pretrained(args.model, dtype=dtype).to(device)
    asr_model.eval()

    print(f"parakeet server listening on {args.host}:{args.port} ({device}, {dtype})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
