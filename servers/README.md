# Sidecar servers

Backends that the router launches directly, for models no supported inference
engine can serve. Each one is a single Python file plus a venv of its own —
`~/vllm-env` is not to be perturbed to accommodate them.

## parakeet_server.py

An OpenAI-compatible ASR server for `nvidia/parakeet-tdt-0.6b-v3`, driven by
the `parakeet-asr` backend descriptor in `src/process.ts`.

### Why it exists

vLLM has no registered architecture for FastConformer-TDT — there is no
`ParakeetForTDT` — so the model cannot be served through the engine the rest of
the fleet uses. Support is tracked upstream in vllm-project/vllm issue #23943
and PR #41708; both are open. **If that PR lands, this sidecar and its backend
descriptor are deletable**: the entry becomes an ordinary `vllm` model.

Until then the sidecar wraps the Transformers implementation
(`AutoModelForTDT`, the path NVIDIA documents on the model card) behind the two
endpoints the router needs.

### Setup

The venv is separate from `~/vllm-env` because nothing else in the fleet needs
Transformers. `src/process.ts` looks for it at `~/parakeet-env`; override with
`PARAKEET_PYTHON`.

```sh
uv venv ~/parakeet-env --python 3.12
uv pip install --python ~/parakeet-env/bin/python \
    torch "transformers>=5.14" librosa soundfile numpy \
    fastapi "uvicorn[standard]" python-multipart
```

`ffmpeg` must also be on PATH: uploads are decoded to mono float32 at 16 kHz by
shelling out to it, so any container it understands is accepted.

### config.json entry

```json
{
  "name": "parakeet",
  "backend": "parakeet-asr",
  "modelPath": "nvidia/parakeet-tdt-0.6b-v3",
  "parameterCount": 0.6,
  "vramGB": 0,
  "gpu": 1,
  "idleTimeout": 300000,
  "aliases": ["parakeet-tdt-0.6b-v3"],
  "extraArgs": ["--device", "cpu"]
}
```

- `--device cpu` keeps 0.6B of weights off cards that are wanted for the large
  models. `auto` and `cuda` work too; on CUDA the model runs in bfloat16.
- `vramGB: 0` is what makes a CPU entry honest to the scheduler. Without it the
  card would be charged for VRAM this process never allocates, and the
  scheduler would evict a real tenant to make room for nothing.
- `gpu` is mandatory in every model entry, so this one names a card. With
  `--device cpu` the value is meaningless — it only ends up in
  `CUDA_VISIBLE_DEVICES` for a process that never opens a CUDA context.

### Endpoints

- `GET /health` — `{"status": "ok"}` once the weights are resident. The router
  polls this, and matches the `parakeet server listening` line on stdout as the
  readiness signal.
- `POST /v1/audio/transcriptions` — multipart, `file` is required.
  `response_format` accepts `json` (default, `{"text": ...}`), `text` (the
  transcript as `text/plain`), and `verbose_json` (adds `task` and `duration`
  in seconds).
  `model` and `language` are accepted and ignored: the router has already
  routed on `model`, and parakeet-tdt-0.6b-v3 detects language itself and takes
  no language conditioning.

Requests are serialized: one set of weights on one device, so overlapping
`generate()` calls would race over both.
