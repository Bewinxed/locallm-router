/**
 * The estimation ladder: which rung supplies the number, and in what order.
 *
 * The case that matters is a model whose GGUF header parses perfectly and whose
 * config overrides it anyway. vramGB is a human correcting an estimate that
 * came out wrong, so no estimate is allowed to win against it.
 */

import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { ModelConfig } from "./config";
import { resolveEstimate } from "./model-size";

const GGUF_UINT32 = 4;
const GGUF_STRING = 8;
/** ggml type id for F16, the dtype of the one tensor these headers declare. */
const GGML_F16 = 1;

function u32(value: number): Buffer {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(value);
  return buf;
}

function u64(value: number): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64LE(BigInt(value));
  return buf;
}

function gstring(value: string): Buffer {
  return Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);
}

function kvU32(key: string, value: number): Buffer {
  return Buffer.concat([gstring(key), u32(GGUF_UINT32), u32(value)]);
}

/**
 * A GGUF header and nothing else. The tensor data it points at is never
 * written, because the parser only ever reads the header — which is the whole
 * reason the router can size a model it hasn't downloaded.
 */
function writeTinyGguf(path: string): void {
  const metadata = [
    Buffer.concat([gstring("general.architecture"), u32(GGUF_STRING), gstring("llama")]),
    kvU32("llama.block_count", 32),
    kvU32("llama.embedding_length", 4096),
    kvU32("llama.attention.head_count", 32),
    kvU32("llama.attention.head_count_kv", 8),
  ];

  const tensor = Buffer.concat([
    gstring("token_embd.weight"),
    u32(2),
    u64(4096),
    u64(1000),
    u32(GGML_F16),
    u64(0),
  ]);

  writeFileSync(
    path,
    Buffer.concat([
      Buffer.from("GGUF"),
      u32(3),
      u64(1),
      u64(metadata.length),
      ...metadata,
      tensor,
    ])
  );
}

function ggufConfig(modelPath: string): ModelConfig {
  return {
    name: "tiny-gguf",
    modelPath,
    backend: "llama-server",
    gpu: 0,
    idleTimeout: 600_000,
  };
}

describe("resolveEstimate", () => {
  const dir = mkdtempSync(join(tmpdir(), "locallm-model-size-"));
  const modelPath = join(dir, "tiny.gguf");
  writeTinyGguf(modelPath);

  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("reads the local GGUF header when nothing overrides it", async () => {
    const estimate = await resolveEstimate(ggufConfig(modelPath));

    expect(estimate.confidence).toBe("gguf-local");
    expect(estimate.totalVRAM).toBeGreaterThan(0);
  });

  test("vramGB outranks a header that parsed perfectly well", async () => {
    const parsed = await resolveEstimate(ggufConfig(modelPath));
    const overridden = await resolveEstimate({ ...ggufConfig(modelPath), vramGB: 19 });

    expect(overridden.totalVRAM).toBe(19);
    expect(overridden.confidence).toBe("manual");
    // Without this the test would pass on an override that happened to agree.
    expect(parsed.totalVRAM).not.toBe(19);
  });

  test("falls back to the heuristic when there is no header and no override", async () => {
    const estimate = await resolveEstimate({
      ...ggufConfig(join(dir, "absent.gguf")),
      parameterCount: 27,
    });

    expect(estimate.confidence).toBe("exact");
    expect(estimate.parametersBillions).toBe(27);
  });
});
