/**
 * Model Download Manager
 *
 * Downloads HuggingFace models to the shared volume (/models)
 * before backend processes start. Uses huggingface-cli for
 * reliable, resumable downloads with proper caching.
 */

import { spawn } from "child_process";

const MODELS_DIR = process.env.HF_HOME || "/models";

/** Check if a model path is a local file (GGUF, etc.) vs a HuggingFace repo */
export function isLocalModel(modelPath: string): boolean {
  return modelPath.startsWith("/") || modelPath.startsWith("./");
}

/** Check if an HF model is already downloaded (has a snapshot) */
export async function isModelCached(modelPath: string): Promise<boolean> {
  if (isLocalModel(modelPath)) {
    // For local files, check if file exists
    const file = Bun.file(modelPath);
    return file.exists();
  }

  const repoDir = modelPath.replace("/", "--");
  const snapshotsDir = `${MODELS_DIR}/hub/models--${repoDir}/snapshots`;

  try {
    for await (const _entry of new Bun.Glob("*").scan({ cwd: snapshotsDir, onlyFiles: false })) {
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/** Download an HF model using huggingface-cli. Returns when download completes. */
export async function downloadModel(
  modelPath: string,
  onProgress?: (line: string) => void
): Promise<void> {
  if (isLocalModel(modelPath)) {
    const exists = await isModelCached(modelPath);
    if (!exists) {
      throw new Error(`Local model file not found: ${modelPath}`);
    }
    return; // Already exists
  }

  const cached = await isModelCached(modelPath);
  if (cached) {
    onProgress?.(`Model ${modelPath} already cached`);
    return;
  }

  onProgress?.(`Downloading ${modelPath} to ${MODELS_DIR}...`);

  return new Promise((resolve, reject) => {
    const proc = spawn(
      "huggingface-cli",
      ["download", modelPath, "--cache-dir", `${MODELS_DIR}/hub`],
      {
        env: {
          ...process.env,
          HF_HOME: MODELS_DIR,
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    proc.stdout?.on("data", (data) => {
      const text = data.toString().trim();
      if (text) onProgress?.(text);
    });

    proc.stderr?.on("data", (data) => {
      const text = data.toString().trim();
      if (text) onProgress?.(text);
    });

    proc.on("error", (err) => {
      reject(new Error(`Failed to start huggingface-cli: ${err.message}`));
    });

    proc.on("exit", (code) => {
      if (code === 0) {
        onProgress?.(`Download complete: ${modelPath}`);
        resolve();
      } else {
        reject(new Error(`huggingface-cli exited with code ${code}`));
      }
    });
  });
}
