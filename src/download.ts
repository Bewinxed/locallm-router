/**
 * Model Download Manager
 *
 * Downloads HuggingFace models to the shared volume (/models)
 * before backend processes start. Uses huggingface-cli for
 * reliable, resumable downloads with proper caching.
 */

import { spawn } from "child_process";

export const MODELS_DIR = process.env.HF_HOME || "/models";

/** Check if a model path is a local file (GGUF, etc.) vs a HuggingFace repo */
export function isLocalModel(modelPath: string): boolean {
  return modelPath.startsWith("/") || modelPath.startsWith("./");
}

/**
 * Local path a single-file download (config.modelFile) resolves to.
 * Kept outside the HF hub cache layout so the path is deterministic and
 * llama-server can be pointed at it directly.
 */
export function singleFileLocalPath(modelPath: string, modelFile: string): string {
  return `${MODELS_DIR}/gguf/${modelPath.replace("/", "--")}/${modelFile}`;
}

/**
 * Direct URL to a file in an HF repo. Supports range requests, so a header can
 * be read from it without pulling the file down.
 */
export function hfFileUrl(modelPath: string, modelFile: string): string {
  return `https://huggingface.co/${modelPath}/resolve/main/${modelFile}`;
}

/**
 * The path a backend should actually load: the local file itself, the
 * single-file download location, or the HF repo ID (vLLM/SGLang resolve
 * repos natively).
 */
export function resolveModelPath(config: { modelPath: string; modelFile?: string }): string {
  if (isLocalModel(config.modelPath)) return config.modelPath;
  if (config.modelFile) return singleFileLocalPath(config.modelPath, config.modelFile);
  return config.modelPath;
}

/** The CLI downloader: prefer the modern `hf`, fall back to `huggingface-cli`. */
function downloaderBin(): string {
  return Bun.which("hf") ? "hf" : "huggingface-cli";
}

/** Check if an HF model is already downloaded (has a snapshot) */
export async function isModelCached(modelPath: string, modelFile?: string): Promise<boolean> {
  if (isLocalModel(modelPath)) {
    // For local files, check if file exists
    const file = Bun.file(modelPath);
    return file.exists();
  }

  if (modelFile) {
    return Bun.file(singleFileLocalPath(modelPath, modelFile)).exists();
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

/** Download an HF model using the HF CLI. Returns when download completes. */
export async function downloadModel(
  modelPath: string,
  onProgress?: (line: string) => void,
  modelFile?: string
): Promise<void> {
  if (isLocalModel(modelPath)) {
    const exists = await isModelCached(modelPath);
    if (!exists) {
      throw new Error(`Local model file not found: ${modelPath}`);
    }
    return; // Already exists
  }

  const cached = await isModelCached(modelPath, modelFile);
  if (cached) {
    onProgress?.(`Model ${modelPath} already cached`);
    return;
  }

  onProgress?.(`Downloading ${modelPath}${modelFile ? `/${modelFile}` : ""} to ${MODELS_DIR}...`);

  // Single-file downloads (GGUF quants) land in a deterministic local dir;
  // whole-repo downloads use the standard HF hub cache.
  const args = modelFile
    ? [
        "download", modelPath, modelFile,
        "--local-dir", `${MODELS_DIR}/gguf/${modelPath.replace("/", "--")}`,
      ]
    : ["download", modelPath, "--cache-dir", `${MODELS_DIR}/hub`];

  return new Promise((resolve, reject) => {
    const proc = spawn(
      downloaderBin(),
      args,
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
