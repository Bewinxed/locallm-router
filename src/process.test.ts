/**
 * The vLLM command line.
 *
 * An entry that opts out of sleep mode must not also be handed the flag that
 * opts in. argparse takes the last one either way, so the process behaves —
 * but the command line and the log it is printed to contradict each other, and
 * that is the line anyone debugging a sleep problem reads first.
 */

import { describe, expect, spyOn, test } from "bun:test";
import type { ModelConfig } from "./config";
import { createProcess } from "./process";

/** A port no backend is configured for: the real ones start at 40001. */
const SCRATCH_PORT = 45999;

function vllmConfig(extraArgs: string[]): ModelConfig {
  return {
    name: "sleep-flags",
    modelPath: "vendor/Some-Model",
    backend: "vllm",
    gpu: 0,
    idleTimeout: 600_000,
    extraArgs,
  };
}

/**
 * Runs a launch as far as the spawn and no further. The backend never starts:
 * the stub records the argv and throws, which launch() reports as a failed
 * spawn without leaving a process behind.
 */
async function spawnArgs(config: ModelConfig): Promise<string[]> {
  const realSpawn = Bun.spawn;
  let cmd: string[] = [];

  const spy = spyOn(Bun, "spawn").mockImplementation(((options: any) => {
    // claimPort probes the port with `ss` on the way in; that one is real.
    if (options.cmd[0] === "ss") return realSpawn(options);
    cmd = options.cmd;
    throw new Error("stubbed spawn");
  }) as typeof Bun.spawn);

  try {
    await createProcess(config, SCRATCH_PORT)
      .start()
      .catch(() => {});
  } finally {
    spy.mockRestore();
  }
  return cmd;
}

describe("vLLM sleep flags", () => {
  test("an entry that disables sleep mode is not also handed --enable-sleep-mode", async () => {
    const cmd = await spawnArgs(
      vllmConfig(["--no-enable-sleep-mode", "--tensor-parallel-size", "2"])
    );

    expect(cmd).toContain("--no-enable-sleep-mode");
    expect(cmd).not.toContain("--enable-sleep-mode");
  });

  test("every other entry still gets sleep mode", async () => {
    const cmd = await spawnArgs(vllmConfig(["--max-model-len", "8192"]));

    expect(cmd).toContain("--enable-sleep-mode");
  });
});
