export interface ModelConfig {
	name: string;
	modelPath: string;
	/** Single file to fetch from a GGUF repo (llama-server only) */
	modelFile?: string;
	backend?: 'vllm' | 'llama-server' | 'sglang';
	gpu: number;
	gpus?: number[];
	idleTimeout: number;
	extraArgs?: string[];
	dtype?: string;
	parameterCount?: number;
	activatedParameterCount?: number;
	vramGB?: number;
	priority?: number;
	sleepDelay?: number;
	aliases?: string[];
	balanceGpus?: number[];
	balanceGpuArgs?: Record<string, string[]>;
	startupTimeout?: number;
}

export interface ManagerConfig {
	models: ModelConfig[];
	port: number;
	managementPort: number;
	healthCheckInterval: number;
	startupTimeout: number;
}

export const BACKENDS = [
	{ value: 'vllm', label: 'vLLM' },
	{ value: 'llama-server', label: 'llama-server' },
	{ value: 'sglang', label: 'SGLang' }
] as const;

export function newModel(): ModelConfig {
	return {
		name: '',
		modelPath: '',
		gpu: 0,
		idleTimeout: 5 * 60_000
	};
}

/**
 * CLI args ↔ textarea text. Flag+value pairs share a line ("-c 32768");
 * bare flags stand alone.
 */
export function extraArgsToText(args?: string[]): string {
	if (!args?.length) return '';
	const lines: string[] = [];
	for (let i = 0; i < args.length; i++) {
		const flag = args[i];
		const next = args[i + 1];
		if (flag.startsWith('-') && next !== undefined && !next.startsWith('-')) {
			lines.push(`${flag} ${next}`);
			i++;
		} else {
			lines.push(flag);
		}
	}
	return lines.join('\n');
}

export function textToExtraArgs(text: string): string[] {
	return text
		.split('\n')
		.flatMap((line) => line.trim().split(/\s+/))
		.filter(Boolean);
}

/** Comma/space separated text ↔ string list (for aliases). */
export function listToText(items?: string[]): string {
	return items?.join(', ') ?? '';
}

export function textToList(text: string): string[] | undefined {
	const items = text
		.split(/[,\s]+/)
		.map((s) => s.trim())
		.filter(Boolean);
	return items.length ? items : undefined;
}

/** Comma-separated GPU indices ↔ number list (for balanceGpus / gpus). */
export function gpuListToText(gpus?: number[]): string {
	return gpus?.join(', ') ?? '';
}

export function textToGpuList(text: string): number[] | undefined {
	const gpus = text
		.split(/[,\s]+/)
		.map((s) => parseInt(s, 10))
		.filter((n) => Number.isInteger(n) && n >= 0);
	return gpus.length ? gpus : undefined;
}
