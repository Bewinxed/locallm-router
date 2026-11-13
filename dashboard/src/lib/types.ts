export type ModelState = 'stopped' | 'starting' | 'running' | 'stopping' | 'sleeping';

export interface GpuStats {
	index: number;
	name: string;
	temperatureC: number;
	utilizationPercent: number;
	memoryUsedMB: number;
	memoryTotalMB: number;
	memoryFreeMB: number;
	powerDrawW: number;
	powerLimitW: number;
}

export interface ModelStatus {
	name: string;
	backend: string;
	modelPath: string;
	status: ModelState;
	lastActivity: number;
	downloading: boolean;
	downloadError: string | null;
	aliases: string[];
	balance: BalanceGroupStatus | null;
	priority?: number;
	sleepDelay?: number;
	gpu?: number;
	gpus?: number[];
}

export interface BalanceGroupStatus {
	baseName: string;
	primaryGpu: number;
	instances: BalanceInstance[];
}

export interface BalanceInstance {
	gpu: number;
	status: ModelState;
	inflight: number;
	internalPort: number;
	lastActivity: number;
}

/** VRAM figures from the GPU lock are gigabytes, unlike nvidia-smi's megabytes. */
export interface GpuLockStatus {
	totalVRAM: number;
	usedVRAM: number;
	availableVRAM: number;
	loaded: {
		name: string;
		estimatedVRAM: number;
		observedVRAM: number | null;
		/** Observed once measured, falling back to the estimate. */
		effectiveVRAM: number;
		priority: number;
		lastActivity: number;
	}[];
}

export interface DashboardData {
	models: ModelStatus[];
	gpus: Record<string, GpuLockStatus>;
	balanceGroups: BalanceGroupStatus[];
	gpuStats: GpuStats[];
}

/** An entry from the router's OpenAI-compatible /v1/models listing. */
export interface RouterModel {
	id: string;
	object: 'model';
	owned_by: string;
	root: string;
	status: ModelState;
	gpu: number[];
	aliases: string[];
	balance?: {
		enabled: boolean;
		gpus: number[];
		/** Routable names, e.g. "qwen:balance", "qwen:gpu0" */
		modifiers: string[];
		instances: { gpu: number; status: ModelState }[];
	};
}
