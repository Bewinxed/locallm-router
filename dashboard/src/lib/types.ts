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
	status: 'stopped' | 'starting' | 'running' | 'stopping';
	lastActivity: number;
	downloading: boolean;
	downloadError: string | null;
	aliases: string[];
	balance: BalanceGroupStatus | null;
}

export interface BalanceGroupStatus {
	baseName: string;
	primaryGpu: number;
	instances: BalanceInstance[];
}

export interface BalanceInstance {
	gpu: number;
	status: string;
	inflight: number;
	internalPort: number;
	lastActivity: number;
}

export interface GpuLockStatus {
	totalVRAM: number;
	usedVRAM: number;
	availableVRAM: number;
	loaded: { name: string; vram: number; lastActivity: number }[];
}

export interface ManagerStatus {
	models: ModelStatus[];
	gpus: Record<string, GpuLockStatus>;
	balanceGroups: BalanceGroupStatus[];
}

export interface RequestLogEntry {
	id: string;
	timestamp: number;
	model: string;
	modifier: string | null;
	targetGpu: number;
	status: 'pending' | 'streaming' | 'complete' | 'error';
	promptTokens?: number;
	completionTokens?: number;
	tokensPerSecond?: number;
	durationMs?: number;
}

export interface DashboardData {
	models: ModelStatus[];
	gpus: Record<string, GpuLockStatus>;
	balanceGroups: BalanceGroupStatus[];
	gpuStats: GpuStats[];
}
