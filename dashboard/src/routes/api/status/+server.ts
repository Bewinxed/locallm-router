import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import type { GpuStats, ManagerStatus, DashboardData } from '$lib/types';

const MANAGER_URL = 'http://localhost:30000/manager/status';

function parseNvidiaSmi(output: string): GpuStats[] {
	const lines = output.trim().split('\n').filter(Boolean);
	return lines.map((line) => {
		const parts = line.split(',').map((s) => s.trim());
		return {
			index: parseInt(parts[0], 10),
			name: parts[1],
			temperatureC: parseInt(parts[2], 10),
			utilizationPercent: parseInt(parts[3], 10),
			memoryUsedMB: parseInt(parts[4], 10),
			memoryTotalMB: parseInt(parts[5], 10),
			memoryFreeMB: parseInt(parts[6], 10),
			powerDrawW: parseFloat(parts[7]),
			powerLimitW: parseFloat(parts[8])
		};
	});
}

async function getGpuStats(): Promise<GpuStats[]> {
	try {
		const proc = Bun.spawn([
			'nvidia-smi',
			'--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,memory.free,power.draw,power.limit',
			'--format=csv,noheader,nounits'
		], {
			stdout: 'pipe',
			stderr: 'pipe'
		});

		const timeout = setTimeout(() => proc.kill(), 5000);
		try {
			const [stdout, exitCode] = await Promise.all([
				new Response(proc.stdout).text(),
				proc.exited
			]);

			if (exitCode !== 0) return [];
			return parseNvidiaSmi(stdout);
		} finally {
			clearTimeout(timeout);
		}
	} catch {
		return [];
	}
}

async function getManagerStatus(): Promise<ManagerStatus | null> {
	try {
		const res = await fetch(MANAGER_URL, {
			signal: AbortSignal.timeout(5000)
		});
		if (!res.ok) return null;
		return (await res.json()) as ManagerStatus;
	} catch {
		return null;
	}
}

export const GET: RequestHandler = async () => {
	const [managerStatus, gpuStats] = await Promise.all([getManagerStatus(), getGpuStats()]);

	const result: DashboardData = {
		models: managerStatus?.models ?? [],
		gpus: managerStatus?.gpus ?? {},
		balanceGroups: managerStatus?.balanceGroups ?? [],
		gpuStats
	};

	return json(result, {
		headers: {
			'Cache-Control': 'no-store'
		}
	});
};
