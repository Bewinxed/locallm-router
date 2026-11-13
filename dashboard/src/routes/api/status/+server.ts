import { json } from '@sveltejs/kit';
import { callUpstream, proxyUrl } from '$lib/server/manager';
import type { DashboardData, GpuStats } from '$lib/types';
import type { RequestHandler } from './$types';

const NVIDIA_SMI_FIELDS = [
	'index',
	'name',
	'temperature.gpu',
	'utilization.gpu',
	'memory.used',
	'memory.total',
	'memory.free',
	'power.draw',
	'power.limit'
] as const;

function parseNvidiaSmi(output: string): GpuStats[] {
	return output
		.trim()
		.split('\n')
		.filter(Boolean)
		.map((line) => {
			const [index, name, temp, util, used, total, free, draw, limit] = line
				.split(',')
				.map((s) => s.trim());
			return {
				index: parseInt(index, 10),
				name,
				temperatureC: parseInt(temp, 10),
				utilizationPercent: parseInt(util, 10),
				memoryUsedMB: parseInt(used, 10),
				memoryTotalMB: parseInt(total, 10),
				memoryFreeMB: parseInt(free, 10),
				powerDrawW: parseFloat(draw),
				powerLimitW: parseFloat(limit)
			};
		})
		.filter((gpu) => Number.isFinite(gpu.index));
}

/** Returns [] on any failure — a CPU-only host is a valid state, not an error. */
async function getGpuStats(): Promise<GpuStats[]> {
	try {
		const proc = Bun.spawn(
			['nvidia-smi', `--query-gpu=${NVIDIA_SMI_FIELDS.join(',')}`, '--format=csv,noheader,nounits'],
			{ stdout: 'pipe', stderr: 'ignore' }
		);

		const kill = setTimeout(() => proc.kill(), 5000);
		try {
			const [stdout, exitCode] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);
			return exitCode === 0 ? parseNvidiaSmi(stdout) : [];
		} finally {
			clearTimeout(kill);
		}
	} catch {
		return [];
	}
}

export const GET: RequestHandler = async () => {
	const [manager, gpuStats] = await Promise.all([
		callUpstream(proxyUrl('/manager/status'), { timeoutMs: 5000 }),
		getGpuStats()
	]);

	const managerData =
		manager.ok && manager.status === 200 ? (manager.body as Partial<DashboardData>) : null;

	const result: DashboardData = {
		models: managerData?.models ?? [],
		gpus: managerData?.gpus ?? {},
		balanceGroups: managerData?.balanceGroups ?? [],
		gpuStats
	};

	// A reachable dashboard with an unreachable router should say so, rather
	// than silently rendering an empty model list as if nothing is configured.
	const status = managerData ? 200 : 502;
	return json(result, { status, headers: { 'Cache-Control': 'no-store' } });
};
