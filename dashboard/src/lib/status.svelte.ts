import type { DashboardData } from './types';

/**
 * Single shared poll of /api/status for the whole app.
 *
 * Pages subscribe via `useStatus()` inside an $effect; the underlying interval
 * is reference-counted, so navigating between Overview and Settings never runs
 * two polls at once, and the last page to unmount stops it.
 */
const POLL_MS = 2000;

let data = $state<DashboardData | null>(null);
let error = $state<string | null>(null);
let loading = $state(true);

let timer: ReturnType<typeof setInterval> | null = null;
let subscribers = 0;
let inFlight = false;

export async function refresh(): Promise<void> {
	// Slow networks or a wedged router shouldn't stack requests.
	if (inFlight) return;
	inFlight = true;
	try {
		const res = await fetch('/api/status');
		if (!res.ok) throw new Error(`Router returned ${res.status}`);
		data = await res.json();
		error = null;
	} catch (err) {
		error = err instanceof Error ? err.message : 'Cannot reach the router';
	} finally {
		loading = false;
		inFlight = false;
	}
}

/** Subscribe to the shared poll. Call inside an $effect and return its result. */
export function useStatus(): () => void {
	subscribers++;
	refresh();
	timer ??= setInterval(refresh, POLL_MS);

	return () => {
		subscribers--;
		if (subscribers === 0 && timer) {
			clearInterval(timer);
			timer = null;
		}
	};
}

export const status = {
	get data() {
		return data;
	},
	get error() {
		return error;
	},
	get loading() {
		return loading;
	},
	get models() {
		return data?.models ?? [];
	},
	get gpuStats() {
		return data?.gpuStats ?? [];
	},
	get gpuLocks() {
		return data?.gpus ?? {};
	}
};

/** A clock that ticks only while something is rendering relative times. */
export function useNow(intervalMs = 1000): { readonly current: number } {
	let value = $state(Date.now());
	$effect(() => {
		const id = setInterval(() => (value = Date.now()), intervalMs);
		return () => clearInterval(id);
	});
	return {
		get current() {
			return value;
		}
	};
}
