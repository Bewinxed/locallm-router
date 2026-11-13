import { toast } from 'svelte-sonner';
import type { ModelState } from './types';

export type LifecycleAction = 'start' | 'stop' | 'restart';

const PENDING_LABEL: Record<LifecycleAction, string> = {
	start: 'starting',
	stop: 'stopping',
	restart: 'restarting'
};

/**
 * In-flight lifecycle actions, keyed by model name. Module-scoped so a start
 * kicked off on Overview still reads as pending if you navigate to Settings.
 */
const pending = $state<Record<string, LifecycleAction | undefined>>({});

export function pendingAction(model: string): LifecycleAction | undefined {
	return pending[model];
}

export function isPending(model: string): boolean {
	return pending[model] !== undefined;
}

/**
 * Drive a model lifecycle action, reporting the outcome as a toast.
 * Resolves true on success so callers can refresh.
 */
export async function runModelAction(
	model: string,
	action: LifecycleAction
): Promise<boolean> {
	if (pending[model]) return false;
	pending[model] = action;

	const toastId = toast.loading(`${PENDING_LABEL[action]} ${model}…`);
	try {
		const res = await fetch(
			`/api/models/${encodeURIComponent(model)}/${action}`,
			{ method: 'POST' }
		);
		const data = await res.json().catch(() => ({}));
		if (!res.ok) throw new Error(data?.error || `Could not ${action} ${model}`);

		toast.success(`${model} ${data?.status ?? `${action}ed`}`, { id: toastId });
		return true;
	} catch (err) {
		toast.error(err instanceof Error ? err.message : `Could not ${action} ${model}`, {
			id: toastId
		});
		return false;
	} finally {
		pending[model] = undefined;
	}
}

// ─── Presentation ─────────────────────────────────────────────

export const STATUS_META: Record<ModelState, { label: string; dot: string; text: string }> = {
	running: { label: 'Running', dot: 'bg-emerald-500', text: 'text-emerald-500' },
	sleeping: { label: 'Sleeping', dot: 'bg-violet-400', text: 'text-violet-400' },
	starting: { label: 'Starting', dot: 'bg-amber-500 animate-pulse', text: 'text-amber-500' },
	stopping: { label: 'Stopping', dot: 'bg-amber-500 animate-pulse', text: 'text-amber-500' },
	stopped: { label: 'Stopped', dot: 'bg-muted-foreground/30', text: 'text-muted-foreground' }
};

export function statusMeta(status: string) {
	return STATUS_META[status as ModelState] ?? STATUS_META.stopped;
}

/** A model counts as holding VRAM when it is resident on a GPU. */
export function isResident(status: string): boolean {
	return status === 'running' || status === 'starting' || status === 'sleeping';
}

const HUES = [265, 160, 40, 310, 95, 200, 345, 130, 20, 235];

/**
 * Stable per-model colour. Derived from a hash of the name so a model keeps
 * its colour across reloads, reorders and GPUs — unlike an insertion counter.
 */
export function modelColor(name: string): string {
	let hash = 0;
	for (let i = 0; i < name.length; i++) {
		hash = (hash * 31 + name.charCodeAt(i)) | 0;
	}
	return `oklch(0.68 0.15 ${HUES[Math.abs(hash) % HUES.length]})`;
}
