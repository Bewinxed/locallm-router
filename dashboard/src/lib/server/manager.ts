/**
 * Upstream router endpoints.
 *
 * Two servers, deliberately:
 *   PROXY (config.port, default 30000)      — OpenAI surface + /manager/status
 *   MANAGEMENT (managementPort, def. 30099) — lifecycle: start/stop/restart/reload
 *
 * `restart` and `reload-config` exist ONLY on the management port, so all
 * lifecycle actions go there for consistency.
 */
const PROXY_URL = process.env.ROUTER_PROXY_URL ?? 'http://localhost:30000';
const MANAGEMENT_URL = process.env.ROUTER_MANAGEMENT_URL ?? 'http://localhost:30099';

export const proxyUrl = (path: string) => `${PROXY_URL}${path}`;
export const managementUrl = (path: string) => `${MANAGEMENT_URL}${path}`;

/** Lifecycle actions the management API accepts, as a runtime-checkable list. */
export const MODEL_ACTIONS = ['start', 'stop', 'restart', 'reload-config'] as const;
export type ModelAction = (typeof MODEL_ACTIONS)[number];

export function isModelAction(value: string): value is ModelAction {
	return (MODEL_ACTIONS as readonly string[]).includes(value);
}

/**
 * Fetch with a timeout that reports upstream-unreachable as a typed result
 * instead of throwing, so routes can return a clean 502 with a useful message.
 */
export async function callUpstream(
	url: string,
	init: RequestInit & { timeoutMs?: number } = {}
): Promise<{ ok: true; status: number; body: unknown } | { ok: false; error: string }> {
	const { timeoutMs = 30_000, ...rest } = init;
	try {
		const res = await fetch(url, { ...rest, signal: AbortSignal.timeout(timeoutMs) });
		const text = await res.text();
		let body: unknown = null;
		try {
			body = text ? JSON.parse(text) : null;
		} catch {
			body = { error: text || res.statusText };
		}
		return { ok: true, status: res.status, body };
	} catch (err) {
		const message =
			err instanceof Error
				? err.name === 'TimeoutError'
					? 'Router did not respond in time'
					: err.message
				: 'Unknown error';
		return { ok: false, error: message };
	}
}
