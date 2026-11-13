import { json } from '@sveltejs/kit';
import { callUpstream, proxyUrl } from '$lib/server/manager';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async () => {
	// Deliberately /manager/stop-all on the proxy port, NOT :30099/stop-all —
	// the management-port variant calls proxy.stop(), which also shuts down
	// the inference listener itself.
	const result = await callUpstream(proxyUrl('/manager/stop-all'), {
		method: 'POST',
		timeoutMs: 60_000
	});

	if (!result.ok) {
		return json({ error: `Failed to reach router: ${result.error}` }, { status: 502 });
	}
	return json(result.body, { status: result.status });
};
