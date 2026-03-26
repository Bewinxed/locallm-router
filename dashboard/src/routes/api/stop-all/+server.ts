import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const MANAGER_URL = 'http://localhost:30099/stop-all';

export const POST: RequestHandler = async () => {
	try {
		const res = await fetch(MANAGER_URL, {
			method: 'POST',
			signal: AbortSignal.timeout(30000)
		});

		if (!res.ok) {
			const text = await res.text();
			return json(
				{ error: `Manager returned ${res.status}: ${text}` },
				{ status: res.status }
			);
		}

		const data = await res.json();
		return json(data);
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Failed to reach manager: ${message}` }, { status: 502 });
	}
};
