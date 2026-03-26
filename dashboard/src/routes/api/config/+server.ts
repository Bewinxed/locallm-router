import { json } from '@sveltejs/kit';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import type { RequestHandler } from './$types';

function getConfigPath(): string {
	// Docker path first, fallback to local dev
	const dockerPath = '/app/config.json';
	const localPath = new URL('../../../../config.json', import.meta.url).pathname;

	if (existsSync(dockerPath)) {
		return dockerPath;
	}
	return localPath;
}

export const GET: RequestHandler = async () => {
	try {
		const configPath = getConfigPath();
		const raw = readFileSync(configPath, 'utf-8');
		const config = JSON.parse(raw);
		return json(config);
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Failed to read config: ${message}` }, { status: 500 });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const configPath = getConfigPath();
		const body = await request.json();
		writeFileSync(configPath, JSON.stringify(body, null, 2), 'utf-8');
		return json({ success: true });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		return json({ error: `Failed to write config: ${message}` }, { status: 500 });
	}
};
