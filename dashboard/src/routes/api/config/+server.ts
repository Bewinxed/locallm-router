import { json } from '@sveltejs/kit';
import { copyFileSync, existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { RequestHandler } from './$types';

/**
 * Same resolution order as the router itself: LLM_ROUTER_CONFIG first, then
 * the Docker path, then the repo root during local dev. (import.meta.url is
 * useless after bundling — it points inside build/.)
 */
function getConfigPath(): string {
	if (process.env.LLM_ROUTER_CONFIG) return process.env.LLM_ROUTER_CONFIG;
	if (existsSync('/app/config.json')) return '/app/config.json';
	return resolve(process.cwd(), '../config.json');
}

export const GET: RequestHandler = async () => {
	try {
		const config = JSON.parse(readFileSync(getConfigPath(), 'utf-8'));
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

		// Reject obviously-invalid shapes before touching the file: a bad write
		// here silently downgrades the router to "zero models configured".
		if (!body || typeof body !== 'object' || !Array.isArray(body.models)) {
			return json({ error: 'Config must be an object with a models array' }, { status: 400 });
		}

		// Atomic write: temp file + rename, with a .bak of the previous version,
		// so a crash mid-write can never leave config.json unparseable.
		if (existsSync(configPath)) {
			copyFileSync(configPath, `${configPath}.bak`);
		}
		const tmpPath = `${configPath}.tmp`;
		writeFileSync(tmpPath, JSON.stringify(body, null, 2) + '\n', 'utf-8');
		renameSync(tmpPath, configPath);

		return json({ success: true });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Unknown error';
		const hint =
			message.includes('EROFS') || message.includes('read-only')
				? ' (config.json is mounted read-only — remove the :ro flag in docker-compose.yml to edit from the dashboard)'
				: '';
		return json({ error: `Failed to write config: ${message}${hint}` }, { status: 500 });
	}
};
