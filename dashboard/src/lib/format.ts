/** Compact duration since a timestamp: "now", "42s", "8m", "3h", "2d". */
export function relativeTime(timestamp: number, now: number = Date.now()): string {
	if (!timestamp) return 'never';
	const seconds = Math.floor((now - timestamp) / 1000);
	if (seconds < 5) return 'now';
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours}h`;
	return `${Math.floor(hours / 24)}d`;
}

/** Megabytes to a one-decimal gigabyte string, without the unit. */
export function formatGB(mb: number): string {
	return (mb / 1024).toFixed(1);
}

export function percent(used: number, total: number): number {
	if (!total) return 0;
	return Math.round((used / total) * 100);
}
