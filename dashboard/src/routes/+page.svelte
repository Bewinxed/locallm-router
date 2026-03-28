<script lang="ts">
	import { onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import SpinnerIcon from '~icons/ph/spinner';
	import PlayIcon from '~icons/ph/play';
	import StopIcon from '~icons/ph/stop';
	import ArrowClockwiseIcon from '~icons/ph/arrow-clockwise';
	import MoonIcon from '~icons/ph/moon';
	import type { DashboardData, GpuStats, GpuLockStatus, ModelStatus } from '$lib/types';

	let data = $state<DashboardData | null>(null);
	let error: string | null = $state(null);
	let loading = $state(true);
	let now = $state(Date.now());
	let hoveredGpu = $state<number | null>(null);
	let modelActions: Record<string, 'restarting' | 'stopping' | 'starting' | null> = $state({});

	// Deterministic model colors -- hue-shifted from a base
	const modelColorCache = new Map<string, string>();
	let colorIndex = 0;
	const hues = [260, 160, 45, 300, 80, 200, 340, 120, 20, 240];
	function modelColor(name: string): string {
		if (modelColorCache.has(name)) return modelColorCache.get(name)!;
		const hue = hues[colorIndex % hues.length];
		colorIndex++;
		const color = `oklch(0.65 0.15 ${hue})`;
		modelColorCache.set(name, color);
		return color;
	}

	async function modelAction(modelName: string, action: 'restart' | 'stop' | 'start') {
		modelActions[modelName] = action === 'restart' ? 'restarting' : action === 'stop' ? 'stopping' : 'starting';
		try {
			const res = await fetch(`/api/models/${encodeURIComponent(modelName)}/${action}`, { method: 'POST' });
			const d = await res.json();
			if (!res.ok) throw new Error(d.error || `Failed to ${action} ${modelName}`);
			toast.success(`${modelName}: ${d.status || action + 'ed'}`);
		} catch (e) {
			toast.error(e instanceof Error ? e.message : `Failed to ${action} ${modelName}`);
		} finally {
			modelActions[modelName] = null;
		}
	}

	function relativeTime(timestamp: number): string {
		if (!timestamp) return 'never';
		const diff = now - timestamp;
		if (diff < 0) return 'now';
		const s = Math.floor(diff / 1000);
		if (s < 5) return 'now';
		if (s < 60) return `${s}s`;
		const m = Math.floor(s / 60);
		if (m < 60) return `${m}m`;
		const h = Math.floor(m / 60);
		if (h < 24) return `${h}h`;
		return `${Math.floor(h / 24)}d`;
	}

	function formatGB(mb: number): string {
		return `${(mb / 1024).toFixed(1)}`;
	}

	function vramPercent(gpu: GpuStats): number {
		if (!gpu.memoryTotalMB) return 0;
		return Math.round((gpu.memoryUsedMB / gpu.memoryTotalMB) * 100);
	}

	async function fetchStatus() {
		try {
			const res = await fetch('/api/status');
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			data = await res.json();
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fetch';
		} finally {
			loading = false;
		}
	}

	let pollInterval: ReturnType<typeof setInterval> | undefined;
	let timeInterval: ReturnType<typeof setInterval> | undefined;

	$effect(() => {
		fetchStatus();
		pollInterval = setInterval(fetchStatus, 2000);
		timeInterval = setInterval(() => { now = Date.now(); }, 1000);
		return () => {
			if (pollInterval) clearInterval(pollInterval);
			if (timeInterval) clearInterval(timeInterval);
		};
	});

	let gpuStats = $derived(data?.gpuStats ?? []);
	let models = $derived(data?.models ?? []);
	let gpuLocks = $derived(data?.gpus ?? {});

	// Sort: running first, then sleeping, then starting, then stopped
	let sortedModels = $derived(
		[...models].sort((a, b) => {
			const order: Record<string, number> = { running: 0, sleeping: 1, starting: 2, stopping: 3, stopped: 4 };
			return (order[a.status] ?? 5) - (order[b.status] ?? 5);
		})
	);

	let runningCount = $derived(models.filter(m => m.status === 'running').length);
	let sleepingCount = $derived(models.filter(m => m.status === 'sleeping').length);

	// Status styling
	function statusDot(status: string): string {
		switch (status) {
			case 'running': return 'bg-emerald-500';
			case 'sleeping': return 'bg-violet-400';
			case 'starting': return 'bg-amber-500 animate-pulse';
			case 'stopping': return 'bg-amber-500 animate-pulse';
			default: return 'bg-muted-foreground/30';
		}
	}

	function statusLabel(status: string): string {
		switch (status) {
			case 'running': return 'Running';
			case 'sleeping': return 'Sleeping';
			case 'starting': return 'Starting';
			case 'stopping': return 'Stopping';
			default: return 'Stopped';
		}
	}
</script>

<svelte:head>
	<title>locallm-router</title>
</svelte:head>

<div class="flex flex-col h-full">
	<!-- Header -->
	<header class="flex items-center justify-between px-6 py-4 border-b border-border">
		<div class="flex items-center gap-3">
			<h1 class="text-sm font-semibold text-foreground">Overview</h1>
			{#if !loading}
				<span class="text-xs text-muted-foreground">
					{runningCount} running{#if sleepingCount > 0}, {sleepingCount} sleeping{/if}
					&middot; {models.length} total
				</span>
			{/if}
		</div>
		{#if error}
			<span class="text-xs text-destructive">Disconnected</span>
		{/if}
	</header>

	<div class="flex-1 overflow-auto">
		<div class="max-w-6xl mx-auto px-6 py-5 space-y-6">

			<!-- GPU Strip -->
			{#if gpuStats.length > 0}
				<section class="space-y-3">
					<div class="text-xs font-medium text-muted-foreground uppercase tracking-wider">GPUs</div>

					<div class="space-y-2">
						{#each gpuStats as gpu (gpu.index)}
							{@const pct = vramPercent(gpu)}
							{@const lock = gpuLocks[String(gpu.index)]}
							{@const isHovered = hoveredGpu === gpu.index}

							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<div
								class="group rounded-lg border border-border bg-card p-3 transition-colors {isHovered ? 'ring-1 ring-ring/20' : ''}"
								onmouseenter={() => hoveredGpu = gpu.index}
								onmouseleave={() => hoveredGpu = null}
							>
								<!-- GPU header row -->
								<div class="flex items-center gap-3 mb-2">
									<span class="text-xs font-semibold text-foreground w-12 shrink-0">GPU {gpu.index}</span>
									<span class="text-xs text-muted-foreground truncate flex-1">{gpu.name}</span>
									<div class="flex items-center gap-3 text-xs tabular-nums text-muted-foreground">
										<span>{gpu.temperatureC}°C</span>
										<span>{gpu.utilizationPercent}%</span>
										<span>{Math.round(gpu.powerDrawW)}W</span>
									</div>
									<span class="text-xs tabular-nums font-medium text-foreground w-28 text-right">
										{formatGB(gpu.memoryUsedMB)} / {formatGB(gpu.memoryTotalMB)} GB
									</span>
								</div>

								<!-- Segmented VRAM bar -->
								<div class="h-3 w-full rounded-sm bg-secondary overflow-hidden flex">
									{#if lock?.loaded?.length}
										{#each lock.loaded as loaded (loaded.name)}
											{@const segPct = (loaded.vram / gpu.memoryTotalMB) * 100}
											<Tooltip.Root>
												<Tooltip.Trigger>
													<div
														class="h-full vram-segment rounded-sm"
														style="width: {segPct}%; background-color: {modelColor(loaded.name)}; min-width: 2px;"
													></div>
												</Tooltip.Trigger>
												<Tooltip.Content class="text-xs">
													<span class="font-medium">{loaded.name}</span>
													<span class="text-muted-foreground ml-1">{formatGB(loaded.vram)} GB</span>
												</Tooltip.Content>
											</Tooltip.Root>
										{/each}
									{:else}
										<!-- System/other VRAM usage -->
										<div
											class="h-full rounded-sm bg-muted-foreground/15 vram-segment"
											style="width: {pct}%"
										></div>
									{/if}
								</div>

								<!-- Loaded model labels -->
								{#if lock?.loaded?.length}
									<div class="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5">
										{#each lock.loaded as loaded (loaded.name)}
											<span class="flex items-center gap-1.5 text-[11px] text-muted-foreground">
												<span
													class="inline-block size-2 rounded-sm shrink-0"
													style="background-color: {modelColor(loaded.name)}"
												></span>
												{loaded.name}
												<span class="tabular-nums">{formatGB(loaded.vram)}</span>
											</span>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				</section>
			{/if}

			<!-- Model List -->
			<section class="space-y-3">
				<div class="text-xs font-medium text-muted-foreground uppercase tracking-wider">Models</div>

				{#if loading && !data}
					<div class="flex items-center gap-2 py-8 justify-center text-sm text-muted-foreground">
						<SpinnerIcon class="size-4 animate-spin" />
						Loading...
					</div>
				{:else if sortedModels.length === 0}
					<div class="py-12 text-center text-sm text-muted-foreground">
						No models configured. Add models to config.json and restart.
					</div>
				{:else}
					<!-- Table header -->
					<div class="grid grid-cols-[auto_1fr_80px_60px_70px_90px_100px] items-center gap-x-3 px-3 py-1.5 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
						<div class="w-2"></div>
						<div>Name</div>
						<div>Backend</div>
						<div>GPU</div>
						<div>Status</div>
						<div class="text-right">Last active</div>
						<div class="text-right">Actions</div>
					</div>

					<!-- Model rows -->
					<div class="rounded-lg border border-border bg-card overflow-hidden divide-y divide-border">
						{#each sortedModels as model (model.name)}
							{@const actionState = modelActions[model.name]}
							{@const isStopped = model.status === 'stopped'}
							{@const isSleeping = model.status === 'sleeping'}

							<div
								class="model-row grid grid-cols-[auto_1fr_80px_60px_70px_90px_100px] items-center gap-x-3 px-3 py-2.5 {isStopped ? 'opacity-50' : ''}"
							>
								<!-- Status dot -->
								<div class="flex items-center justify-center w-2">
									<span class="size-2 rounded-full {statusDot(model.status)}"></span>
								</div>

								<!-- Name + aliases -->
								<div class="min-w-0">
									<div class="flex items-center gap-2">
										<span class="text-sm font-medium text-foreground truncate">{model.name}</span>
										{#if model.downloading}
											<SpinnerIcon class="size-3 animate-spin text-muted-foreground shrink-0" />
										{/if}
										{#if isSleeping}
											<MoonIcon class="size-3 text-violet-400 shrink-0" />
										{/if}
									</div>
									{#if model.aliases.length > 0}
										<div class="text-[11px] text-muted-foreground truncate">
											{model.aliases.join(', ')}
										</div>
									{/if}
									{#if model.downloadError}
										<div class="text-[11px] text-destructive truncate">{model.downloadError}</div>
									{/if}
								</div>

								<!-- Backend -->
								<div class="text-xs text-muted-foreground tabular-nums">{model.backend}</div>

								<!-- GPU -->
								<div class="text-xs text-muted-foreground tabular-nums">
									{#if model.balance}
										{model.balance.instances.map(i => i.gpu).join(',')}
									{:else if model.gpus?.length}
										{model.gpus.join(',')}
									{:else if model.gpu != null}
										{model.gpu}
									{:else}
										-
									{/if}
								</div>

								<!-- Status -->
								<div class="text-xs text-muted-foreground">{statusLabel(model.status)}</div>

								<!-- Last active -->
								<div class="text-xs tabular-nums text-muted-foreground text-right">
									{relativeTime(model.lastActivity)}
								</div>

								<!-- Actions -->
								<div class="flex items-center justify-end gap-1">
									{#if model.status === 'running' || model.status === 'starting' || model.status === 'sleeping'}
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Button
													variant="ghost"
													size="icon"
													class="size-7 text-muted-foreground hover:text-destructive"
													disabled={!!actionState}
													onclick={() => modelAction(model.name, 'stop')}
												>
													{#if actionState === 'stopping'}
														<SpinnerIcon class="size-3.5 animate-spin" />
													{:else}
														<StopIcon class="size-3.5" />
													{/if}
												</Button>
											</Tooltip.Trigger>
											<Tooltip.Content class="text-xs">Stop</Tooltip.Content>
										</Tooltip.Root>
									{:else}
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Button
													variant="ghost"
													size="icon"
													class="size-7 text-muted-foreground hover:text-foreground"
													disabled={!!actionState}
													onclick={() => modelAction(model.name, 'start')}
												>
													{#if actionState === 'starting'}
														<SpinnerIcon class="size-3.5 animate-spin" />
													{:else}
														<PlayIcon class="size-3.5" />
													{/if}
												</Button>
											</Tooltip.Trigger>
											<Tooltip.Content class="text-xs">Start</Tooltip.Content>
										</Tooltip.Root>
									{/if}
									<Tooltip.Root>
										<Tooltip.Trigger>
											<Button
												variant="ghost"
												size="icon"
												class="size-7 text-muted-foreground hover:text-foreground"
												disabled={!!actionState}
												onclick={() => modelAction(model.name, 'restart')}
											>
												{#if actionState === 'restarting'}
													<SpinnerIcon class="size-3.5 animate-spin" />
												{:else}
													<ArrowClockwiseIcon class="size-3.5" />
												{/if}
											</Button>
										</Tooltip.Trigger>
										<Tooltip.Content class="text-xs">Restart</Tooltip.Content>
									</Tooltip.Root>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</section>

		</div>
	</div>
</div>
