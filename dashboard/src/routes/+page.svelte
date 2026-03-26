<script lang="ts">
	import { onDestroy } from 'svelte';
	import { autoAnimate } from '@formkit/auto-animate';
	import { toast } from 'svelte-sonner';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	// Progress not currently used but kept available
	// import { Progress } from '$lib/components/ui/progress';
	import GpuIcon from '~icons/ph/graphics-card';
	import ThermometerIcon from '~icons/ph/thermometer';
	import LightningIcon from '~icons/ph/lightning';
	import HardDrivesIcon from '~icons/ph/hard-drives';
	import PulseIcon from '~icons/ph/pulse';
	import SpinnerIcon from '~icons/ph/spinner';
	import CircleIcon from '~icons/ph/circle';
	import ClockIcon from '~icons/ph/clock';
	import ArrowsLeftRightIcon from '~icons/ph/arrows-left-right';
	import PlayIcon from '~icons/ph/play';
	import StopIcon from '~icons/ph/stop';
	import ArrowClockwiseIcon from '~icons/ph/arrow-clockwise';
	import type { DashboardData, GpuStats, ModelStatus, BalanceGroupStatus } from '$lib/types';

	let data = $state<DashboardData | null>(null);
	let error: string | null = $state(null);
	let loading = $state(true);

	let now = $state(Date.now());

	/** Per-model action state */
	let modelActions: Record<string, 'restarting' | 'stopping' | 'starting' | null> = $state({});

	async function modelAction(modelName: string, action: 'restart' | 'stop' | 'start') {
		modelActions[modelName] = action === 'restart' ? 'restarting' : action === 'stop' ? 'stopping' : 'starting';
		try {
			const res = await fetch(`/api/models/${encodeURIComponent(modelName)}/${action}`, {
				method: 'POST'
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data.error || `Failed to ${action} ${modelName}`);
			}
			toast.success(`${modelName}: ${data.status || action + 'ed'}`);
		} catch (e) {
			const msg = e instanceof Error ? e.message : `Failed to ${action} ${modelName}`;
			toast.error(msg);
		} finally {
			modelActions[modelName] = null;
		}
	}

	const statusColors: Record<string, string> = {
		running: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25',
		starting: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
		stopping: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
		stopped: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/25'
	};

	function relativeTime(timestamp: number): string {
		if (!timestamp) return 'never';
		const diff = now - timestamp;
		if (diff < 0) return 'just now';
		const seconds = Math.floor(diff / 1000);
		if (seconds < 5) return 'just now';
		if (seconds < 60) return `${seconds}s ago`;
		const minutes = Math.floor(seconds / 60);
		if (minutes < 60) return `${minutes}m ago`;
		const hours = Math.floor(minutes / 60);
		if (hours < 24) return `${hours}h ago`;
		const days = Math.floor(hours / 24);
		return `${days}d ago`;
	}

	function formatMB(mb: number): string {
		if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
		return `${Math.round(mb)} MB`;
	}

	function vramPercent(gpu: GpuStats): number {
		if (!gpu.memoryTotalMB) return 0;
		return Math.round((gpu.memoryUsedMB / gpu.memoryTotalMB) * 100);
	}

	function vramBarColor(percent: number): string {
		if (percent > 90) return 'bg-red-500';
		if (percent > 70) return 'bg-amber-500';
		return 'bg-emerald-500';
	}

	function tempColor(temp: number): string {
		if (temp > 85) return 'text-red-400';
		if (temp > 70) return 'text-amber-400';
		return 'text-emerald-400';
	}

	async function fetchStatus() {
		try {
			const res = await fetch('/api/status');
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			data = await res.json();
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fetch status';
		} finally {
			loading = false;
		}
	}

	let pollInterval: ReturnType<typeof setInterval> | undefined;
	let timeInterval: ReturnType<typeof setInterval> | undefined;

	$effect(() => {
		fetchStatus();
		pollInterval = setInterval(fetchStatus, 2000);
		timeInterval = setInterval(() => {
			now = Date.now();
		}, 1000);

		return () => {
			if (pollInterval) clearInterval(pollInterval);
			if (timeInterval) clearInterval(timeInterval);
		};
	});

	let gpuStats = $derived(data?.gpuStats ?? []);
	let models = $derived(data?.models ?? []);
	let balanceGroups = $derived(data?.balanceGroups ?? []);
	let gpuLocks = $derived(data?.gpus ?? {});

	// Models that don't have a balance group entry (standalone)
	let standaloneModels = $derived(
		models.filter((m) => !m.balance)
	);

	// Models that have balance groups
	let balancedModels = $derived(
		models.filter((m) => m.balance)
	);
</script>

<svelte:head>
	<title>locallm-router - Dashboard</title>
</svelte:head>

<div class="flex flex-col gap-6 p-6">
	<!-- Page header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold tracking-tight text-foreground">Dashboard</h1>
			<p class="text-sm text-muted-foreground">GPU cluster status and model management</p>
		</div>
		{#if loading && !data}
			<div class="flex items-center gap-2 text-sm text-muted-foreground">
				<SpinnerIcon class="size-4 animate-spin" />
				Loading...
			</div>
		{/if}
		{#if error}
			<Badge
				variant="destructive"
				class="animate-pulse"
			>
				Connection error: {error}
			</Badge>
		{/if}
	</div>

	<!-- GPU Cards -->
	{#if gpuStats.length > 0}
		<section>
			<h2 class="mb-3 text-sm font-medium uppercase tracking-wider text-muted-foreground">
				GPUs
			</h2>
			<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
				{#each gpuStats as gpu (gpu.index)}
					{@const pct = vramPercent(gpu)}
					<Card.Root class="overflow-hidden border-border/50 bg-card">
						<Card.Header class="pb-3">
							<div class="flex items-center justify-between">
								<div class="flex items-center gap-2">
									<div
										class="flex size-8 items-center justify-center rounded-md bg-amber-500/15 text-amber-500"
									>
										<GpuIcon class="size-4" />
									</div>
									<div>
										<Card.Title class="text-sm">GPU {gpu.index}</Card.Title>
										<Card.Description class="text-xs">{gpu.name}</Card.Description>
									</div>
								</div>
								<span class="{tempColor(gpu.temperatureC)} flex items-center gap-1 text-sm font-mono">
									<ThermometerIcon class="size-3.5" />
									{gpu.temperatureC}°C
								</span>
							</div>
						</Card.Header>
						<Card.Content class="space-y-3">
							<!-- VRAM bar -->
							<div class="space-y-1.5">
								<div class="flex items-center justify-between text-xs">
									<span class="text-muted-foreground">VRAM</span>
									<span class="font-mono text-foreground">
										{formatMB(gpu.memoryUsedMB)} / {formatMB(gpu.memoryTotalMB)}
									</span>
								</div>
								<div class="h-2 w-full overflow-hidden rounded-full bg-secondary">
									<div
										class="{vramBarColor(pct)} h-full rounded-full transition-all duration-500 ease-out"
										style="width: {pct}%"
									></div>
								</div>
								<div class="text-right text-xs font-mono text-muted-foreground">{pct}%</div>
							</div>

							<!-- Stats row -->
							<div class="flex items-center justify-between border-t border-border/50 pt-3">
								<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
									<PulseIcon class="size-3.5" />
									<span>Util: <span class="font-mono text-foreground">{gpu.utilizationPercent}%</span></span>
								</div>
								<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
									<LightningIcon class="size-3.5" />
									<span>Power: <span class="font-mono text-foreground">{Math.round(gpu.powerDrawW)}W</span> / {Math.round(gpu.powerLimitW)}W</span>
								</div>
							</div>

							<!-- Loaded models on this GPU -->
							{#if gpuLocks[String(gpu.index)]}
								{@const lock = gpuLocks[String(gpu.index)]}
								{#if lock.loaded.length > 0}
									<div class="border-t border-border/50 pt-3">
										<div class="text-xs text-muted-foreground mb-1.5">Loaded models</div>
										<div class="flex flex-wrap gap-1.5">
											{#each lock.loaded as loaded}
												<Badge variant="secondary" class="text-xs font-mono">
													{loaded.name}
													<span class="ml-1 text-muted-foreground">{formatMB(loaded.vram)}</span>
												</Badge>
											{/each}
										</div>
									</div>
								{/if}
							{/if}
						</Card.Content>
					</Card.Root>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Balance Groups -->
	{#if balancedModels.length > 0}
		<section>
			<h2 class="mb-3 text-sm font-medium uppercase tracking-wider text-muted-foreground">
				Balance Groups
			</h2>
			<div class="grid grid-cols-1 gap-4">
			{#each balancedModels as model (model.name)}
				{@const bg = model.balance}
				{@const actionState = modelActions[model.name]}
				{#if bg}
					<Card.Root class="overflow-hidden border-amber-500/20 bg-card">
						<Card.Header class="pb-3">
							<div class="flex items-center justify-between">
								<div class="flex items-center gap-3">
									<div
										class="flex size-8 items-center justify-center rounded-md bg-amber-500/15 text-amber-500"
									>
										<ArrowsLeftRightIcon class="size-4" />
									</div>
									<div>
										<Card.Title class="text-sm">{model.name}</Card.Title>
										<Card.Description class="text-xs">
											Balanced across {bg.instances.length} GPUs · Primary: GPU {bg.primaryGpu}
										</Card.Description>
									</div>
								</div>
								<div class="flex items-center gap-2">
									{#if model.status === 'running' || model.status === 'starting'}
										<Button
											variant="destructive"
											size="sm"
											class="h-7 text-xs"
											disabled={!!actionState}
											onclick={() => modelAction(model.name, 'stop')}
										>
											{#if actionState === 'stopping'}
												<SpinnerIcon class="size-3 animate-spin" />
											{:else}
												<StopIcon class="size-3" />
											{/if}
											Stop
										</Button>
									{:else}
										<Button
											variant="outline"
											size="sm"
											class="h-7 text-xs"
											disabled={!!actionState}
											onclick={() => modelAction(model.name, 'start')}
										>
											{#if actionState === 'starting'}
												<SpinnerIcon class="size-3 animate-spin" />
											{:else}
												<PlayIcon class="size-3" />
											{/if}
											Start
										</Button>
									{/if}
									<Button
										variant="outline"
										size="sm"
										class="h-7 text-xs"
										disabled={!!actionState}
										onclick={() => modelAction(model.name, 'restart')}
									>
										{#if actionState === 'restarting'}
											<SpinnerIcon class="size-3 animate-spin" />
										{:else}
											<ArrowClockwiseIcon class="size-3" />
										{/if}
										Restart
									</Button>
									<Badge
										variant="outline"
										class={statusColors[model.status] ?? statusColors.stopped}
									>
										{model.status}
									</Badge>
								</div>
							</div>
						</Card.Header>
							<Card.Content>
								<div class="grid grid-cols-1 gap-3 sm:grid-cols-{bg.instances.length}">
									{#each bg.instances as instance (instance.gpu)}
										{@const isActive = instance.inflight > 0}
										<div
											class="relative rounded-lg border p-3 transition-colors {isActive
												? 'border-amber-500/40 bg-amber-500/5'
												: 'border-border/50 bg-secondary/30'}"
										>
											<!-- Active indicator dot -->
											{#if isActive}
												<div class="absolute right-2 top-2">
													<span class="relative flex size-2.5">
														<span
															class="absolute inline-flex size-full animate-ping rounded-full bg-amber-400 opacity-75"
														></span>
														<span class="relative inline-flex size-2.5 rounded-full bg-amber-500"></span>
													</span>
												</div>
											{/if}

											<div class="flex items-center gap-2 mb-2">
												<GpuIcon class="size-3.5 text-muted-foreground" />
												<span class="text-xs font-medium">GPU {instance.gpu}</span>
												{#if instance.gpu === bg.primaryGpu}
													<Badge variant="outline" class="text-[10px] px-1.5 py-0 border-amber-500/30 text-amber-400">
														primary
													</Badge>
												{/if}
											</div>

											<div class="space-y-1">
												<div class="flex items-center justify-between text-xs">
													<span class="text-muted-foreground">Status</span>
													<span class="font-mono {instance.status === 'running' ? 'text-emerald-400' : 'text-zinc-400'}">
														{instance.status}
													</span>
												</div>
												<div class="flex items-center justify-between text-xs">
													<span class="text-muted-foreground">Inflight</span>
													<span class="font-mono {instance.inflight > 0 ? 'text-amber-400' : 'text-foreground'}">
														{instance.inflight}
													</span>
												</div>
												<div class="flex items-center justify-between text-xs">
													<span class="text-muted-foreground">Port</span>
													<span class="font-mono text-foreground">{instance.internalPort}</span>
												</div>
												<div class="flex items-center justify-between text-xs">
													<span class="text-muted-foreground">Last active</span>
													<span class="font-mono text-foreground">{relativeTime(instance.lastActivity)}</span>
												</div>
											</div>
										</div>
									{/each}
								</div>
							</Card.Content>
						</Card.Root>
					{/if}
				{/each}
			</div>
		</section>
	{/if}

	<!-- Model Cards -->
	<section>
		<h2 class="mb-3 text-sm font-medium uppercase tracking-wider text-muted-foreground">
			Models
		</h2>
		{#if models.length === 0 && !loading}
			<Card.Root class="border-dashed border-border/50">
				<Card.Content class="flex flex-col items-center justify-center py-12 text-center">
					<HardDrivesIcon class="size-10 text-muted-foreground/40 mb-3" />
					<p class="text-sm text-muted-foreground">No models loaded</p>
					<p class="text-xs text-muted-foreground/60 mt-1">
						Models will appear here when the proxy receives requests
					</p>
				</Card.Content>
			</Card.Root>
		{:else}
			<div class="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3" use:autoAnimate>
			{#each standaloneModels as model (model.name)}
				{@const actionState = modelActions[model.name]}
				<Card.Root class="overflow-hidden border-border/50 bg-card">
					<Card.Header class="pb-2">
						<div class="flex items-center justify-between">
							<div class="flex items-center gap-2 min-w-0">
								<div
									class="flex size-7 shrink-0 items-center justify-center rounded-md bg-secondary"
								>
									<HardDrivesIcon class="size-3.5 text-muted-foreground" />
								</div>
								<Card.Title class="truncate text-sm">{model.name}</Card.Title>
							</div>
							<Badge
								variant="outline"
								class="{statusColors[model.status] ?? statusColors.stopped} shrink-0"
							>
								{#if model.status === 'starting'}
									<SpinnerIcon class="size-3 animate-spin mr-1" />
								{/if}
								{#if model.status === 'running'}
									<CircleIcon class="size-3 mr-1" />
								{/if}
								{model.status}
							</Badge>
						</div>
					</Card.Header>
					<Card.Content class="space-y-2.5">
						<!-- Backend -->
						<div class="flex items-center justify-between text-xs">
							<span class="text-muted-foreground">Backend</span>
							<Badge variant="secondary" class="text-xs font-mono">{model.backend}</Badge>
						</div>

						<!-- Model path (truncated) -->
						<div class="flex items-center justify-between text-xs">
							<span class="text-muted-foreground">Path</span>
							<span class="max-w-[60%] truncate font-mono text-foreground/80" title={model.modelPath}>
								{model.modelPath}
							</span>
						</div>

						<!-- Last activity -->
						<div class="flex items-center justify-between text-xs">
							<span class="text-muted-foreground flex items-center gap-1">
								<ClockIcon class="size-3" />
								Last activity
							</span>
							<span class="font-mono text-foreground/80">
								{relativeTime(model.lastActivity)}
							</span>
						</div>

						<!-- Aliases -->
						{#if model.aliases.length > 0}
							<div class="border-t border-border/50 pt-2">
								<div class="text-xs text-muted-foreground mb-1">Aliases</div>
								<div class="flex flex-wrap gap-1">
									{#each model.aliases as alias}
										<Badge variant="outline" class="text-[10px] font-mono">{alias}</Badge>
									{/each}
								</div>
							</div>
						{/if}

						<!-- Download status -->
						{#if model.downloading}
							<div class="border-t border-border/50 pt-2">
								<div class="flex items-center gap-2 text-xs text-amber-400">
									<SpinnerIcon class="size-3 animate-spin" />
									Downloading...
								</div>
							</div>
						{/if}

						{#if model.downloadError}
							<div class="border-t border-border/50 pt-2">
								<div class="text-xs text-red-400 break-all">
									Download error: {model.downloadError}
								</div>
							</div>
						{/if}

						<!-- Model Actions -->
						<div class="border-t border-border/50 pt-2.5 flex items-center gap-1.5">
							{#if model.status === 'running' || model.status === 'starting'}
								<Button
									variant="destructive"
									size="sm"
									class="h-7 text-xs"
									disabled={!!actionState}
									onclick={() => modelAction(model.name, 'stop')}
								>
									{#if actionState === 'stopping'}
										<SpinnerIcon class="size-3 animate-spin" />
									{:else}
										<StopIcon class="size-3" />
									{/if}
									Stop
								</Button>
							{:else}
								<Button
									variant="outline"
									size="sm"
									class="h-7 text-xs"
									disabled={!!actionState}
									onclick={() => modelAction(model.name, 'start')}
								>
									{#if actionState === 'starting'}
										<SpinnerIcon class="size-3 animate-spin" />
									{:else}
										<PlayIcon class="size-3" />
									{/if}
									Start
								</Button>
							{/if}
							<Button
								variant="outline"
								size="sm"
								class="h-7 text-xs"
								disabled={!!actionState}
								onclick={() => modelAction(model.name, 'restart')}
							>
								{#if actionState === 'restarting'}
									<SpinnerIcon class="size-3 animate-spin" />
								{:else}
									<ArrowClockwiseIcon class="size-3" />
								{/if}
								Restart
							</Button>
						</div>
					</Card.Content>
				</Card.Root>
			{/each}
			</div>
		{/if}
	</section>
</div>
