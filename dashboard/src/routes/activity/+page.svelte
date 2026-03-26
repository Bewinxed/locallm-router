<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import autoAnimate from '@formkit/auto-animate';

	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import * as Card from '$lib/components/ui/card';

	import PulseIcon from '~icons/ph/pulse';
	import ArrowsClockwiseIcon from '~icons/ph/arrows-clockwise';
	import HardDrivesIcon from '~icons/ph/hard-drives';
	import CpuIcon from '~icons/ph/cpu';
	import CircleIcon from '~icons/ph/circle';
	import ClockIcon from '~icons/ph/clock';
	import SpinnerIcon from '~icons/ph/spinner';
	import PauseIcon from '~icons/ph/pause';
	import PlayIcon from '~icons/ph/play';

	interface ModelStatus {
		name: string;
		backend: string;
		modelPath: string;
		status: 'stopped' | 'starting' | 'running' | 'stopping';
		lastActivity: number;
		downloading: boolean;
		aliases: string[];
		balance: BalanceGroupStatus | null;
	}

	interface BalanceInstance {
		gpu: number;
		status: string;
		inflight: number;
		internalPort: number;
		lastActivity: number;
	}

	interface BalanceGroupStatus {
		baseName: string;
		primaryGpu: number;
		instances: BalanceInstance[];
	}

	interface GpuStatus {
		gpu: number;
		locked: boolean;
		owner: string | null;
	}

	interface StatusResponse {
		models: ModelStatus[];
		gpus: GpuStatus[];
		balanceGroups: BalanceGroupStatus[];
	}

	let status = $state<StatusResponse | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let polling = $state(true);
	let lastUpdated = $state<Date | null>(null);
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const statusVariant = $derived((s: string) => {
		switch (s) {
			case 'running':
				return 'default' as const;
			case 'starting':
				return 'secondary' as const;
			case 'stopping':
				return 'secondary' as const;
			case 'stopped':
				return 'outline' as const;
			default:
				return 'outline' as const;
		}
	});

	function formatTimeAgo(timestamp: number): string {
		if (!timestamp) return 'Never';
		const seconds = Math.floor((Date.now() - timestamp) / 1000);
		if (seconds < 5) return 'Just now';
		if (seconds < 60) return `${seconds}s ago`;
		const minutes = Math.floor(seconds / 60);
		if (minutes < 60) return `${minutes}m ago`;
		const hours = Math.floor(minutes / 60);
		if (hours < 24) return `${hours}h ${minutes % 60}m ago`;
		return `${Math.floor(hours / 24)}d ago`;
	}

	function formatTimestamp(timestamp: number): string {
		if (!timestamp) return '-';
		return new Date(timestamp).toLocaleTimeString();
	}

	async function fetchStatus() {
		try {
			const res = await fetch('/api/status');
			if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
			status = await res.json();
			lastUpdated = new Date();
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fetch status';
		} finally {
			loading = false;
		}
	}

	function startPolling() {
		polling = true;
		pollTimer = setInterval(fetchStatus, 2000);
	}

	function stopPolling() {
		polling = false;
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	function togglePolling() {
		if (polling) {
			stopPolling();
		} else {
			startPolling();
		}
	}

	let runningCount = $derived(status?.models.filter((m) => m.status === 'running').length ?? 0);
	let totalCount = $derived(status?.models.length ?? 0);

	onMount(() => {
		fetchStatus();
		startPolling();
	});

	onDestroy(() => {
		stopPolling();
	});
</script>

<svelte:head>
	<title>Activity - locallm-router</title>
</svelte:head>

<div class="mx-auto max-w-5xl space-y-6 p-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold tracking-tight">Activity Monitor</h1>
			<p class="text-muted-foreground text-sm">
				Live model status and activity.
				{#if lastUpdated}
					Last updated {formatTimestamp(lastUpdated.getTime())}.
				{/if}
			</p>
		</div>
		<div class="flex items-center gap-2">
			<Badge variant={runningCount > 0 ? 'default' : 'outline'}>
				{runningCount}/{totalCount} running
			</Badge>
			<Button variant="outline" size="sm" onclick={togglePolling}>
				{#if polling}
					<PauseIcon class="size-4" />
					Pause
				{:else}
					<PlayIcon class="size-4" />
					Resume
				{/if}
			</Button>
			<Button variant="outline" size="icon-sm" onclick={fetchStatus}>
				<ArrowsClockwiseIcon class="size-4" />
			</Button>
		</div>
	</div>

	<Separator />

	{#if loading}
		<div class="flex items-center justify-center py-20">
			<SpinnerIcon class="text-muted-foreground size-6 animate-spin" />
			<span class="text-muted-foreground ml-2 text-sm">Loading status...</span>
		</div>
	{:else if error && !status}
		<Card.Root>
			<Card.Content class="py-10 text-center">
				<p class="text-destructive text-sm">{error}</p>
				<Button variant="outline" size="sm" class="mt-4" onclick={fetchStatus}>Retry</Button>
			</Card.Content>
		</Card.Root>
	{:else if status}
		<!-- Model Status Table -->
		<Card.Root>
			<Card.Header>
				<Card.Title class="flex items-center gap-2 text-base">
					<PulseIcon class="size-4" />
					Models
				</Card.Title>
			</Card.Header>
			<Card.Content class="p-0">
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b">
								<th class="text-muted-foreground px-4 py-2.5 text-left text-xs font-medium">
									Model
								</th>
								<th class="text-muted-foreground px-4 py-2.5 text-left text-xs font-medium">
									Backend
								</th>
								<th class="text-muted-foreground px-4 py-2.5 text-left text-xs font-medium">
									Status
								</th>
								<th class="text-muted-foreground px-4 py-2.5 text-left text-xs font-medium">
									Last Activity
								</th>
								<th class="text-muted-foreground px-4 py-2.5 text-right text-xs font-medium">
									Inflight
								</th>
							</tr>
						</thead>
						<tbody use:autoAnimate>
							{#each status.models as model (model.name)}
								{@const totalInflight =
									model.balance?.instances.reduce((sum, inst) => sum + inst.inflight, 0) ?? 0}
								<tr class="border-b last:border-b-0 transition-colors hover:bg-muted/50">
									<td class="px-4 py-2.5">
										<div class="flex items-center gap-2">
											<HardDrivesIcon class="text-muted-foreground size-3.5" />
											<span class="font-medium">{model.name}</span>
											{#if model.downloading}
												<Badge variant="secondary" class="text-xs">
													<SpinnerIcon class="mr-1 size-3 animate-spin" />
													downloading
												</Badge>
											{/if}
										</div>
										{#if model.aliases.length > 0}
											<div class="text-muted-foreground mt-0.5 text-xs">
												aka {model.aliases.join(', ')}
											</div>
										{/if}
									</td>
									<td class="text-muted-foreground px-4 py-2.5 text-xs">
										{model.backend}
									</td>
									<td class="px-4 py-2.5">
										<Badge variant={statusVariant(model.status)}>
											{#if model.status === 'running'}
												<CircleIcon class="mr-1 size-3 text-green-500" />
											{:else if model.status === 'starting' || model.status === 'stopping'}
												<SpinnerIcon class="mr-1 size-3 animate-spin" />
											{/if}
											{model.status}
										</Badge>
									</td>
									<td class="px-4 py-2.5">
										<div class="flex items-center gap-1.5 text-xs">
											<ClockIcon class="text-muted-foreground size-3" />
											<span class="text-muted-foreground">
												{formatTimeAgo(model.lastActivity)}
											</span>
										</div>
									</td>
									<td class="px-4 py-2.5 text-right">
										{#if model.balance}
											<Badge variant={totalInflight > 0 ? 'default' : 'outline'}>
												{totalInflight}
											</Badge>
										{:else}
											<span class="text-muted-foreground text-xs">-</span>
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</Card.Content>
		</Card.Root>

		<!-- Balance Groups -->
		{#if status.balanceGroups.length > 0}
			<div class="space-y-4">
				<h2 class="flex items-center gap-2 text-lg font-semibold">
					<CpuIcon class="size-4" />
					Balance Groups
				</h2>
				{#each status.balanceGroups as group (group.baseName)}
					<Card.Root>
						<Card.Header class="pb-3">
							<div class="flex items-center justify-between">
								<Card.Title class="text-sm font-semibold">
									{group.baseName}
								</Card.Title>
								<Badge variant="outline" class="text-xs">
									Primary: GPU {group.primaryGpu}
								</Badge>
							</div>
						</Card.Header>
						<Card.Content>
							<div
								class="grid gap-3"
								style="grid-template-columns: repeat({group.instances.length}, minmax(0, 1fr));"
								use:autoAnimate
							>
								{#each group.instances as instance (instance.gpu)}
									{@const isActive = instance.status === 'running'}
									<Card.Root
										class="transition-shadow {isActive
											? 'border-green-500/30 shadow-green-500/5 shadow-sm'
											: ''}"
									>
										<Card.Content class="space-y-3 p-4">
											<div class="flex items-center justify-between">
												<div class="flex items-center gap-2">
													<CpuIcon
														class="size-4 {isActive
															? 'text-green-500'
															: 'text-muted-foreground'}"
													/>
													<span class="text-sm font-medium">GPU {instance.gpu}</span>
												</div>
												<Badge variant={statusVariant(instance.status)}>
													{#if instance.status === 'running'}
														<CircleIcon class="mr-1 size-3 text-green-500" />
													{:else if instance.status === 'starting' || instance.status === 'stopping'}
														<SpinnerIcon class="mr-1 size-3 animate-spin" />
													{/if}
													{instance.status}
												</Badge>
											</div>

											<Separator />

											<div class="grid grid-cols-2 gap-2 text-xs">
												<div>
													<span class="text-muted-foreground">Inflight</span>
													<div
														class="text-lg font-semibold {instance.inflight > 0
															? 'text-chart-1'
															: ''}"
													>
														{instance.inflight}
													</div>
												</div>
												<div>
													<span class="text-muted-foreground">Port</span>
													<div class="font-mono text-sm">{instance.internalPort}</div>
												</div>
											</div>

											<div class="text-xs">
												<span class="text-muted-foreground">Last active: </span>
												<span>{formatTimeAgo(instance.lastActivity)}</span>
											</div>
										</Card.Content>
									</Card.Root>
								{/each}
							</div>
						</Card.Content>
					</Card.Root>
				{/each}
			</div>
		{/if}

		<!-- GPU Status -->
		{#if status.gpus.length > 0}
			<Card.Root>
				<Card.Header>
					<Card.Title class="flex items-center gap-2 text-base">
						<CpuIcon class="size-4" />
						GPU Locks
					</Card.Title>
				</Card.Header>
				<Card.Content>
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4" use:autoAnimate>
						{#each status.gpus as gpu (gpu.gpu)}
							<div
								class="border-border rounded-lg border p-3 {gpu.locked
									? 'bg-muted/50'
									: ''}"
							>
								<div class="flex items-center justify-between">
									<span class="text-sm font-medium">GPU {gpu.gpu}</span>
									<Badge variant={gpu.locked ? 'default' : 'outline'} class="text-xs">
										{gpu.locked ? 'Locked' : 'Free'}
									</Badge>
								</div>
								{#if gpu.owner}
									<p class="text-muted-foreground mt-1 truncate text-xs">
										{gpu.owner}
									</p>
								{/if}
							</div>
						{/each}
					</div>
				</Card.Content>
			</Card.Root>
		{/if}

		<!-- Connection error overlay -->
		{#if error}
			<div class="bg-destructive/10 text-destructive rounded-lg p-3 text-center text-xs">
				Connection issue: {error}. Retrying...
			</div>
		{/if}
	{/if}
</div>
