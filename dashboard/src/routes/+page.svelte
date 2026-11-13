<script lang="ts">
	import { toast } from 'svelte-sonner';
	import GpuCard from '$lib/components/gpu-card.svelte';
	import ModelRow from '$lib/components/model-row.svelte';
	import { Button } from '$lib/components/ui/button';
	import { refresh, status, useNow, useStatus } from '$lib/status.svelte';
	import { isResident } from '$lib/models.svelte';
	import SpinnerIcon from '~icons/ph/spinner';
	import StopIcon from '~icons/ph/stop';
	import WarningIcon from '~icons/ph/warning-circle';

	const now = useNow();
	$effect(useStatus);

	let stoppingAll = $state(false);

	// Running first, then the rest by residency, so the active set stays on top.
	const ORDER = { running: 0, starting: 1, sleeping: 2, stopping: 3, stopped: 4 } as const;
	const models = $derived(
		[...status.models].sort(
			(a, b) => (ORDER[a.status] ?? 5) - (ORDER[b.status] ?? 5) || a.name.localeCompare(b.name)
		)
	);
	const residentCount = $derived(status.models.filter((m) => isResident(m.status)).length);

	async function stopAll() {
		stoppingAll = true;
		const toastId = toast.loading('Stopping all models…');
		try {
			const res = await fetch('/api/stop-all', { method: 'POST' });
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data?.error || 'Could not stop models');
			toast.success('All models stopped', { id: toastId });
			await refresh();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : 'Could not stop models', { id: toastId });
		} finally {
			stoppingAll = false;
		}
	}
</script>

<svelte:head>
	<title>Overview · locallm-router</title>
</svelte:head>

<div class="flex h-full flex-col">
	<header class="flex items-center justify-between border-b border-border px-6 py-4">
		<div class="flex items-baseline gap-3">
			<h1 class="text-sm font-semibold">Overview</h1>
			{#if !status.loading}
				<span class="text-xs text-muted-foreground">
					{residentCount} of {status.models.length} loaded
				</span>
			{/if}
		</div>

		<div class="flex items-center gap-3">
			{#if status.error}
				<span class="flex items-center gap-1.5 text-xs text-destructive">
					<WarningIcon class="size-3.5" />
					{status.error}
				</span>
			{/if}
			<Button
				variant="outline"
				size="sm"
				class="h-7"
				disabled={stoppingAll || residentCount === 0}
				onclick={stopAll}
			>
				{#if stoppingAll}
					<SpinnerIcon class="size-3.5 animate-spin" />
				{:else}
					<StopIcon class="size-3.5" />
				{/if}
				Stop all
			</Button>
		</div>
	</header>

	<div class="min-h-0 flex-1 overflow-auto">
		<div class="mx-auto max-w-6xl space-y-6 px-6 py-5">
			{#if status.gpuStats.length}
				<section class="space-y-2">
					<h2 class="text-xs font-medium uppercase tracking-wider text-muted-foreground">GPUs</h2>
					{#each status.gpuStats as gpu (gpu.index)}
						<GpuCard {gpu} lock={status.gpuLocks[String(gpu.index)]} />
					{/each}
				</section>
			{/if}

			<section class="space-y-2">
				<h2 class="text-xs font-medium uppercase tracking-wider text-muted-foreground">Models</h2>

				{#if status.loading}
					<div class="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
						<SpinnerIcon class="size-4 animate-spin" />
						Loading
					</div>
				{:else if models.length === 0}
					<div
						class="rounded-lg border border-dashed border-border py-12 text-center text-sm text-muted-foreground"
					>
						No models configured.
						<a href="/config" class="text-foreground underline underline-offset-4">
							Add one in Settings
						</a>
					</div>
				{:else}
					<div
						class="grid grid-cols-[1.25rem_1fr_5.5rem_4rem_5rem_5.5rem_4.5rem] gap-x-3 px-3 pb-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground"
					>
						<span></span>
						<span>Name</span>
						<span>Backend</span>
						<span>GPU</span>
						<span>Status</span>
						<span class="text-right">Last active</span>
						<span class="text-right">Actions</span>
					</div>
					<div class="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card">
						{#each models as model (model.name)}
							<ModelRow {model} now={now.current} />
						{/each}
					</div>
				{/if}
			</section>
		</div>
	</div>
</div>
