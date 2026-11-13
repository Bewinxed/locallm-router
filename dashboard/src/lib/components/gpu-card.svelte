<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { formatGB, percent } from '$lib/format';
	import { modelColor } from '$lib/models.svelte';
	import type { GpuLockStatus, GpuStats } from '$lib/types';

	let { gpu, lock }: { gpu: GpuStats; lock?: GpuLockStatus } = $props();

	/** The lock reports gigabytes; everything else here is nvidia-smi megabytes. */
	const loaded = $derived(
		(lock?.loaded ?? []).map((m) => ({ name: m.name, vramMB: m.effectiveVRAM * 1024 }))
	);
	const usedPct = $derived(percent(gpu.memoryUsedMB, gpu.memoryTotalMB));
	/** VRAM held by tracked models; the remainder is driver/other processes. */
	const trackedMB = $derived(loaded.reduce((sum, m) => sum + m.vramMB, 0));
	const untrackedMB = $derived(Math.max(0, gpu.memoryUsedMB - trackedMB));
</script>

<div class="rounded-lg border border-border bg-card p-3">
	<div class="mb-2 flex items-center gap-3 text-xs">
		<span class="w-12 shrink-0 font-semibold">GPU {gpu.index}</span>
		<span class="min-w-0 flex-1 truncate text-muted-foreground">{gpu.name}</span>
		<span class="tabular-nums text-muted-foreground">{gpu.temperatureC}°C</span>
		<span class="tabular-nums text-muted-foreground">{gpu.utilizationPercent}% util</span>
		<span class="tabular-nums text-muted-foreground">{Math.round(gpu.powerDrawW)}W</span>
		<span class="w-28 text-right font-medium tabular-nums">
			{formatGB(gpu.memoryUsedMB)} / {formatGB(gpu.memoryTotalMB)} GB
		</span>
	</div>

	<div
		class="flex h-3 w-full overflow-hidden rounded-sm bg-secondary"
		role="img"
		aria-label="{usedPct}% of VRAM in use"
	>
		{#each loaded as model (model.name)}
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<div
							{...props}
							class="h-full min-w-0.5 transition-[width] duration-500 ease-out"
							style="width: {percent(model.vramMB, gpu.memoryTotalMB)}%; background-color: {modelColor(
								model.name
							)}"
						></div>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>
					{model.name} · {formatGB(model.vramMB)} GB
				</Tooltip.Content>
			</Tooltip.Root>
		{/each}

		{#if untrackedMB > 0}
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<div
							{...props}
							class="h-full bg-muted-foreground/20 transition-[width] duration-500 ease-out"
							style="width: {percent(untrackedMB, gpu.memoryTotalMB)}%"
						></div>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>
					Other processes · {formatGB(untrackedMB)} GB
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}
	</div>

	{#if loaded.length}
		<div class="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
			{#each loaded as model (model.name)}
				<span class="flex items-center gap-1.5">
					<span
						class="size-2 shrink-0 rounded-sm"
						style="background-color: {modelColor(model.name)}"
					></span>
					{model.name}
					<span class="tabular-nums">{formatGB(model.vramMB)}</span>
				</span>
			{/each}
		</div>
	{/if}
</div>
