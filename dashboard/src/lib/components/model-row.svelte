<script lang="ts">
	import { slide } from 'svelte/transition';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { relativeTime } from '$lib/format';
	import { isPending, isResident, pendingAction, runModelAction, statusMeta } from '$lib/models.svelte';
	import type { ModelStatus } from '$lib/types';
	import CaretRightIcon from '~icons/ph/caret-right';
	import PlayIcon from '~icons/ph/play';
	import SpinnerIcon from '~icons/ph/spinner';
	import StopIcon from '~icons/ph/stop';
	import ArrowClockwiseIcon from '~icons/ph/arrow-clockwise';

	let { model, now }: { model: ModelStatus; now: number } = $props();

	let expanded = $state(false);

	const meta = $derived(statusMeta(model.status));
	const pending = $derived(pendingAction(model.name));
	const busy = $derived(isPending(model.name));
	const instances = $derived(model.balance?.instances ?? []);
	const inflight = $derived(instances.reduce((sum, i) => sum + i.inflight, 0));
	const gpuLabel = $derived(
		instances.length
			? instances.map((i) => i.gpu).join(', ')
			: model.gpus?.length
				? model.gpus.join(', ')
				: model.gpu != null
					? String(model.gpu)
					: '—'
	);
</script>

<div class="transition-colors hover:bg-muted/40" class:opacity-55={model.status === 'stopped'}>
	<div
		class="grid grid-cols-[1.25rem_1fr_5.5rem_4rem_5rem_5.5rem_4.5rem] items-center gap-x-3 px-3 py-2.5"
	>
		{#if instances.length}
			<button
				class="flex size-5 items-center justify-center rounded text-muted-foreground hover:bg-accent hover:text-foreground"
				aria-expanded={expanded}
				aria-label={expanded ? 'Hide instances' : 'Show instances'}
				onclick={() => (expanded = !expanded)}
			>
				<CaretRightIcon class="size-3 transition-transform {expanded ? 'rotate-90' : ''}" />
			</button>
		{:else}
			<span class="flex size-5 items-center justify-center">
				<span class="size-2 rounded-full {meta.dot}"></span>
			</span>
		{/if}

		<div class="min-w-0">
			<div class="flex items-center gap-2">
				{#if instances.length}
					<span class="size-2 shrink-0 rounded-full {meta.dot}"></span>
				{/if}
				<span class="truncate text-sm font-medium">{model.name}</span>
				{#if model.downloading}
					<Tooltip.Root>
						<Tooltip.Trigger>
							{#snippet child({ props })}
								<SpinnerIcon {...props} class="size-3 shrink-0 animate-spin text-muted-foreground" />
							{/snippet}
						</Tooltip.Trigger>
						<Tooltip.Content>Downloading weights</Tooltip.Content>
					</Tooltip.Root>
				{/if}
				{#if inflight > 0}
					<span
						class="shrink-0 rounded-full bg-primary/15 px-1.5 text-[10px] font-medium tabular-nums text-primary"
					>
						{inflight} in flight
					</span>
				{/if}
			</div>
			{#if model.aliases.length}
				<div class="truncate text-[11px] text-muted-foreground">{model.aliases.join(', ')}</div>
			{/if}
			{#if model.downloadError}
				<div class="truncate text-[11px] text-destructive">{model.downloadError}</div>
			{/if}
		</div>

		<div class="truncate text-xs text-muted-foreground">{model.backend}</div>
		<div class="text-xs tabular-nums text-muted-foreground">{gpuLabel}</div>
		<div class="text-xs {meta.text}">{meta.label}</div>
		<div class="text-right text-xs tabular-nums text-muted-foreground">
			{relativeTime(model.lastActivity, now)}
		</div>

		<div class="flex items-center justify-end gap-0.5">
			{#if isResident(model.status)}
				<Tooltip.Root>
					<Tooltip.Trigger>
						{#snippet child({ props })}
							<Button
								{...props}
								variant="ghost"
								size="icon"
								class="size-7 text-muted-foreground hover:text-destructive"
								disabled={busy}
								onclick={() => runModelAction(model.name, 'stop')}
							>
								{#if pending === 'stop'}
									<SpinnerIcon class="size-3.5 animate-spin" />
								{:else}
									<StopIcon class="size-3.5" />
								{/if}
								<span class="sr-only">Stop {model.name}</span>
							</Button>
						{/snippet}
					</Tooltip.Trigger>
					<Tooltip.Content>Stop</Tooltip.Content>
				</Tooltip.Root>
			{:else}
				<Tooltip.Root>
					<Tooltip.Trigger>
						{#snippet child({ props })}
							<Button
								{...props}
								variant="ghost"
								size="icon"
								class="size-7 text-muted-foreground hover:text-foreground"
								disabled={busy}
								onclick={() => runModelAction(model.name, 'start')}
							>
								{#if pending === 'start'}
									<SpinnerIcon class="size-3.5 animate-spin" />
								{:else}
									<PlayIcon class="size-3.5" />
								{/if}
								<span class="sr-only">Start {model.name}</span>
							</Button>
						{/snippet}
					</Tooltip.Trigger>
					<Tooltip.Content>Start</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<Button
							{...props}
							variant="ghost"
							size="icon"
							class="size-7 text-muted-foreground hover:text-foreground"
							disabled={busy}
							onclick={() => runModelAction(model.name, 'restart')}
						>
							{#if pending === 'restart'}
								<SpinnerIcon class="size-3.5 animate-spin" />
							{:else}
								<ArrowClockwiseIcon class="size-3.5" />
							{/if}
							<span class="sr-only">Restart {model.name}</span>
						</Button>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>Restart with fresh config</Tooltip.Content>
			</Tooltip.Root>
		</div>
	</div>

	{#if expanded && instances.length}
		<div transition:slide={{ duration: 150 }} class="border-t border-border bg-muted/30 px-3 py-2">
			<div class="grid gap-1">
				{#each instances as instance (instance.gpu)}
					{@const instanceMeta = statusMeta(instance.status)}
					<div class="grid grid-cols-[1.25rem_1fr_5rem_6rem_5.5rem] items-center gap-x-3 text-xs">
						<span class="flex size-5 items-center justify-center">
							<span class="size-1.5 rounded-full {instanceMeta.dot}"></span>
						</span>
						<span class="tabular-nums">GPU {instance.gpu}</span>
						<span class={instanceMeta.text}>{instanceMeta.label}</span>
						<span class="tabular-nums text-muted-foreground">
							{instance.inflight} in flight
						</span>
						<span class="text-right tabular-nums text-muted-foreground">
							:{instance.internalPort}
						</span>
					</div>
				{/each}
			</div>
			<p class="mt-2 text-[11px] text-muted-foreground">
				Route with <code class="rounded bg-muted px-1">{model.name}:balance</code> or pin to
				<code class="rounded bg-muted px-1">{model.name}:gpu{instances[0]?.gpu ?? 0}</code>
			</p>
		</div>
	{/if}
</div>
