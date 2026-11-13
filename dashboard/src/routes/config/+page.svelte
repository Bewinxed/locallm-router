<script lang="ts">
	import { toast } from 'svelte-sonner';
	import autoAnimate from '@formkit/auto-animate';
	import ModelConfigCard from '$lib/components/model-config-card.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Card from '$lib/components/ui/card';
	import * as Dialog from '$lib/components/ui/dialog';
	import { newModel, type ManagerConfig } from '$lib/config';
	import { runModelAction } from '$lib/models.svelte';
	import { refresh, status, useStatus } from '$lib/status.svelte';
	import FloppyDiskIcon from '~icons/ph/floppy-disk';
	import PlusIcon from '~icons/ph/plus';
	import SpinnerIcon from '~icons/ph/spinner';
	import WarningIcon from '~icons/ph/warning-circle';

	$effect(useStatus);

	let config = $state<ManagerConfig | null>(null);
	let loading = $state(true);
	let saving = $state(false);
	let loadError = $state<string | null>(null);
	let openCards = $state<boolean[]>([]);
	let removeIndex = $state<number | null>(null);

	const statusByName = $derived(
		new Map(status.models.map((m) => [m.name, m.status] as const))
	);

	async function fetchConfig() {
		loading = true;
		loadError = null;
		try {
			const res = await fetch('/api/config');
			const data = await res.json();
			if (!res.ok) throw new Error(data?.error || 'Could not load config');
			config = data;
			openCards = (config?.models ?? []).map(() => false);
		} catch (err) {
			loadError = err instanceof Error ? err.message : 'Could not load config';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		fetchConfig();
	});

	async function saveConfig(): Promise<boolean> {
		if (!config) return false;

		const names = config.models.map((m) => m.name.trim()).filter(Boolean);
		if (names.length < config.models.length) {
			toast.error('Every model needs a name');
			return false;
		}
		if (new Set(names).size !== names.length) {
			toast.error('Model names must be unique');
			return false;
		}

		saving = true;
		try {
			const res = await fetch('/api/config', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(config)
			});
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data?.error || 'Could not save config');
			toast.success('Configuration saved');
			return true;
		} catch (err) {
			toast.error(err instanceof Error ? err.message : 'Could not save config');
			return false;
		} finally {
			saving = false;
		}
	}

	async function saveAndRestart(name: string) {
		if (await saveConfig()) {
			await runModelAction(name, 'restart');
			await refresh();
		}
	}

	function addModel() {
		if (!config) return;
		config.models.push(newModel());
		openCards.push(true);
	}

	function confirmRemove() {
		if (config && removeIndex !== null) {
			config.models.splice(removeIndex, 1);
			openCards.splice(removeIndex, 1);
			removeIndex = null;
			toast.info('Model removed — save to apply');
		}
	}
</script>

<svelte:head>
	<title>Settings · locallm-router</title>
</svelte:head>

<div class="flex h-full flex-col">
	<header class="flex items-center justify-between border-b border-border px-6 py-4">
		<div class="flex items-baseline gap-3">
			<h1 class="text-sm font-semibold">Settings</h1>
			<span class="text-xs text-muted-foreground">
				Edits are saved to config.json; restart a model to apply
			</span>
		</div>
		<div class="flex items-center gap-2">
			<Button variant="outline" size="sm" class="h-7" onclick={addModel} disabled={!config}>
				<PlusIcon class="size-3.5" />
				New model
			</Button>
			<Button size="sm" class="h-7" onclick={saveConfig} disabled={saving || !config}>
				{#if saving}
					<SpinnerIcon class="size-3.5 animate-spin" />
				{:else}
					<FloppyDiskIcon class="size-3.5" />
				{/if}
				Save
			</Button>
		</div>
	</header>

	<div class="min-h-0 flex-1 overflow-auto">
		<div class="mx-auto max-w-4xl space-y-6 px-6 py-5">
			{#if loading}
				<div class="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
					<SpinnerIcon class="size-4 animate-spin" />
					Loading configuration
				</div>
			{:else if loadError}
				<div
					class="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border py-12"
				>
					<WarningIcon class="size-6 text-destructive" />
					<p class="text-sm text-destructive">{loadError}</p>
					<Button variant="outline" size="sm" onclick={fetchConfig}>Retry</Button>
				</div>
			{:else if config}
				<section class="space-y-2">
					<h2 class="text-xs font-medium uppercase tracking-wider text-muted-foreground">
						Router
					</h2>
					<Card.Root class="py-4">
						<Card.Content class="grid grid-cols-2 gap-4 sm:grid-cols-4">
							<div class="space-y-1.5">
								<Label for="proxy-port">Proxy port</Label>
								<Input id="proxy-port" type="number" bind:value={config.port} />
							</div>
							<div class="space-y-1.5">
								<Label for="mgmt-port">Management port</Label>
								<Input id="mgmt-port" type="number" bind:value={config.managementPort} />
							</div>
							<div class="space-y-1.5">
								<Label for="health-interval">Health check (ms)</Label>
								<Input id="health-interval" type="number" bind:value={config.healthCheckInterval} />
							</div>
							<div class="space-y-1.5">
								<Label for="startup-timeout">Startup timeout (ms)</Label>
								<Input id="startup-timeout" type="number" bind:value={config.startupTimeout} />
							</div>
						</Card.Content>
					</Card.Root>
				</section>

				<section class="space-y-2">
					<h2 class="text-xs font-medium uppercase tracking-wider text-muted-foreground">
						Models
					</h2>
					{#if config.models.length === 0}
						<div
							class="rounded-lg border border-dashed border-border py-12 text-center text-sm text-muted-foreground"
						>
							No models yet.
							<button class="text-foreground underline underline-offset-4" onclick={addModel}>
								Add your first model
							</button>
						</div>
					{:else}
						<div class="space-y-2" use:autoAnimate>
							{#each config.models as model, i (i)}
								<ModelConfigCard
									bind:model={config.models[i]}
									bind:open={openCards[i]}
									status={statusByName.get(model.name) ?? 'stopped'}
									onSaveRestart={() => saveAndRestart(model.name)}
									onRemove={() => (removeIndex = i)}
								/>
							{/each}
						</div>
					{/if}
				</section>
			{/if}
		</div>
	</div>
</div>

<Dialog.Root open={removeIndex !== null} onOpenChange={(open) => !open && (removeIndex = null)}>
	<Dialog.Content class="sm:max-w-sm">
		<Dialog.Header>
			<Dialog.Title>Remove model?</Dialog.Title>
			<Dialog.Description>
				{#if removeIndex !== null && config}
					<span class="font-medium text-foreground">
						{config.models[removeIndex]?.name || 'This model'}
					</span>
					will be removed from the configuration. Weights on disk are not deleted. This takes
					effect when you save.
				{/if}
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" size="sm" onclick={() => (removeIndex = null)}>Cancel</Button>
			<Button variant="destructive" size="sm" onclick={confirmRemove}>Remove</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
