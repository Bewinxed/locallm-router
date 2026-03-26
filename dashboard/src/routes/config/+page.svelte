<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import autoAnimate from '@formkit/auto-animate';
	import { toast } from 'svelte-sonner';

	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Badge } from '$lib/components/ui/badge';
	import { Separator } from '$lib/components/ui/separator';
	import * as Card from '$lib/components/ui/card';
	import * as Select from '$lib/components/ui/select';
	import * as Collapsible from '$lib/components/ui/collapsible';

	import FloppyDiskIcon from '~icons/ph/floppy-disk';
	import StopIcon from '~icons/ph/stop';
	import ArrowClockwiseIcon from '~icons/ph/arrow-clockwise';
	import CaretDownIcon from '~icons/ph/caret-down';
	import CpuIcon from '~icons/ph/cpu';
	import HardDrivesIcon from '~icons/ph/hard-drives';
	import ClockIcon from '~icons/ph/clock';
	import SpinnerIcon from '~icons/ph/spinner';
	import PlayIcon from '~icons/ph/play';
	import CircleIcon from '~icons/ph/circle';

	interface ModelConfig {
		name: string;
		modelPath: string;
		backend?: 'vllm' | 'llama-server' | 'sglang';
		gpu: number;
		gpus?: number[];
		idleTimeout: number;
		extraArgs?: string[];
		dtype?: string;
		parameterCount?: number;
		activatedParameterCount?: number;
		vramGB?: number;
		aliases?: string[];
		balanceGpus?: number[];
		balanceGpuArgs?: Record<string, string[]>;
		startupTimeout?: number;
	}

	interface ManagerConfig {
		models: ModelConfig[];
		port: number;
		managementPort: number;
		healthCheckInterval: number;
		startupTimeout: number;
	}

	interface ModelStatus {
		name: string;
		status: string;
		backend: string;
		downloading: boolean;
	}

	let config = $state<ManagerConfig | null>(null);
	let loading = $state(true);
	let saving = $state(false);
	let stopping = $state(false);
	let openCards: Record<number, boolean> = $state({});
	let error = $state<string | null>(null);

	/** Per-model action state: tracks which models are being restarted/stopped */
	let modelActions: Record<string, 'restarting' | 'stopping' | 'starting' | null> = $state({});
	/** Live model statuses from the management API */
	let modelStatuses: Record<string, ModelStatus> = $state({});

	let statusInterval: ReturnType<typeof setInterval> | null = null;

	const backendOptions = [
		{ value: 'vllm', label: 'vLLM' },
		{ value: 'llama-server', label: 'llama-server' },
		{ value: 'sglang', label: 'SGLang' }
	];

	function statusColor(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (status) {
			case 'running':
				return 'default';
			case 'starting':
				return 'secondary';
			case 'stopping':
				return 'destructive';
			default:
				return 'outline';
		}
	}

	function statusDotClass(status: string): string {
		switch (status) {
			case 'running':
				return 'text-emerald-500';
			case 'starting':
				return 'text-amber-500 animate-pulse';
			case 'stopping':
				return 'text-red-500 animate-pulse';
			default:
				return 'text-zinc-500';
		}
	}

	async function fetchStatus() {
		try {
			const res = await fetch('/api/status');
			if (!res.ok) return;
			const data = await res.json();
			if (data.models) {
				const newStatuses: Record<string, ModelStatus> = {};
				for (const m of data.models) {
					newStatuses[m.name] = m;
				}
				modelStatuses = newStatuses;
			}
		} catch {
			// Silently ignore status poll failures
		}
	}

	async function fetchConfig() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/config');
			if (!res.ok) throw new Error(`Failed to load config: ${res.statusText}`);
			config = await res.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load config';
			toast.error(error);
		} finally {
			loading = false;
		}
	}

	async function saveConfig() {
		if (!config) return;
		saving = true;
		try {
			const res = await fetch('/api/config', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(config)
			});
			if (!res.ok) throw new Error(`Failed to save: ${res.statusText}`);
			toast.success('Configuration saved to disk');
		} catch (e) {
			const msg = e instanceof Error ? e.message : 'Failed to save config';
			toast.error(msg);
		} finally {
			saving = false;
		}
	}

	async function stopAll() {
		stopping = true;
		try {
			const res = await fetch('/api/stop-all', { method: 'POST' });
			if (!res.ok) throw new Error(`Failed to stop: ${res.statusText}`);
			const data = await res.json();
			toast.success(data.status || 'All models stopped');
			await fetchStatus();
		} catch (e) {
			const msg = e instanceof Error ? e.message : 'Failed to stop models';
			toast.error(msg);
		} finally {
			stopping = false;
		}
	}

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
			await fetchStatus();
		} catch (e) {
			const msg = e instanceof Error ? e.message : `Failed to ${action} ${modelName}`;
			toast.error(msg);
		} finally {
			modelActions[modelName] = null;
		}
	}

	async function saveAndRestart(modelName: string) {
		await saveConfig();
		await modelAction(modelName, 'restart');
	}

	function extraArgsToText(args?: string[]): string {
		if (!args || args.length === 0) return '';
		const lines: string[] = [];
		for (let i = 0; i < args.length; i++) {
			const arg = args[i];
			if (arg.startsWith('-') && i + 1 < args.length && !args[i + 1].startsWith('-')) {
				lines.push(`${arg} ${args[i + 1]}`);
				i++;
			} else {
				lines.push(arg);
			}
		}
		return lines.join('\n');
	}

	function textToExtraArgs(text: string): string[] {
		if (!text.trim()) return [];
		const args: string[] = [];
		for (const line of text.split('\n')) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			const parts = trimmed.split(/\s+/);
			for (const part of parts) {
				args.push(part);
			}
		}
		return args;
	}

	function idleTimeoutMinutes(ms: number): number {
		return Math.round(ms / 60000);
	}

	function minutesToMs(minutes: number): number {
		return minutes * 60000;
	}

	onMount(() => {
		fetchConfig();
		fetchStatus();
		statusInterval = setInterval(fetchStatus, 3000);
	});

	onDestroy(() => {
		if (statusInterval) clearInterval(statusInterval);
	});
</script>

<svelte:head>
	<title>Configuration - locallm-router</title>
</svelte:head>

<div class="mx-auto max-w-4xl space-y-6 p-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold tracking-tight">Model Configuration</h1>
			<p class="text-muted-foreground text-sm">
				Edit model parameters, save, then restart individual models to apply.
			</p>
		</div>
		<div class="flex items-center gap-2">
			<Button variant="destructive" size="sm" onclick={stopAll} disabled={stopping}>
				{#if stopping}
					<SpinnerIcon class="size-4 animate-spin" />
				{:else}
					<StopIcon class="size-4" />
				{/if}
				Stop All
			</Button>
			<Button size="sm" onclick={saveConfig} disabled={saving || !config}>
				{#if saving}
					<SpinnerIcon class="size-4 animate-spin" />
				{:else}
					<FloppyDiskIcon class="size-4" />
				{/if}
				Save
			</Button>
		</div>
	</div>

	<Separator />

	<!-- Loading / Error -->
	{#if loading}
		<div class="flex items-center justify-center py-20">
			<SpinnerIcon class="text-muted-foreground size-6 animate-spin" />
			<span class="text-muted-foreground ml-2 text-sm">Loading configuration...</span>
		</div>
	{:else if error}
		<Card.Root>
			<Card.Content class="py-10 text-center">
				<p class="text-destructive text-sm">{error}</p>
				<Button variant="outline" size="sm" class="mt-4" onclick={fetchConfig}>
					Retry
				</Button>
			</Card.Content>
		</Card.Root>
	{:else if config}
		<!-- Global Settings -->
		<Card.Root>
			<Card.Header>
				<Card.Title class="text-base">Global Settings</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
					<div class="space-y-1.5">
						<Label>Proxy Port</Label>
						<Input type="number" bind:value={config.port} />
					</div>
					<div class="space-y-1.5">
						<Label>Management Port</Label>
						<Input type="number" bind:value={config.managementPort} />
					</div>
					<div class="space-y-1.5">
						<Label>Health Check (ms)</Label>
						<Input type="number" bind:value={config.healthCheckInterval} />
					</div>
					<div class="space-y-1.5">
						<Label>Startup Timeout (ms)</Label>
						<Input type="number" bind:value={config.startupTimeout} />
					</div>
				</div>
			</Card.Content>
		</Card.Root>

		<!-- Model Cards -->
		<div class="space-y-3" use:autoAnimate>
			{#each config.models as model, i (model.name + i)}
				{@const isOpen = openCards[i] ?? false}
				{@const status = modelStatuses[model.name]}
				{@const modelStatus = status?.status ?? 'stopped'}
				{@const actionState = modelActions[model.name]}
				<Collapsible.Root
					open={isOpen}
					onOpenChange={(open) => {
						openCards[i] = open;
					}}
				>
					<Card.Root class={modelStatus === 'running' ? 'border-emerald-500/30' : ''}>
						<Collapsible.Trigger class="w-full">
							<Card.Header class="cursor-pointer select-none">
								<div class="flex items-center justify-between">
									<div class="flex items-center gap-3">
										<HardDrivesIcon class="text-muted-foreground size-4" />
										<div class="text-left">
											<Card.Title class="text-sm font-semibold">
												{model.name}
											</Card.Title>
											<Card.Description class="text-xs">
												{model.modelPath}
											</Card.Description>
										</div>
									</div>
									<div class="flex items-center gap-2">
										<!-- Status badge -->
										<Badge variant={statusColor(modelStatus)}>
											<CircleIcon
												class="mr-1 size-2 {statusDotClass(modelStatus)}"
											/>
											{modelStatus}
										</Badge>
										<Badge variant="secondary">
											{model.backend || 'vllm'}
										</Badge>
										<Badge variant="outline">
											<CpuIcon class="mr-1 size-3" />
											GPU {model.gpu}
										</Badge>
										{#if model.balanceGpus?.length}
											<Badge
												variant="outline"
												class="text-chart-1 border-chart-1/30"
											>
												Balance [{model.balanceGpus.join(', ')}]
											</Badge>
										{/if}
										<CaretDownIcon
											class="text-muted-foreground size-4 transition-transform {isOpen
												? 'rotate-180'
												: ''}"
										/>
									</div>
								</div>
							</Card.Header>
						</Collapsible.Trigger>

						<Collapsible.Content>
							<Card.Content class="space-y-4 border-t pt-4">
								<!-- Model Actions Row -->
								<div
									class="bg-muted/50 flex items-center justify-between rounded-lg px-4 py-3"
								>
									<div class="text-sm">
										<span class="text-muted-foreground">Status:</span>
										<span class="ml-1 font-medium">{modelStatus}</span>
										{#if status?.downloading}
											<Badge variant="secondary" class="ml-2">
												<SpinnerIcon class="mr-1 size-3 animate-spin" />
												Downloading
											</Badge>
										{/if}
									</div>
									<div class="flex items-center gap-2">
										{#if modelStatus === 'running' || modelStatus === 'starting'}
											<Button
												variant="destructive"
												size="sm"
												disabled={!!actionState}
												onclick={(e) => {
													e.stopPropagation();
													modelAction(model.name, 'stop');
												}}
											>
												{#if actionState === 'stopping'}
													<SpinnerIcon class="size-3.5 animate-spin" />
												{:else}
													<StopIcon class="size-3.5" />
												{/if}
												Stop
											</Button>
										{:else}
											<Button
												variant="outline"
												size="sm"
												disabled={!!actionState}
												onclick={(e) => {
													e.stopPropagation();
													modelAction(model.name, 'start');
												}}
											>
												{#if actionState === 'starting'}
													<SpinnerIcon class="size-3.5 animate-spin" />
												{:else}
													<PlayIcon class="size-3.5" />
												{/if}
												Start
											</Button>
										{/if}
										<Button
											variant="outline"
											size="sm"
											disabled={!!actionState}
											onclick={(e) => {
												e.stopPropagation();
												saveAndRestart(model.name);
											}}
										>
											{#if actionState === 'restarting'}
												<SpinnerIcon class="size-3.5 animate-spin" />
											{:else}
												<ArrowClockwiseIcon class="size-3.5" />
											{/if}
											Save & Restart
										</Button>
									</div>
								</div>

								<!-- Row 1: Name + Backend -->
								<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
									<div class="space-y-1.5">
										<Label for="name-{i}">Name</Label>
										<Input id="name-{i}" bind:value={model.name} />
									</div>
									<div class="space-y-1.5">
										<Label>Backend</Label>
										<Select.Root
											type="single"
											value={model.backend || 'vllm'}
											onValueChange={(v) => {
												if (
													v === 'vllm' ||
													v === 'llama-server' ||
													v === 'sglang'
												) {
													model.backend = v === 'vllm' ? undefined : v;
												}
											}}
										>
											<Select.Trigger class="w-full">
												{backendOptions.find(
													(o) => o.value === (model.backend || 'vllm')
												)?.label ?? 'Select backend'}
											</Select.Trigger>
											<Select.Content>
												{#each backendOptions as opt}
													<Select.Item value={opt.value} label={opt.label} />
												{/each}
											</Select.Content>
										</Select.Root>
									</div>
								</div>

								<!-- Row 2: Model Path -->
								<div class="space-y-1.5">
									<Label for="path-{i}">Model Path</Label>
									<Input
										id="path-{i}"
										bind:value={model.modelPath}
										placeholder="huggingface/repo-id or /path/to/model.gguf"
									/>
								</div>

								<!-- Row 3: GPU, Idle Timeout, Dtype -->
								<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
									<div class="space-y-1.5">
										<Label for="gpu-{i}">
											<CpuIcon class="mr-1 inline size-3" />
											GPU
										</Label>
										<Input
											id="gpu-{i}"
											type="number"
											min="0"
											bind:value={model.gpu}
										/>
									</div>
									<div class="space-y-1.5">
										<Label for="idle-{i}">
											<ClockIcon class="mr-1 inline size-3" />
											Idle Timeout (min)
										</Label>
										<Input
											id="idle-{i}"
											type="number"
											min="1"
											value={idleTimeoutMinutes(model.idleTimeout)}
											oninput={(e) => {
												const val = parseInt(e.currentTarget.value);
												if (!isNaN(val) && val > 0)
													model.idleTimeout = minutesToMs(val);
											}}
										/>
									</div>
									<div class="space-y-1.5">
										<Label for="dtype-{i}">Dtype</Label>
										<Input
											id="dtype-{i}"
											bind:value={model.dtype}
											placeholder="bfloat16"
										/>
									</div>
								</div>

								<!-- Row 4: Param Count, Activated Params, VRAM -->
								<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
									<div class="space-y-1.5">
										<Label for="params-{i}">Params (B)</Label>
										<Input
											id="params-{i}"
											type="number"
											step="0.1"
											bind:value={model.parameterCount}
											placeholder="e.g. 7"
										/>
									</div>
									<div class="space-y-1.5">
										<Label for="aparams-{i}">Activated Params (B)</Label>
										<Input
											id="aparams-{i}"
											type="number"
											step="0.1"
											bind:value={model.activatedParameterCount}
											placeholder="MoE only"
										/>
									</div>
									<div class="space-y-1.5">
										<Label for="vram-{i}">VRAM Override (GB)</Label>
										<Input
											id="vram-{i}"
											type="number"
											step="0.1"
											bind:value={model.vramGB}
											placeholder="Auto"
										/>
									</div>
								</div>

								<!-- Balance GPUs -->
								{#if model.balanceGpus?.length}
									<div class="space-y-1.5">
										<Label>Balance GPUs</Label>
										<div class="flex flex-wrap gap-1.5">
											{#each model.balanceGpus as gpuIdx}
												<Badge variant="outline">GPU {gpuIdx}</Badge>
											{/each}
										</div>
										<p class="text-muted-foreground text-xs">
											Requests can use <code class="bg-muted rounded px-1"
												>{model.name}:balance</code
											>
											for auto-routing
											or
											<code class="bg-muted rounded px-1"
												>{model.name}:gpu{'{N}'}</code
											> to pin.
										</p>
									</div>
								{/if}

								<!-- Multi-GPU -->
								{#if model.gpus?.length}
									<div class="space-y-1.5">
										<Label>Multi-GPU (Tensor Parallel)</Label>
										<div class="flex flex-wrap gap-1.5">
											{#each model.gpus as gpuIdx}
												<Badge variant="secondary">GPU {gpuIdx}</Badge>
											{/each}
										</div>
									</div>
								{/if}

								<!-- Aliases -->
								{#if model.aliases?.length}
									<div class="space-y-1.5">
										<Label>Aliases</Label>
										<div class="flex flex-wrap gap-1.5">
											{#each model.aliases as alias}
												<Badge variant="secondary">{alias}</Badge>
											{/each}
										</div>
									</div>
								{/if}

								<!-- Extra Args -->
								<div class="space-y-1.5">
									<Label for="args-{i}">Extra Args (one flag pair per line)</Label>
									<Textarea
										id="args-{i}"
										rows={Math.max(3, (model.extraArgs?.length ?? 0) / 2 + 1)}
										class="font-mono text-xs"
										value={extraArgsToText(model.extraArgs)}
										oninput={(e) => {
											model.extraArgs = textToExtraArgs(e.currentTarget.value);
										}}
										placeholder="--gpu-memory-utilization 0.85&#10;--max-model-len 8192"
									/>
								</div>
							</Card.Content>
						</Collapsible.Content>
					</Card.Root>
				</Collapsible.Root>
			{/each}
		</div>

		<!-- Bottom actions -->
		<div class="flex justify-end gap-2 pt-2">
			<Button onclick={saveConfig} disabled={saving}>
				{#if saving}
					<SpinnerIcon class="size-4 animate-spin" />
				{:else}
					<FloppyDiskIcon class="size-4" />
				{/if}
				Save Configuration
			</Button>
		</div>
	{/if}
</div>
