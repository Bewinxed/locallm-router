<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Textarea } from '$lib/components/ui/textarea';
	import * as Card from '$lib/components/ui/card';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Select from '$lib/components/ui/select';
	import {
		BACKENDS,
		extraArgsToText,
		gpuListToText,
		listToText,
		textToExtraArgs,
		textToGpuList,
		textToList,
		type ModelConfig
	} from '$lib/config';
	import { isPending, pendingAction, runModelAction, statusMeta, isResident } from '$lib/models.svelte';
	import CaretDownIcon from '~icons/ph/caret-down';
	import PlayIcon from '~icons/ph/play';
	import SpinnerIcon from '~icons/ph/spinner';
	import StopIcon from '~icons/ph/stop';
	import ArrowClockwiseIcon from '~icons/ph/arrow-clockwise';
	import TrashIcon from '~icons/ph/trash';

	let {
		model = $bindable(),
		status = 'stopped',
		open = $bindable(false),
		onSaveRestart,
		onRemove
	}: {
		model: ModelConfig;
		status?: string;
		open?: boolean;
		onSaveRestart: () => void;
		onRemove: () => void;
	} = $props();

	const meta = $derived(statusMeta(status));
	const pending = $derived(pendingAction(model.name));
	const busy = $derived(isPending(model.name));
	const idLabel = $derived(model.name || 'new-model');
</script>

<Collapsible.Root bind:open>
	<Card.Root class="gap-0 overflow-hidden py-0">
		<Collapsible.Trigger class="w-full">
			<div class="flex cursor-pointer items-center gap-3 px-4 py-3 select-none">
				<span class="size-2 shrink-0 rounded-full {meta.dot}"></span>
				<div class="min-w-0 flex-1 text-left">
					<div class="truncate text-sm font-medium">{model.name || 'Untitled model'}</div>
					<div class="truncate text-xs text-muted-foreground">
						{model.modelPath || 'No model path'}
					</div>
				</div>
				<span class="hidden text-xs text-muted-foreground sm:block">
					{BACKENDS.find((b) => b.value === (model.backend ?? 'vllm'))?.label}
				</span>
				<span class="hidden text-xs tabular-nums text-muted-foreground sm:block">
					GPU {model.balanceGpus?.length ? model.balanceGpus.join(',') : (model.gpus?.join(',') ?? model.gpu)}
				</span>
				<span class="text-xs {meta.text}">{meta.label}</span>
				<CaretDownIcon
					class="size-4 shrink-0 text-muted-foreground transition-transform {open ? 'rotate-180' : ''}"
				/>
			</div>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div class="space-y-4 border-t border-border px-4 py-4">
				<!-- Lifecycle -->
				<div class="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-muted/50 px-3 py-2">
					<span class="text-xs text-muted-foreground">
						Edits apply after <span class="font-medium text-foreground">Save & restart</span>
					</span>
					<div class="flex items-center gap-1.5">
						{#if isResident(status)}
							<Button
								variant="outline"
								size="sm"
								class="h-7"
								disabled={busy}
								onclick={() => runModelAction(model.name, 'stop')}
							>
								{#if pending === 'stop'}
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
								class="h-7"
								disabled={busy || !model.name}
								onclick={() => runModelAction(model.name, 'start')}
							>
								{#if pending === 'start'}
									<SpinnerIcon class="size-3.5 animate-spin" />
								{:else}
									<PlayIcon class="size-3.5" />
								{/if}
								Start
							</Button>
						{/if}
						<Button
							size="sm"
							class="h-7"
							disabled={busy || !model.name}
							onclick={onSaveRestart}
						>
							{#if pending === 'restart'}
								<SpinnerIcon class="size-3.5 animate-spin" />
							{:else}
								<ArrowClockwiseIcon class="size-3.5" />
							{/if}
							Save & restart
						</Button>
						<Button
							variant="ghost"
							size="icon"
							class="size-7 text-muted-foreground hover:text-destructive"
							disabled={busy}
							onclick={onRemove}
						>
							<TrashIcon class="size-3.5" />
							<span class="sr-only">Remove {idLabel}</span>
						</Button>
					</div>
				</div>

				<!-- Identity -->
				<div class="grid gap-4 sm:grid-cols-2">
					<div class="space-y-1.5">
						<Label for="{idLabel}-name">Name</Label>
						<Input id="{idLabel}-name" bind:value={model.name} placeholder="qwen-7b" />
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-backend">Backend</Label>
						<Select.Root
							type="single"
							value={model.backend ?? 'vllm'}
							onValueChange={(v) => {
								model.backend = v === 'vllm' ? undefined : (v as ModelConfig['backend']);
							}}
						>
							<Select.Trigger id="{idLabel}-backend" class="w-full">
								{BACKENDS.find((b) => b.value === (model.backend ?? 'vllm'))?.label}
							</Select.Trigger>
							<Select.Content>
								{#each BACKENDS as backend (backend.value)}
									<Select.Item value={backend.value} label={backend.label} />
								{/each}
							</Select.Content>
						</Select.Root>
					</div>
				</div>

				<div class="grid gap-4 sm:grid-cols-[2fr_1fr]">
					<div class="space-y-1.5">
						<Label for="{idLabel}-path">Model path</Label>
						<Input
							id="{idLabel}-path"
							bind:value={model.modelPath}
							placeholder="org/repo-id on HuggingFace, or /models/file.gguf"
						/>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-file">Repo file</Label>
						<Input
							id="{idLabel}-file"
							value={model.modelFile ?? ''}
							oninput={(e) => (model.modelFile = e.currentTarget.value.trim() || undefined)}
							placeholder="model-Q4_K_M.gguf"
						/>
						<p class="text-[11px] text-muted-foreground">
							For GGUF repos with many quants — downloads only this file
						</p>
					</div>
				</div>

				<!-- Placement & lifecycle -->
				<div class="grid gap-4 sm:grid-cols-3">
					<div class="space-y-1.5">
						<Label for="{idLabel}-gpu">GPU</Label>
						<Input id="{idLabel}-gpu" type="number" min="0" bind:value={model.gpu} />
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-idle">Idle timeout (min)</Label>
						<Input
							id="{idLabel}-idle"
							type="number"
							min="1"
							value={Math.round(model.idleTimeout / 60_000)}
							oninput={(e) => {
								const minutes = parseInt(e.currentTarget.value, 10);
								if (minutes > 0) model.idleTimeout = minutes * 60_000;
							}}
						/>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-dtype">Dtype</Label>
						<Input id="{idLabel}-dtype" bind:value={model.dtype} placeholder="bfloat16" />
					</div>
				</div>

				<div class="grid gap-4 sm:grid-cols-3">
					<div class="space-y-1.5">
						<Label for="{idLabel}-priority">Priority</Label>
						<Input
							id="{idLabel}-priority"
							type="number"
							min="0"
							value={model.priority ?? 0}
							oninput={(e) => {
								const priority = parseInt(e.currentTarget.value, 10);
								model.priority = Number.isNaN(priority) ? 0 : priority;
							}}
						/>
						<p class="text-[11px] text-muted-foreground">Higher survives VRAM eviction longer</p>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-sleep">Sleep delay (s)</Label>
						<Input
							id="{idLabel}-sleep"
							type="number"
							min="0"
							value={Math.round((model.sleepDelay ?? 0) / 1000)}
							oninput={(e) => {
								const seconds = parseInt(e.currentTarget.value, 10);
								model.sleepDelay = Number.isNaN(seconds) ? 0 : seconds * 1000;
							}}
						/>
						<p class="text-[11px] text-muted-foreground">Grace period before sleeping. 0 = immediate</p>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-vram">VRAM override (GB)</Label>
						<Input
							id="{idLabel}-vram"
							type="number"
							step="0.1"
							bind:value={model.vramGB}
							placeholder="Auto-estimated"
						/>
					</div>
				</div>

				<!-- Sizing -->
				<div class="grid gap-4 sm:grid-cols-2">
					<div class="space-y-1.5">
						<Label for="{idLabel}-params">Parameters (B)</Label>
						<Input
							id="{idLabel}-params"
							type="number"
							step="0.1"
							bind:value={model.parameterCount}
							placeholder="7"
						/>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-aparams">Activated parameters (B)</Label>
						<Input
							id="{idLabel}-aparams"
							type="number"
							step="0.1"
							bind:value={model.activatedParameterCount}
							placeholder="MoE models only"
						/>
					</div>
				</div>

				<!-- Routing -->
				<div class="grid gap-4 sm:grid-cols-2">
					<div class="space-y-1.5">
						<Label for="{idLabel}-aliases">Aliases</Label>
						<Input
							id="{idLabel}-aliases"
							value={listToText(model.aliases)}
							oninput={(e) => (model.aliases = textToList(e.currentTarget.value))}
							placeholder="default, gpt-4"
						/>
					</div>
					<div class="space-y-1.5">
						<Label for="{idLabel}-balance">Balance across GPUs</Label>
						<Input
							id="{idLabel}-balance"
							value={gpuListToText(model.balanceGpus)}
							oninput={(e) => (model.balanceGpus = textToGpuList(e.currentTarget.value))}
							placeholder="0, 1"
						/>
						{#if model.balanceGpus?.length}
							<p class="text-[11px] text-muted-foreground">
								Route with <code class="rounded bg-muted px-1">{model.name}:balance</code> or pin with
								<code class="rounded bg-muted px-1">{model.name}:gpu{model.balanceGpus[0]}</code>
							</p>
						{/if}
					</div>
				</div>

				<div class="space-y-1.5">
					<Label for="{idLabel}-args">Extra args</Label>
					<Textarea
						id="{idLabel}-args"
						rows={Math.max(3, Math.ceil((model.extraArgs?.length ?? 0) / 2) + 1)}
						class="font-mono text-xs"
						value={extraArgsToText(model.extraArgs)}
						oninput={(e) => (model.extraArgs = textToExtraArgs(e.currentTarget.value))}
						placeholder={'--gpu-memory-utilization 0.85\n--max-model-len 8192'}
					/>
					<p class="text-[11px] text-muted-foreground">One flag (with its value) per line</p>
				</div>
			</div>
		</Collapsible.Content>
	</Card.Root>
</Collapsible.Root>
