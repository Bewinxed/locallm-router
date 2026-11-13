<script lang="ts">
	import { Chat } from '@ai-sdk/svelte';
	import { DefaultChatTransport } from 'ai';
	import type { UIMessage } from 'ai';
	import ChatList from '$lib/components/chat-list.svelte';
	import Markdown from '$lib/components/markdown.svelte';
	import ParamSlider from '$lib/components/param-slider.svelte';
	import ThinkingBlock from '$lib/components/thinking-block.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import { Separator } from '$lib/components/ui/separator';
	import { Textarea } from '$lib/components/ui/textarea';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Select from '$lib/components/ui/select';
	import type { ChatMetadata } from '../api/chat/+server';
	import type { RouterModel } from '$lib/types';
	import CaretRightIcon from '~icons/ph/caret-right';
	import ChatIcon from '~icons/ph/chat';
	import PaperPlaneRightIcon from '~icons/ph/paper-plane-right';
	import SidebarIcon from '~icons/ph/sidebar-simple';
	import SpinnerIcon from '~icons/ph/spinner';
	import StopIcon from '~icons/ph/stop';
	import TrashIcon from '~icons/ph/trash';

	// ─── Parameters ───────────────────────────────────────────
	let selectedModel = $state('');
	let systemPrompt = $state('');
	let temperature = $state(0.6);
	let topP = $state(0.95);
	let topK = $state(20);
	let maxTokens = $state(4096);
	let presencePenalty = $state(0);
	let frequencyPenalty = $state(0);

	let sidebarOpen = $state(true);
	let systemPromptOpen = $state(false);
	let inputText = $state('');

	// ─── Models ───────────────────────────────────────────────
	let models = $state<RouterModel[]>([]);
	let modelsLoading = $state(true);
	let modelsError = $state<string | null>(null);

	$effect(() => {
		(async () => {
			try {
				const res = await fetch('/api/models');
				const data = await res.json();
				if (!res.ok) throw new Error(data?.error || 'Could not load models');
				models = data;
				if (models.length && !selectedModel) selectedModel = models[0].id;
			} catch (err) {
				modelsError = err instanceof Error ? err.message : 'Could not load models';
			} finally {
				modelsLoading = false;
			}
		})();
	});

	// ─── Chat ─────────────────────────────────────────────────
	const chat = new Chat({
		transport: new DefaultChatTransport({
			api: '/api/chat',
			prepareSendMessagesRequest: ({ messages }) => ({
				body: {
					messages,
					model: selectedModel,
					system: systemPrompt || undefined,
					temperature,
					topP,
					topK,
					maxTokens,
					presencePenalty,
					frequencyPenalty
				}
			})
		})
	});

	const isLoading = $derived(chat.status === 'streaming' || chat.status === 'submitted');
	const lastMessageId = $derived(chat.messages.at(-1)?.id);

	async function handleSend() {
		const text = inputText.trim();
		if (!text || isLoading || !selectedModel) return;
		inputText = '';
		await chat.sendMessage({ text });
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
			event.preventDefault();
			handleSend();
		}
	}

	// ─── Message parsing ──────────────────────────────────────
	/**
	 * Split an assistant message into thinking and answer.
	 *
	 * Reasoning arrives two ways depending on backend: as AI SDK `reasoning`
	 * parts (vLLM with a reasoning parser), or inline `<think>…</think>` tags
	 * in the text (llama-server). Handles both, including an unclosed tag
	 * mid-stream.
	 */
	function parseMessage(message: UIMessage): {
		thinking: string;
		content: string;
		thinkingActive: boolean;
	} {
		const reasoning = message.parts
			.filter((p) => p.type === 'reasoning')
			.map((p) => p.text)
			.join('');
		const text = message.parts
			.filter((p) => p.type === 'text')
			.map((p) => p.text)
			.join('');

		const closed = text.match(/^\s*<think>([\s\S]*?)<\/think>\s*/);
		if (closed) {
			return {
				thinking: reasoning + closed[1].trim(),
				content: text.slice(closed[0].length),
				thinkingActive: false
			};
		}
		const openTag = text.match(/^\s*<think>([\s\S]*)/);
		if (openTag) {
			return { thinking: reasoning + openTag[1].trim(), content: '', thinkingActive: true };
		}
		return {
			thinking: reasoning,
			content: text,
			thinkingActive: reasoning.length > 0 && text.length === 0
		};
	}

	function messageText(message: UIMessage): string {
		return message.parts
			.filter((p) => p.type === 'text')
			.map((p) => p.text)
			.join('');
	}
</script>

<svelte:head>
	<title>Playground · locallm-router</title>
</svelte:head>

<div class="flex h-full w-full overflow-hidden">
	{#if sidebarOpen}
		<aside class="flex h-full w-72 shrink-0 flex-col border-r border-border bg-card">
			<div class="flex h-12 shrink-0 items-center justify-between border-b border-border px-3">
				<span class="text-sm font-semibold">Parameters</span>
				<Button variant="ghost" size="icon" class="size-7" onclick={() => (sidebarOpen = false)}>
					<SidebarIcon class="size-4" />
					<span class="sr-only">Hide parameters</span>
				</Button>
			</div>

			<div class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-3">
				<div class="flex flex-col gap-1.5">
					<Label class="text-xs text-muted-foreground">Model</Label>
					{#if modelsLoading}
						<div class="flex h-9 items-center gap-2 rounded-md border border-input px-3">
							<SpinnerIcon class="size-3.5 animate-spin text-muted-foreground" />
							<span class="text-sm text-muted-foreground">Loading…</span>
						</div>
					{:else if modelsError}
						<p class="text-xs text-destructive">{modelsError}</p>
					{:else}
						<Select.Root bind:value={selectedModel} type="single">
							<Select.Trigger class="w-full">
								<span class="truncate font-mono text-xs">
									{selectedModel || 'Select a model'}
								</span>
							</Select.Trigger>
							<Select.Content>
								{#each models as model (model.id)}
									<Select.Item value={model.id} label={model.id} />
									{#each model.balance?.modifiers ?? [] as modifier (modifier)}
										<Select.Item value={modifier} label={modifier} class="pl-6 font-mono text-xs" />
									{/each}
								{/each}
							</Select.Content>
						</Select.Root>
					{/if}
				</div>

				<Separator />

				<Collapsible.Root bind:open={systemPromptOpen}>
					<Collapsible.Trigger
						class="flex w-full items-center justify-between text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
					>
						<span>System prompt</span>
						<CaretRightIcon
							class="size-3.5 transition-transform {systemPromptOpen ? 'rotate-90' : ''}"
						/>
					</Collapsible.Trigger>
					<Collapsible.Content>
						<Textarea
							bind:value={systemPrompt}
							placeholder="You are a helpful assistant…"
							class="mt-2 max-h-48 min-h-20 text-xs"
						/>
					</Collapsible.Content>
				</Collapsible.Root>

				<Separator />

				<ParamSlider label="Temperature" bind:value={temperature} min={0} max={2} step={0.1} decimals={1} />
				<ParamSlider label="Top P" bind:value={topP} min={0} max={1} step={0.05} decimals={2} />
				<ParamSlider label="Top K" bind:value={topK} min={0} max={100} step={1} />
				<ParamSlider label="Max tokens" bind:value={maxTokens} min={1} max={32768} step={1} />
				<ParamSlider
					label="Presence penalty"
					bind:value={presencePenalty}
					min={-2}
					max={2}
					step={0.1}
					decimals={1}
				/>
				<ParamSlider
					label="Frequency penalty"
					bind:value={frequencyPenalty}
					min={-2}
					max={2}
					step={0.1}
					decimals={1}
				/>
			</div>
		</aside>
	{/if}

	<div class="flex min-w-0 flex-1 flex-col overflow-hidden">
		<header class="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
			<div class="flex min-w-0 items-center gap-2">
				{#if !sidebarOpen}
					<Button variant="ghost" size="icon" class="size-7" onclick={() => (sidebarOpen = true)}>
						<SidebarIcon class="size-4" />
						<span class="sr-only">Show parameters</span>
					</Button>
				{/if}
				<span class="text-sm font-semibold">Playground</span>
				{#if selectedModel}
					<Badge variant="secondary" class="max-w-64 truncate font-mono text-xs">
						{selectedModel}
					</Badge>
				{/if}
			</div>
			<div class="flex items-center gap-1">
				{#if isLoading}
					<Button variant="ghost" size="icon" class="size-7" onclick={() => chat.stop()}>
						<StopIcon class="size-4" />
						<span class="sr-only">Stop generating</span>
					</Button>
				{/if}
				<Button
					variant="ghost"
					size="icon"
					class="size-7"
					disabled={chat.messages.length === 0}
					onclick={() => (chat.messages = [])}
				>
					<TrashIcon class="size-4" />
					<span class="sr-only">Clear conversation</span>
				</Button>
			</div>
		</header>

		<div class="min-h-0 flex-1">
			{#if chat.messages.length === 0}
				<div
					class="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground"
				>
					<ChatIcon class="size-10 opacity-25" />
					{#if modelsError}
						<p class="text-sm">Can't reach the router</p>
						<p class="text-xs">{modelsError}</p>
					{:else if !modelsLoading && models.length === 0}
						<p class="text-sm">No models configured</p>
						<a href="/config" class="text-xs underline underline-offset-4">Add one in Settings</a>
					{:else}
						<p class="text-sm">Send a message to start</p>
						<p class="text-xs">A cold model loads on first request — expect a short wait</p>
					{/if}
				</div>
			{:else}
				<ChatList>
					<div class="mx-auto w-full max-w-3xl space-y-5">
						{#each chat.messages as message (message.id)}
							{#if message.role === 'user'}
								<div class="flex justify-end">
									<div
										class="max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-primary px-3.5 py-2 text-sm text-primary-foreground"
									>
										{messageText(message)}
									</div>
								</div>
							{:else if message.role === 'assistant'}
								{@const parsed = parseMessage(message)}
								{@const streamingThis = chat.status === 'streaming' && message.id === lastMessageId}
								{@const metadata = message.metadata as ChatMetadata | undefined}
								<div class="flex flex-col gap-2">
									{#if parsed.thinking}
										<ThinkingBlock
											text={parsed.thinking}
											active={streamingThis && parsed.thinkingActive}
										/>
									{/if}

									{#if parsed.content}
										<Markdown
											source={parsed.content}
											streaming={streamingThis}
											streamId={message.id}
										/>
									{:else if !parsed.thinking && !streamingThis}
										<p class="text-sm italic text-muted-foreground">Empty response</p>
									{/if}

									{#if metadata?.totalTokens != null || metadata?.finishReason}
										<div
											class="flex flex-wrap items-center gap-x-2.5 text-[10px] tabular-nums text-muted-foreground/70"
										>
											{#if metadata.inputTokens != null}<span>{metadata.inputTokens} in</span>{/if}
											{#if metadata.outputTokens != null}<span>{metadata.outputTokens} out</span>{/if}
											{#if metadata.tokensPerSecond != null}
												<span>{metadata.tokensPerSecond} tok/s</span>
											{/if}
											{#if metadata.durationMs != null}
												<span>{(metadata.durationMs / 1000).toFixed(1)}s</span>
											{/if}
											{#if metadata.finishReason && metadata.finishReason !== 'stop'}
												<span class="text-amber-500">{metadata.finishReason}</span>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						{/each}

						{#if chat.status === 'submitted'}
							<div class="flex items-center gap-2 text-sm text-muted-foreground">
								<SpinnerIcon class="size-3.5 animate-spin" />
								Waiting for the model…
							</div>
						{/if}
					</div>
				</ChatList>
			{/if}
		</div>

		<footer class="shrink-0 border-t border-border bg-card p-3">
			<div class="mx-auto flex max-w-3xl items-end gap-2">
				<Textarea
					bind:value={inputText}
					onkeydown={handleKeydown}
					placeholder={selectedModel ? 'Send a message…' : 'Select a model first'}
					rows={1}
					disabled={!selectedModel}
					class="max-h-40 min-h-10 flex-1 resize-none"
				/>
				<Button
					size="icon"
					disabled={!inputText.trim() || isLoading || !selectedModel}
					onclick={handleSend}
				>
					<PaperPlaneRightIcon class="size-4" />
					<span class="sr-only">Send</span>
				</Button>
			</div>
			{#if chat.error}
				<p class="mx-auto mt-2 max-w-3xl text-xs text-destructive">{chat.error.message}</p>
			{/if}
		</footer>
	</div>
</div>
