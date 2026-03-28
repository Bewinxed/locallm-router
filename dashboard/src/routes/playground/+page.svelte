<script lang="ts">
	import { Chat, type UIMessage } from '@ai-sdk/svelte';
	import { DefaultChatTransport } from 'ai';
	import { onMount } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import { Slider } from '$lib/components/ui/slider';
	import { Label } from '$lib/components/ui/label';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Separator } from '$lib/components/ui/separator';
	import { Badge } from '$lib/components/ui/badge';
	import * as Select from '$lib/components/ui/select';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as ScrollArea from '$lib/components/ui/scroll-area';
	import * as ChatUI from '$lib/components/ui/chat';
	import PaperPlaneRightIcon from '~icons/ph/paper-plane-right';
	import TrashIcon from '~icons/ph/trash';
	import SidebarCloseIcon from '~icons/ph/sidebar-simple';
	import SidebarOpenIcon from '~icons/ph/sidebar-simple';
	import CaretDownIcon from '~icons/ph/caret-down';
	import CaretRightIcon from '~icons/ph/caret-right';
	import BrainIcon from '~icons/ph/brain';
	import StopIcon from '~icons/ph/stop';
	import SpinnerIcon from '~icons/ph/spinner';
	import RobotIcon from '~icons/ph/robot';
	import UserIcon from '~icons/ph/user';

	// --- State: Parameters ---
	let selectedModel = $state('');
	let systemPrompt = $state('');
	let temperature = $state([0.6]);
	let topP = $state([0.95]);
	let topK = $state([20]);
	let maxTokens = $state([4096]);
	let presencePenalty = $state([0]);
	let frequencyPenalty = $state([0]);

	// --- State: UI ---
	let sidebarOpen = $state(true);
	let systemPromptOpen = $state(false);
	let inputText = $state('');
	let thinkingStates: Record<string, boolean> = $state({});

	// --- State: Models ---
	type ModelInfo = {
		id: string;
		balance?: { modifiers?: string[] };
	};
	type ModelOption = { value: string; label: string; group: string };

	let models: ModelInfo[] = $state([]);
	let modelOptions: ModelOption[] = $state([]);
	let modelsLoading = $state(true);

	// --- Chat ---
	// DefaultChatTransport handles SSE stream parsing (raw bytes → UIMessageChunk objects).
	// The custom transport was returning raw response.body which caused AbstractChat to crash
	// when calling part.type.startsWith() on unparsed byte chunks.
	const chat = new Chat({
		transport: new DefaultChatTransport({
			api: '/api/chat',
			prepareSendMessagesRequest: async ({ messages }) => ({
				body: {
					messages,
					model: selectedModel,
					system: systemPrompt || undefined,
					temperature: temperature[0],
					topP: topP[0],
					topK: topK[0],
					maxTokens: maxTokens[0],
					presencePenalty: presencePenalty[0],
					frequencyPenalty: frequencyPenalty[0]
				}
			})
		}),
		onError: (error) => {
			console.error('Chat error:', error);
		}
	});

	// --- Fetch models ---
	onMount(async () => {
		try {
			const res = await fetch('/api/models');
			if (res.ok) {
				const data = await res.json();
				models = data;
				const opts: ModelOption[] = [];

				for (const model of models) {
					opts.push({
						value: model.id,
						label: model.id,
						group: 'Base Models'
					});

					if (model.balance?.modifiers) {
						for (const modifier of model.balance.modifiers) {
							opts.push({
								value: modifier,
								label: modifier,
								group: `${model.id} Modifiers`
							});
						}
					}
				}

				modelOptions = opts;
				if (opts.length > 0 && !selectedModel) {
					selectedModel = opts[0].value;
				}
			}
		} catch (e) {
			console.error('Failed to fetch models:', e);
		} finally {
			modelsLoading = false;
		}
	});

	// --- Send message ---
	async function handleSend() {
		const text = inputText.trim();
		if (!text || chat.status === 'streaming' || chat.status === 'submitted') return;
		inputText = '';
		await chat.sendMessage({ text });
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	function clearChat() {
		chat.messages = [];
	}

	// --- Parse thinking from text ---
	function parseThinking(text: string): { thinking: string | null; content: string } {
		const thinkRegex = /^<think>([\s\S]*?)<\/think>\s*/;
		const match = text.match(thinkRegex);
		if (match) {
			return {
				thinking: match[1].trim(),
				content: text.slice(match[0].length).trim()
			};
		}
		// Handle unclosed think tags (streaming)
		const openMatch = text.match(/^<think>([\s\S]*)/);
		if (openMatch) {
			return {
				thinking: openMatch[1].trim(),
				content: ''
			};
		}
		return { thinking: null, content: text };
	}

	function toggleThinking(id: string) {
		thinkingStates[id] = !thinkingStates[id];
	}

	// --- Computed helpers ---
	function getMessageText(message: UIMessage): string {
		return message.parts
			.filter((p): p is { type: 'text'; text: string } => p.type === 'text')
			.map((p) => p.text)
			.join('');
	}

	function getGroups(options: ModelOption[]): string[] {
		const seen = new Set<string>();
		return options.filter((o) => {
			if (seen.has(o.group)) return false;
			seen.add(o.group);
			return true;
		}).map((o) => o.group);
	}

	let isLoading = $derived(chat.status === 'streaming' || chat.status === 'submitted');
</script>

<div class="flex h-full w-full overflow-hidden">
	<!-- Left Panel: Parameters -->
	{#if sidebarOpen}
		<div
			class="flex h-full w-[300px] shrink-0 flex-col border-r border-border bg-card"
		>
			<!-- Sidebar Header -->
			<div class="flex h-12 items-center justify-between border-b border-border px-3">
				<span class="text-sm font-semibold text-foreground">Parameters</span>
				<Button variant="ghost" size="icon-sm" onclick={() => (sidebarOpen = false)}>
					<SidebarCloseIcon class="size-4" />
				</Button>
			</div>

			<!-- Sidebar Content -->
			<ScrollArea.Root class="flex-1">
				<div class="flex flex-col gap-4 p-3">
					<!-- Model Selector -->
					<div class="flex flex-col gap-1.5">
						<Label class="text-xs text-muted-foreground">Model</Label>
						{#if modelsLoading}
							<div class="flex h-9 items-center gap-2 rounded-md border border-input px-3">
								<SpinnerIcon class="size-3.5 animate-spin text-muted-foreground" />
								<span class="text-sm text-muted-foreground">Loading models...</span>
							</div>
						{:else}
							<Select.Root bind:value={selectedModel} type="single">
								<Select.Trigger class="w-full">
									{#snippet children()}
										<span class="truncate text-sm">
											{selectedModel || 'Select a model'}
										</span>
									{/snippet}
								</Select.Trigger>
								<Select.Content>
									{#each getGroups(modelOptions) as group}
										<Select.Group>
											<Select.GroupHeading>{group}</Select.GroupHeading>
											{#each modelOptions.filter((o) => o.group === group) as opt}
												<Select.Item value={opt.value} label={opt.label} />
											{/each}
										</Select.Group>
									{/each}
								</Select.Content>
							</Select.Root>
						{/if}
					</div>

					<Separator />

					<!-- System Prompt -->
					<Collapsible.Root bind:open={systemPromptOpen}>
						<Collapsible.Trigger class="flex w-full items-center justify-between text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
							<span>System Prompt</span>
							{#if systemPromptOpen}
								<CaretDownIcon class="size-3.5" />
							{:else}
								<CaretRightIcon class="size-3.5" />
							{/if}
						</Collapsible.Trigger>
						<Collapsible.Content>
							<div class="pt-2">
								<Textarea
									bind:value={systemPrompt}
									placeholder="You are a helpful assistant..."
									class="min-h-[80px] resize-none text-xs"
								/>
							</div>
						</Collapsible.Content>
					</Collapsible.Root>

					<Separator />

					<!-- Temperature -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Temperature</Label>
							<span class="text-xs font-mono text-foreground">{temperature[0].toFixed(1)}</span>
						</div>
						<Slider type="multiple" bind:value={temperature} min={0} max={2} step={0.1} />
					</div>

					<!-- Top P -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Top P</Label>
							<span class="text-xs font-mono text-foreground">{topP[0].toFixed(2)}</span>
						</div>
						<Slider type="multiple" bind:value={topP} min={0} max={1} step={0.05} />
					</div>

					<!-- Top K -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Top K</Label>
							<span class="text-xs font-mono text-foreground">{topK[0]}</span>
						</div>
						<Slider type="multiple" bind:value={topK} min={0} max={100} step={1} />
					</div>

					<!-- Max Tokens -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Max Tokens</Label>
							<span class="text-xs font-mono text-foreground">{maxTokens[0]}</span>
						</div>
						<Slider type="multiple" bind:value={maxTokens} min={1} max={8192} step={1} />
					</div>

					<!-- Presence Penalty -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Presence Penalty</Label>
							<span class="text-xs font-mono text-foreground">{presencePenalty[0].toFixed(1)}</span>
						</div>
						<Slider type="multiple" bind:value={presencePenalty} min={-2} max={2} step={0.1} />
					</div>

					<!-- Frequency Penalty -->
					<div class="flex flex-col gap-2">
						<div class="flex items-center justify-between">
							<Label class="text-xs text-muted-foreground">Frequency Penalty</Label>
							<span class="text-xs font-mono text-foreground">{frequencyPenalty[0].toFixed(1)}</span>
						</div>
						<Slider type="multiple" bind:value={frequencyPenalty} min={-2} max={2} step={0.1} />
					</div>
				</div>
			</ScrollArea.Root>
		</div>
	{/if}

	<!-- Right Panel: Chat -->
	<div class="flex flex-1 flex-col overflow-hidden">
		<!-- Chat Header -->
		<div class="flex h-12 items-center justify-between border-b border-border px-4">
			<div class="flex items-center gap-2">
				{#if !sidebarOpen}
					<Button variant="ghost" size="icon-sm" onclick={() => (sidebarOpen = true)}>
						<SidebarOpenIcon class="size-4" />
					</Button>
				{/if}
				<span class="text-sm font-semibold text-foreground">Chat</span>
				{#if selectedModel}
					<Badge variant="secondary" class="text-xs font-mono">{selectedModel}</Badge>
				{/if}
				{#if isLoading}
					<SpinnerIcon class="size-3.5 animate-spin text-muted-foreground" />
				{/if}
			</div>
			<div class="flex items-center gap-1">
				{#if isLoading}
					<Button variant="ghost" size="icon-sm" onclick={() => chat.stop()}>
						<StopIcon class="size-4" />
					</Button>
				{/if}
				<Button
					variant="ghost"
					size="icon-sm"
					onclick={clearChat}
					disabled={chat.messages.length === 0}
				>
					<TrashIcon class="size-4" />
				</Button>
			</div>
		</div>

		<!-- Messages Area -->
		<div class="flex-1 overflow-hidden">
			{#if chat.messages.length === 0}
				<div class="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
					<RobotIcon class="size-10 opacity-30" />
					<p class="text-sm">Send a message to start chatting</p>
					{#if !selectedModel}
						<p class="text-xs text-destructive">No model selected</p>
					{/if}
				</div>
			{:else}
				<ChatUI.List class="mx-auto max-w-3xl">
					{#each chat.messages as message (message.id)}
						{#if message.role === 'user'}
							<ChatUI.Bubble variant="sent">
								<ChatUI.BubbleMessage>
									<p class="whitespace-pre-wrap">{getMessageText(message)}</p>
								</ChatUI.BubbleMessage>
								<ChatUI.BubbleAvatar>
									<ChatUI.BubbleAvatarFallback class="bg-primary/15 text-primary">
										<UserIcon class="size-3.5" />
									</ChatUI.BubbleAvatarFallback>
								</ChatUI.BubbleAvatar>
							</ChatUI.Bubble>
						{:else if message.role === 'assistant'}
							{@const fullText = getMessageText(message)}
							{@const parsed = parseThinking(fullText)}
							<ChatUI.Bubble variant="received">
								<ChatUI.BubbleAvatar>
									<ChatUI.BubbleAvatarFallback class="bg-accent text-accent-foreground">
										<RobotIcon class="size-3.5" />
									</ChatUI.BubbleAvatarFallback>
								</ChatUI.BubbleAvatar>
								<div class="order-2 flex flex-col gap-2">
									<!-- Thinking block -->
									{#if parsed.thinking}
										{@const isExpanded = thinkingStates[message.id] ?? false}
										<button
											class="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
											onclick={() => toggleThinking(message.id)}
										>
											<BrainIcon class="size-3" />
											<span class="italic">Thinking...</span>
											{#if isExpanded}
												<CaretDownIcon class="size-3" />
											{:else}
												<CaretRightIcon class="size-3" />
											{/if}
										</button>
										{#if isExpanded}
											<div class="rounded-lg border border-border/50 bg-muted/30 px-3 py-2">
												<p class="whitespace-pre-wrap text-xs italic text-muted-foreground leading-relaxed">
													{parsed.thinking}
												</p>
											</div>
										{/if}
									{/if}

									<!-- Response content -->
									{#if parsed.content}
										<ChatUI.BubbleMessage>
											<p class="whitespace-pre-wrap">{parsed.content}</p>
										</ChatUI.BubbleMessage>
									{:else if !parsed.thinking && !isLoading}
										<ChatUI.BubbleMessage class="text-muted-foreground italic">
											<p>No response</p>
										</ChatUI.BubbleMessage>
									{:else if !parsed.content && parsed.thinking && isLoading}
										<!-- Still thinking, no content yet -->
									{/if}

									<!-- Token info from metadata -->
									{#if message.metadata}
										{@const meta = message.metadata as Record<string, any>}
										{#if meta.usage || meta.finishReason}
											<div class="flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground/70">
												{#if meta.usage?.inputTokens != null}
													<span>{meta.usage.inputTokens} in</span>
												{/if}
												{#if meta.usage?.outputTokens != null}
													<span>{meta.usage.outputTokens} out</span>
												{/if}
												{#if meta.usage?.totalTokens != null}
													<span class="font-medium">{meta.usage.totalTokens} total</span>
												{/if}
												{#if meta.finishReason}
													<span class="text-muted-foreground/50">({meta.finishReason})</span>
												{/if}
												{#if meta.duration}
													<span>{(meta.duration / 1000).toFixed(1)}s</span>
												{/if}
											</div>
										{/if}
									{/if}
								</div>
							</ChatUI.Bubble>
						{/if}
					{/each}

					<!-- Streaming indicator -->
					{#if chat.status === 'submitted'}
						<ChatUI.Bubble variant="received">
							<ChatUI.BubbleAvatar>
								<ChatUI.BubbleAvatarFallback class="bg-accent text-accent-foreground">
									<RobotIcon class="size-3.5" />
								</ChatUI.BubbleAvatarFallback>
							</ChatUI.BubbleAvatar>
							<ChatUI.BubbleMessage typing={true} />
						</ChatUI.Bubble>
					{/if}
				</ChatUI.List>
			{/if}
		</div>

		<!-- Input Area -->
		<div class="border-t border-border bg-card p-3">
			<div class="mx-auto flex max-w-3xl items-end gap-2">
				<div class="relative flex-1">
					<textarea
						bind:value={inputText}
						onkeydown={handleKeydown}
						placeholder="Send a message..."
						rows={1}
						disabled={!selectedModel}
						class="border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 flex min-h-[40px] max-h-[160px] w-full resize-none rounded-lg border bg-transparent px-3 py-2.5 pr-12 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50"
					></textarea>
				</div>
				<Button
					size="icon"
					disabled={!inputText.trim() || isLoading || !selectedModel}
					onclick={handleSend}
				>
					<PaperPlaneRightIcon class="size-4" />
				</Button>
			</div>
			{#if chat.error}
				<p class="mx-auto mt-2 max-w-3xl text-xs text-destructive">
					Error: {chat.error.message}
				</p>
			{/if}
		</div>
	</div>
</div>
