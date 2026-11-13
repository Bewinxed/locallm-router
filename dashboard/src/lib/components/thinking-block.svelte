<script lang="ts">
	import { slide } from 'svelte/transition';
	import Markdown from './markdown.svelte';
	import BrainIcon from '~icons/ph/brain';
	import CaretRightIcon from '~icons/ph/caret-right';
	import SpinnerIcon from '~icons/ph/spinner';

	let {
		text,
		active = false
	}: {
		text: string;
		/** Still receiving reasoning tokens — shown as "Thinking…" with a spinner. */
		active?: boolean;
	} = $props();

	let open = $state(false);
</script>

<div class="min-w-0">
	<button
		class="flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
		aria-expanded={open}
		onclick={() => (open = !open)}
	>
		{#if active}
			<SpinnerIcon class="size-3 animate-spin" />
		{:else}
			<BrainIcon class="size-3" />
		{/if}
		<span>{active ? 'Thinking…' : 'Thought process'}</span>
		<CaretRightIcon class="size-3 transition-transform {open ? 'rotate-90' : ''}" />
	</button>

	{#if open}
		<div
			transition:slide={{ duration: 150 }}
			class="mt-1.5 rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground [&_.markdown-body]:text-xs"
		>
			<Markdown source={text} />
		</div>
	{/if}
</div>
