<script lang="ts">
	import type { Snippet } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import ArrowDownIcon from '~icons/ph/arrow-down';
	import { scale } from 'svelte/transition';

	let { children }: { children: Snippet } = $props();

	let container = $state<HTMLDivElement>();
	let pinnedToBottom = $state(true);

	/**
	 * Stick to the bottom while streaming unless the user scrolls up; a manual
	 * scroll back down re-pins. All listeners clean up on unmount.
	 */
	$effect(() => {
		const el = container;
		if (!el) return;

		el.scrollTo(0, el.scrollHeight);

		const onScroll = () => {
			pinnedToBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
		};
		const observer = new MutationObserver(() => {
			if (pinnedToBottom) el.scrollTo(0, el.scrollHeight);
		});

		el.addEventListener('scroll', onScroll, { passive: true });
		observer.observe(el, { childList: true, subtree: true, characterData: true });
		return () => {
			el.removeEventListener('scroll', onScroll);
			observer.disconnect();
		};
	});

	function scrollToBottom() {
		container?.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
	}
</script>

<div class="relative h-full">
	<div
		bind:this={container}
		class="no-scrollbar flex h-full w-full flex-col gap-5 overflow-y-auto p-4"
	>
		{@render children()}
	</div>

	{#if !pinnedToBottom}
		<div
			class="absolute bottom-3 left-1/2 -translate-x-1/2"
			in:scale={{ start: 0.85, duration: 120, delay: 150 }}
			out:scale={{ start: 0.85, duration: 120 }}
		>
			<Button
				onclick={scrollToBottom}
				variant="outline"
				size="icon"
				class="rounded-full shadow-md"
			>
				<ArrowDownIcon class="size-4" />
				<span class="sr-only">Scroll to latest</span>
			</Button>
		</div>
	{/if}
</div>
