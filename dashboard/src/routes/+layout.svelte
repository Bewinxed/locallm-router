<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	import { Toaster } from '$lib/components/ui/sonner';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import SquaresFourIcon from '~icons/ph/squares-four';
	import ChatIcon from '~icons/ph/chat';
	import GearIcon from '~icons/ph/gear';
	import PulseIcon from '~icons/ph/pulse';

	let { children } = $props();

	const navItems = [
		{ href: '/', label: 'Overview', icon: SquaresFourIcon },
		{ href: '/activity', label: 'Activity', icon: PulseIcon },
		{ href: '/playground', label: 'Playground', icon: ChatIcon },
		{ href: '/config', label: 'Settings', icon: GearIcon }
	] as const;

	function isActive(itemHref: string, currentPath: string): boolean {
		if (itemHref === '/') return currentPath === '/';
		return currentPath.startsWith(itemHref);
	}
</script>

<Toaster richColors closeButton position="bottom-right" />
<Tooltip.Provider>

<div class="flex h-screen w-screen overflow-hidden bg-background">
	<!-- Sidebar: icon-only rail -->
	<aside class="flex h-full w-12 flex-col items-center border-r border-border bg-sidebar py-3">
		<!-- Logo -->
		<div class="mb-4 flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-bold">
			LR
		</div>

		<!-- Nav -->
		<nav class="flex flex-1 flex-col items-center gap-1">
			{#each navItems as item}
				{@const active = isActive(item.href, $page.url.pathname)}
				<Tooltip.Root>
					<Tooltip.Trigger>
						<a
							href={item.href}
							class="flex size-8 items-center justify-center rounded-md transition-colors {active
								? 'bg-accent text-foreground'
								: 'text-muted-foreground hover:text-foreground hover:bg-accent/60'}"
						>
							<item.icon class="size-4" />
						</a>
					</Tooltip.Trigger>
					<Tooltip.Content side="right" class="text-xs">
						{item.label}
					</Tooltip.Content>
				</Tooltip.Root>
			{/each}
		</nav>
	</aside>

	<!-- Main content -->
	<main class="flex-1 overflow-auto">
		{@render children()}
	</main>
</div>

</Tooltip.Provider>
