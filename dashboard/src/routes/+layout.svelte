<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	import { Toaster } from '$lib/components/ui/sonner';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import GpuIcon from '~icons/ph/graphics-card';
	import SquaresFourIcon from '~icons/ph/squares-four';
	import ChatIcon from '~icons/ph/chat';
	import GearIcon from '~icons/ph/gear';
	import PulseIcon from '~icons/ph/pulse';
	import SidebarIcon from '~icons/ph/sidebar';
	import CaretLeftIcon from '~icons/ph/caret-left';

	let { children } = $props();

	let collapsed = $state(false);

	const navItems = [
		{ href: '/', label: 'Dashboard', icon: SquaresFourIcon },
		{ href: '/playground', label: 'Playground', icon: ChatIcon },
		{ href: '/config', label: 'Config', icon: GearIcon },
		{ href: '/activity', label: 'Activity', icon: PulseIcon }
	] as const;

	function isActive(itemHref: string, currentPath: string): boolean {
		if (itemHref === '/') return currentPath === '/';
		return currentPath.startsWith(itemHref);
	}
</script>

<Toaster richColors closeButton position="bottom-right" />

<div class="flex h-screen w-screen overflow-hidden bg-background">
	<!-- Sidebar -->
	<aside
		class="flex h-full flex-col border-r border-border bg-sidebar transition-all duration-200 ease-in-out {collapsed
			? 'w-16'
			: 'w-56'}"
	>
		<!-- Header -->
		<div class="flex h-14 items-center gap-3 border-b border-sidebar-border px-4">
			<div
				class="flex size-8 shrink-0 items-center justify-center rounded-md bg-amber-500/15 text-amber-500"
			>
				<GpuIcon class="size-4" />
			</div>
			{#if !collapsed}
				<span class="truncate text-sm font-semibold text-sidebar-foreground">locallm-router</span>
			{/if}
		</div>

		<!-- Nav -->
		<nav class="flex flex-1 flex-col gap-1 p-2">
			{#each navItems as item}
				{@const active = isActive(item.href, $page.url.pathname)}
				{#if collapsed}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Button
								href={item.href}
								variant="ghost"
								size="icon"
								class="w-full {active
									? 'bg-sidebar-accent text-amber-500'
									: 'text-sidebar-foreground/60 hover:text-sidebar-foreground'}"
							>
								<item.icon class="size-4" />
							</Button>
						</Tooltip.Trigger>
						<Tooltip.Content side="right">
							<p>{item.label}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{:else}
					<Button
						href={item.href}
						variant="ghost"
						size="sm"
						class="w-full justify-start gap-3 {active
							? 'bg-sidebar-accent text-amber-500'
							: 'text-sidebar-foreground/60 hover:text-sidebar-foreground'}"
					>
						<item.icon class="size-4" />
						<span class="truncate">{item.label}</span>
					</Button>
				{/if}
			{/each}
		</nav>

		<!-- Collapse toggle -->
		<div class="border-t border-sidebar-border p-2">
			<Button
				variant="ghost"
				size={collapsed ? 'icon' : 'sm'}
				class="w-full {collapsed ? '' : 'justify-start gap-3'} text-sidebar-foreground/60 hover:text-sidebar-foreground"
				onclick={() => (collapsed = !collapsed)}
			>
			{#if collapsed}
				<SidebarIcon class="size-4" />
			{:else}
				<CaretLeftIcon class="size-4" />
				<span class="truncate">Collapse</span>
			{/if}
			</Button>
		</div>
	</aside>

	<!-- Main content -->
	<main class="flex-1 overflow-auto">
		{@render children()}
	</main>
</div>
