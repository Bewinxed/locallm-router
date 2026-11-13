<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { ModeWatcher, mode, toggleMode } from 'mode-watcher';
	import { Toaster } from '$lib/components/ui/sonner';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import SquaresFourIcon from '~icons/ph/squares-four';
	import ChatIcon from '~icons/ph/chat';
	import GearIcon from '~icons/ph/gear';
	import SunIcon from '~icons/ph/sun';
	import MoonIcon from '~icons/ph/moon';

	let { children } = $props();

	const navItems = [
		{ href: '/', label: 'Overview', icon: SquaresFourIcon },
		{ href: '/playground', label: 'Playground', icon: ChatIcon },
		{ href: '/config', label: 'Settings', icon: GearIcon }
	] as const;

	const isActive = (href: string) =>
		href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href);
</script>

<ModeWatcher defaultMode="dark" />
<Toaster richColors closeButton position="bottom-right" />

<Tooltip.Provider delayDuration={300}>
	<div class="flex h-screen w-screen overflow-hidden bg-background">
		<aside
			class="flex h-full w-12 shrink-0 flex-col items-center gap-1 border-r border-border bg-sidebar py-3"
		>
			<a
				href="/"
				aria-label="locallm-router home"
				class="mb-3 flex size-7 items-center justify-center rounded-md bg-primary text-[11px] font-bold text-primary-foreground"
			>
				LR
			</a>

			<nav class="flex flex-1 flex-col items-center gap-1">
				{#each navItems as item (item.href)}
					<Tooltip.Root>
						<Tooltip.Trigger>
							{#snippet child({ props })}
								<a
									{...props}
									href={item.href}
									aria-current={isActive(item.href) ? 'page' : undefined}
									class="flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground aria-[current=page]:bg-accent aria-[current=page]:text-foreground"
								>
									<item.icon class="size-4" />
									<span class="sr-only">{item.label}</span>
								</a>
							{/snippet}
						</Tooltip.Trigger>
						<Tooltip.Content side="right">{item.label}</Tooltip.Content>
					</Tooltip.Root>
				{/each}
			</nav>

			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<Button
							{...props}
							variant="ghost"
							size="icon"
							class="size-8 text-muted-foreground hover:text-foreground"
							onclick={toggleMode}
						>
							{#if mode.current === 'dark'}
								<MoonIcon class="size-4" />
							{:else}
								<SunIcon class="size-4" />
							{/if}
							<span class="sr-only">Toggle theme</span>
						</Button>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content side="right">
					Switch to {mode.current === 'dark' ? 'light' : 'dark'}
				</Tooltip.Content>
			</Tooltip.Root>
		</aside>

		<main class="min-w-0 flex-1 overflow-hidden">
			{@render children()}
		</main>
	</div>
</Tooltip.Provider>
