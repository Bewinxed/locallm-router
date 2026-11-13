<script lang="ts">
	import SvelteMarkdown from '@humanspeak/svelte-markdown';
	import CodeBlock from './code-block.svelte';

	let {
		source,
		streaming = false,
		streamId
	}: { source: string; streaming?: boolean; streamId?: string } = $props();
</script>

<div class="markdown-body min-w-0 text-sm leading-relaxed">
	<SvelteMarkdown {source} {streaming} {streamId} renderers={{ code: CodeBlock }} />
</div>

<style>
	.markdown-body :global(> * + *) {
		margin-top: 0.6em;
	}
	.markdown-body :global(h1),
	.markdown-body :global(h2),
	.markdown-body :global(h3),
	.markdown-body :global(h4) {
		font-weight: 600;
		line-height: 1.3;
	}
	.markdown-body :global(h1) {
		font-size: 1.15em;
	}
	.markdown-body :global(h2) {
		font-size: 1.1em;
	}
	.markdown-body :global(h3),
	.markdown-body :global(h4) {
		font-size: 1em;
	}
	.markdown-body :global(ul),
	.markdown-body :global(ol) {
		padding-inline-start: 1.4em;
	}
	.markdown-body :global(ul) {
		list-style: disc;
	}
	.markdown-body :global(ol) {
		list-style: decimal;
	}
	.markdown-body :global(li + li) {
		margin-top: 0.2em;
	}
	.markdown-body :global(a) {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.markdown-body :global(blockquote) {
		border-inline-start: 2px solid var(--border);
		padding-inline-start: 0.75em;
		color: var(--muted-foreground);
	}
	.markdown-body :global(code:not(pre code)) {
		background: var(--muted);
		border-radius: 4px;
		padding: 0.1em 0.35em;
		font-size: 0.9em;
	}
	.markdown-body :global(pre) {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.75em 1em;
		overflow-x: auto;
		font-size: 0.85em;
		line-height: 1.55;
		/* Our surface token wins over Shiki's inline theme background. */
		background: var(--muted) !important;
	}
	.markdown-body :global(table) {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.9em;
	}
	.markdown-body :global(th),
	.markdown-body :global(td) {
		border: 1px solid var(--border);
		padding: 0.35em 0.6em;
		text-align: start;
	}
	.markdown-body :global(th) {
		background: var(--muted);
		font-weight: 600;
	}
</style>
