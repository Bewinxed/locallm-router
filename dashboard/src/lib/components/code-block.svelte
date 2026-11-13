<script lang="ts" module>
	import { createShikiHighlighter } from '@humanspeak/svelte-markdown/extensions/shiki';
	import bash from 'shiki/langs/bash.mjs';
	import c from 'shiki/langs/c.mjs';
	import cpp from 'shiki/langs/cpp.mjs';
	import css from 'shiki/langs/css.mjs';
	import go from 'shiki/langs/go.mjs';
	import html from 'shiki/langs/html.mjs';
	import java from 'shiki/langs/java.mjs';
	import javascript from 'shiki/langs/javascript.mjs';
	import json from 'shiki/langs/json.mjs';
	import markdown from 'shiki/langs/markdown.mjs';
	import python from 'shiki/langs/python.mjs';
	import rust from 'shiki/langs/rust.mjs';
	import sql from 'shiki/langs/sql.mjs';
	import typescript from 'shiki/langs/typescript.mjs';
	import yaml from 'shiki/langs/yaml.mjs';
	import githubDark from 'shiki/themes/github-dark-default.mjs';
	import githubLight from 'shiki/themes/github-light-default.mjs';

	/**
	 * Synchronous highlighters (JS regex engine, no WASM) so the code renderer
	 * stays sync and SvelteMarkdown's streaming diff path stays enabled. One
	 * per theme; unknown languages fall back to escaped plaintext.
	 */
	const langs = [
		bash, c, cpp, css, go, html, java, javascript, json,
		markdown, python, rust, sql, typescript, yaml
	];
	const highlighters = {
		dark: createShikiHighlighter({ langs, themes: [githubDark] }),
		light: createShikiHighlighter({ langs, themes: [githubLight] })
	};
</script>

<script lang="ts">
	import { ShikiCode } from '@humanspeak/svelte-markdown/extensions/shiki';
	import { mode } from 'mode-watcher';

	/** Receives the same token props SvelteMarkdown passes any `code` renderer. */
	let { lang, text }: { lang: string; text: string } = $props();
</script>

<ShikiCode {lang} {text} highlighter={highlighters[mode.current === 'light' ? 'light' : 'dark']} />
