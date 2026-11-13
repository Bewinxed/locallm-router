<script lang="ts">
	import { Label } from '$lib/components/ui/label';
	import { Slider } from '$lib/components/ui/slider';

	let {
		label,
		value = $bindable(),
		min,
		max,
		step,
		decimals = 0
	}: {
		label: string;
		value: number;
		min: number;
		max: number;
		step: number;
		decimals?: number;
	} = $props();
</script>

<div class="flex flex-col gap-2">
	<div class="flex items-center justify-between">
		<Label class="text-xs text-muted-foreground">{label}</Label>
		<span class="font-mono text-xs tabular-nums">{value.toFixed(decimals)}</span>
	</div>
	<Slider type="single" bind:value {min} {max} {step} />
</div>
