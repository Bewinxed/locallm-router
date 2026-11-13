import { createOpenAI } from '@ai-sdk/openai';
import { convertToModelMessages, streamText, type UIMessage } from 'ai';
import { proxyUrl } from '$lib/server/manager';
import type { RequestHandler } from './$types';

const openai = createOpenAI({ baseURL: proxyUrl('/v1'), apiKey: 'not-needed' });

/** A cold model may need to download and warm up before the first token. */
const FIRST_TOKEN_TIMEOUT_MS = 300_000;

export interface ChatMetadata {
	inputTokens?: number;
	outputTokens?: number;
	totalTokens?: number;
	finishReason?: string;
	durationMs?: number;
	tokensPerSecond?: number;
}

export const POST: RequestHandler = async ({ request }) => {
	const {
		messages,
		model,
		system,
		temperature,
		topP,
		topK,
		maxTokens,
		presencePenalty,
		frequencyPenalty
	} = (await request.json()) as {
		messages: UIMessage[];
		model?: string;
		system?: string;
		temperature?: number;
		topP?: number;
		topK?: number;
		maxTokens?: number;
		presencePenalty?: number;
		frequencyPenalty?: number;
	};

	if (!model) {
		return Response.json({ error: 'No model specified' }, { status: 400 });
	}

	const startedAt = Date.now();

	// openai() defaults to the Responses API (/v1/responses), which llama-server
	// and vLLM do not implement. .chat() pins it to Chat Completions.
	const result = streamText({
		model: openai.chat(model),
		system: system || undefined,
		messages: await convertToModelMessages(messages),
		temperature,
		topP,
		topK,
		maxOutputTokens: maxTokens,
		presencePenalty,
		frequencyPenalty,
		abortSignal: AbortSignal.timeout(FIRST_TOKEN_TIMEOUT_MS)
	});

	return result.toUIMessageStreamResponse({
		messageMetadata: ({ part }): ChatMetadata | undefined => {
			if (part.type !== 'finish') return;
			const durationMs = Date.now() - startedAt;
			const outputTokens = part.totalUsage?.outputTokens;
			return {
				inputTokens: part.totalUsage?.inputTokens,
				outputTokens,
				totalTokens: part.totalUsage?.totalTokens,
				finishReason: part.finishReason,
				durationMs,
				tokensPerSecond:
					outputTokens && durationMs > 0
						? Number((outputTokens / (durationMs / 1000)).toFixed(1))
						: undefined
			};
		},
		onError: (error) => (error instanceof Error ? error.message : 'Inference failed')
	});
};
