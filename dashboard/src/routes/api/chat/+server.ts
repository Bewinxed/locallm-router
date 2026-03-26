import { createOpenAI } from '@ai-sdk/openai';
import { streamText, type UIMessage, convertToModelMessages } from 'ai';
import type { RequestHandler } from './$types';

const openai = createOpenAI({
	baseURL: 'http://localhost:30000/v1',
	apiKey: 'not-needed',
	// Long timeout: model loading can take 30-60s on first request
	fetch: (url, init) => fetch(url, { ...init, signal: init?.signal ?? AbortSignal.timeout(300_000) })
});

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json();

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
	} = body as {
		messages: UIMessage[];
		model: string;
		system?: string;
		temperature?: number;
		topP?: number;
		topK?: number;
		maxTokens?: number;
		presencePenalty?: number;
		frequencyPenalty?: number;
	};

	if (!model) {
		return new Response(JSON.stringify({ error: 'No model specified' }), {
			status: 400,
			headers: { 'Content-Type': 'application/json' }
		});
	}

	const modelMessages = await convertToModelMessages(messages);

	// MUST use openai.chat() — openai() defaults to the Responses API (/v1/responses)
	// which llama-server/vLLM don't support. .chat() forces Chat Completions API.
	const result = streamText({
		model: openai.chat(model),
		system,
		messages: modelMessages,
		temperature,
		topP,
		topK,
		maxOutputTokens: maxTokens,
		presencePenalty,
		frequencyPenalty
	});

	return result.toUIMessageStreamResponse();
};
