/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func wireResponseForTest(t testing.TB, response *Response) []byte {
	t.Helper()
	if response == nil || response.Semantic == nil {
		t.Fatal("looper response has no neutral response")
	}
	engine := protocolcodec.NewBuiltinEngine()
	if response.Streaming {
		semantic := cloneSemanticResponse(*response.Semantic)
		if !response.IncludeUsage {
			semantic.Usage = llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
		}
		body, _, err := engine.EncodeResponseStream(
			llmprotocol.OpenAIChatV1,
			semantic,
			llmprotocol.StreamContext{PublicModel: response.Model},
		)
		if err != nil {
			t.Fatalf("encode neutral looper stream: %v", err)
		}
		return body
	}
	encoded, err := engine.EncodeResponse(
		llmprotocol.OpenAIChatV1,
		*response.Semantic,
		llmprotocol.Envelope{},
	)
	if err != nil {
		t.Fatalf("encode neutral looper response: %v", err)
	}
	return encoded.Body
}

func responseContentTypeForTest(response *Response) string {
	if response != nil && response.Streaming {
		return "text/event-stream"
	}
	return "application/json"
}

func semanticTextForTest(t testing.TB, response *Response) string {
	t.Helper()
	if response == nil || response.Semantic == nil {
		t.Fatal("looper response has no neutral response")
	}
	text, _, _ := semanticResponseText(response.Semantic)
	return text
}

func modelResponseFromWireForTest(
	t testing.TB,
	body []byte,
	model string,
	streaming bool,
) *ModelResponse {
	t.Helper()
	engine := protocolcodec.NewBuiltinEngine()
	var semantic llmprotocol.Response
	var err error
	if streaming {
		semantic, _, err = engine.DecodeResponseStream(
			llmprotocol.OpenAIChatV1,
			body,
			llmprotocol.StreamContext{PublicModel: model},
		)
	} else {
		semantic, _, _, err = engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
	}
	if err != nil {
		t.Fatalf("decode model response fixture: %v", err)
	}
	neutral := semanticModelResponse(semantic, model)
	content, reasoning, hasToolCalls := semanticResponseText(neutral)
	return &ModelResponse{
		Raw: body, Semantic: neutral, Content: content, ReasoningContent: reasoning,
		Model: model, HasToolCalls: hasToolCalls, IsStreaming: streaming,
		Usage: tokenUsageFromSemantic(neutral.Usage),
	}
}

func parseStreamingUsage(body []byte) TokenUsage {
	response, _, err := protocolcodec.NewBuiltinEngine().DecodeResponseStream(
		llmprotocol.OpenAIChatV1,
		body,
		llmprotocol.StreamContext{},
	)
	if err != nil {
		return unknownTokenUsage()
	}
	return tokenUsageFromSemantic(response.Usage)
}

func tokenUsageMapForTest(usage TokenUsage) map[string]interface{} {
	if !usage.isValid() {
		return nil
	}
	return map[string]interface{}{
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
	}
}
