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

package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDecodeResponseStreamAccumulatesBuiltinFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	response := llmprotocol.Response{
		Generation: 1,
		ID:         "response_1",
		Model:      "public/model",
		Output: []llmprotocol.OutputItem{{
			ID:   "item_1",
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentReasoning, Text: "consider"},
				{Kind: llmprotocol.ContentText, Text: "answer"},
			},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      availableStreamUsage(4, 2),
	}
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			assertDecodedBuiltinResponseStream(t, engine, format, response)
		})
	}
}

func assertDecodedBuiltinResponseStream(
	t *testing.T,
	engine *Engine,
	format llmprotocol.WireFormat,
	response llmprotocol.Response,
) {
	t.Helper()
	wire, _, err := engine.EncodeResponseStream(format, response, llmprotocol.StreamContext{PublicModel: response.Model})
	if err != nil {
		t.Fatalf("EncodeResponseStream() error = %v", err)
	}
	decoded, _, err := engine.DecodeResponseStream(format, wire, llmprotocol.StreamContext{PublicModel: response.Model})
	if err != nil {
		t.Fatalf("DecodeResponseStream() error = %v, wire=%s", err, wire)
	}
	if decoded.ID != response.ID || decoded.Model != response.Model ||
		decoded.StopReason != response.StopReason || len(decoded.Output) == 0 {
		t.Fatalf("decoded response = %+v", decoded)
	}
	reasoning, text := collectDecodedText(decoded.Output)
	if reasoning != "consider" || text != "answer" {
		t.Fatalf("decoded content reasoning=%q text=%q", reasoning, text)
	}
	if decoded.Usage.Total.Value == nil || *decoded.Usage.Total.Value != 6 {
		t.Fatalf("decoded usage = %+v", decoded.Usage)
	}
}

func collectDecodedText(output []llmprotocol.OutputItem) (string, string) {
	var reasoning, text string
	for _, item := range output {
		for _, content := range item.Content {
			switch content.Kind {
			case llmprotocol.ContentReasoning:
				reasoning += content.Text
			case llmprotocol.ContentText:
				text += content.Text
			}
		}
	}
	return reasoning, text
}

func TestDecodeResponseStreamPreservesToolCall(t *testing.T) {
	engine := NewBuiltinEngine()
	response := llmprotocol.Response{
		Generation: 1,
		ID:         "response_tool",
		Model:      "public/model",
		Output: []llmprotocol.OutputItem{{
			ID:   "item_tool",
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{
					ID: "call_1", Name: "lookup", Arguments: `{"query":"neutral"}`,
				},
			}},
		}},
		StopReason: llmprotocol.StopToolCall,
		Usage:      availableStreamUsage(3, 1),
	}
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			wire, _, err := engine.EncodeResponseStream(format, response, llmprotocol.StreamContext{})
			if err != nil {
				t.Fatal(err)
			}
			decoded, _, err := engine.DecodeResponseStream(format, wire, llmprotocol.StreamContext{})
			if err != nil {
				t.Fatal(err)
			}
			if len(decoded.Output) != 1 || len(decoded.Output[0].Content) != 1 ||
				decoded.Output[0].Content[0].ToolCall == nil {
				t.Fatalf("decoded tool response = %+v", decoded)
			}
			call := decoded.Output[0].Content[0].ToolCall
			if call.ID != "call_1" || call.Name != "lookup" || call.Arguments != `{"query":"neutral"}` {
				t.Fatalf("decoded tool call = %+v", call)
			}
		})
	}
}
