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
	"encoding/json"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func resp(prompt, completion, total int64) *ModelResponse {
	return &ModelResponse{Usage: TokenUsage{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: total}}
}

func TestSumUsage(t *testing.T) {
	tests := []struct {
		name  string
		resps []*ModelResponse
		want  TokenUsage
	}{
		{
			name:  "no responses",
			resps: nil,
			want:  TokenUsage{},
		},
		{
			name:  "single response",
			resps: []*ModelResponse{resp(10, 5, 15)},
			want:  TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		{
			name:  "multiple responses summed",
			resps: []*ModelResponse{resp(10, 5, 15), resp(20, 8, 28), resp(1, 1, 2)},
			want:  TokenUsage{PromptTokens: 31, CompletionTokens: 14, TotalTokens: 45},
		},
		{
			name:  "nil entries skipped",
			resps: []*ModelResponse{nil, resp(10, 5, 15), nil},
			want:  TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SumUsage(tt.resps...); got != tt.want {
				t.Errorf("SumUsage() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestTokenUsageAdd_Accumulates(t *testing.T) {
	// Simulates a multi-round looper accumulating across rounds.
	var total TokenUsage
	total = total.Add(resp(10, 5, 15), resp(20, 5, 25))
	total = total.Add(resp(4, 1, 5))

	want := TokenUsage{PromptTokens: 34, CompletionTokens: 11, TotalTokens: 45}
	if total != want {
		t.Errorf("accumulated usage = %+v, want %+v", total, want)
	}
}

func TestTokenUsageAdd_DoesNotMutateReceiver(t *testing.T) {
	base := TokenUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	_ = base.Add(resp(10, 10, 20))
	if (base != TokenUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}) {
		t.Errorf("Add mutated the receiver: %+v", base)
	}
}

func TestTokenUsageAddRejectsNegativeProviderCounters(t *testing.T) {
	usage := SumUsage(&ModelResponse{Usage: TokenUsage{
		PromptTokens:     -10,
		CompletionTokens: 4,
		TotalTokens:      -6,
	}})

	if usage.isValid() || usage.PromptTokens != 0 || usage.CompletionTokens != 0 || usage.TotalTokens != 0 {
		t.Fatalf("SumUsage() = %+v, want the complete provider usage rejected", usage)
	}
}

func TestTokenUsageAddKeepsInvalidAggregatePoisoned(t *testing.T) {
	usage := SumUsage(
		resp(5, 3, 8),
		&ModelResponse{Usage: TokenUsage{PromptTokens: -1, CompletionTokens: 4, TotalTokens: 3}},
		resp(12, 8, 20),
	)

	if usage.isValid() || usage.PromptTokens != 0 || usage.CompletionTokens != 0 || usage.TotalTokens != 0 {
		t.Fatalf("SumUsage() = %+v, want one invalid response to poison the complete aggregate", usage)
	}
}

func TestTokenUsageAddRejectsAggregateOverflow(t *testing.T) {
	usage := TokenUsage{PromptTokens: math.MaxInt64, CompletionTokens: 1, TotalTokens: math.MaxInt64}.
		Add(resp(1, 1, 1))

	if usage.isValid() || usage.PromptTokens != 0 || usage.CompletionTokens != 0 || usage.TotalTokens != 0 {
		t.Fatalf("TokenUsage.Add() = %+v, want overflowed aggregate rejected", usage)
	}
}

func TestTokenUsageMap(t *testing.T) {
	u := TokenUsage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10}
	m := tokenUsageMapForTest(u)

	// Round-trip through JSON to confirm the OpenAI-compatible shape.
	body, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal usage map: %v", err)
	}
	var decoded TokenUsage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal usage map: %v", err)
	}
	if decoded != u {
		t.Errorf("usage map round-trip = %+v, want %+v", decoded, u)
	}
}

// TestParseNonStreamingResponse_PopulatesUsage verifies that usage is lifted
// from the backend completion into ModelResponse.Usage, which every looper then
// aggregates instead of emitting the legacy {0,0,0} block.
func TestParseNonStreamingResponse_PopulatesUsage(t *testing.T) {
	c := &Client{}
	body := []byte(`{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"model": "backend-model",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50}
	}`)

	resp, err := c.parseNonStreamingResponse(body, "logical-model")
	if err != nil {
		t.Fatalf("parseNonStreamingResponse failed: %v", err)
	}

	want := TokenUsage{PromptTokens: 42, CompletionTokens: 8, TotalTokens: 50}
	if resp.Usage != want {
		t.Errorf("ModelResponse.Usage = %+v, want %+v", resp.Usage, want)
	}
}

func TestParseNonStreamingResponseRejectsNegativeProviderCounters(t *testing.T) {
	c := &Client{}
	body := []byte(`{
		"id": "chatcmpl-negative",
		"object": "chat.completion",
		"model": "backend-model",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": -13}
	}`)

	if _, err := c.parseNonStreamingResponse(body, "logical-model"); err == nil {
		t.Fatal("negative provider usage was accepted")
	}
}

func TestParseNonStreamingResponseTreatsMissingUsageAsUnknown(t *testing.T) {
	c := &Client{}
	body := []byte(`{
		"id": "chatcmpl-missing-usage",
		"object": "chat.completion",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
	}`)

	resp, err := c.parseNonStreamingResponse(body, "logical-model")
	if err != nil {
		t.Fatalf("parseNonStreamingResponse failed: %v", err)
	}
	if resp.Usage.isValid() {
		t.Fatalf("ModelResponse.Usage = %+v, want unknown", resp.Usage)
	}
	if tokenUsageMapForTest(resp.Usage) != nil {
		t.Fatalf("unknown usage must serialize as null, got %#v", tokenUsageMapForTest(resp.Usage))
	}
}

func TestExternalTokenUsageConstructionPreservesAuthority(t *testing.T) {
	actual := NewActualTokenUsage(7, 3, 10)
	if !actual.Authoritative() || tokenUsageMapForTest(actual)["total_tokens"] != int64(10) {
		t.Fatalf("actual usage = %+v, map = %#v", actual, tokenUsageMapForTest(actual))
	}
	invalid := NewActualTokenUsage(7, -1, 6)
	if invalid.Authoritative() || tokenUsageMapForTest(invalid) != nil {
		t.Fatalf("invalid actual usage = %+v, map = %#v", invalid, tokenUsageMapForTest(invalid))
	}
	unknown := UnknownTokenUsage()
	if unknown.Authoritative() || tokenUsageMapForTest(unknown) != nil {
		t.Fatalf("unknown usage = %+v, map = %#v", unknown, tokenUsageMapForTest(unknown))
	}
}

func TestParseNonStreamingResponseRejectsPartialUsage(t *testing.T) {
	for _, test := range []struct {
		name  string
		usage string
	}{
		{name: "prompt only", usage: `{"prompt_tokens":4}`},
		{name: "completion only", usage: `{"completion_tokens":3}`},
		{name: "total only", usage: `{"total_tokens":7}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"id":"chatcmpl-partial","model":"backend-model","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":` + test.usage + `}`)
			if _, err := (&Client{}).parseNonStreamingResponse(body, "logical-model"); err == nil {
				t.Fatal("partial provider usage was accepted")
			}
		})
	}
}

// TestBaseLooper_FormatJSONResponse_AggregatesUsage verifies that per-call
// usages are summed in the neutral response and its Chat rendering.
func TestBaseLooper_FormatJSONResponse_AggregatesUsage(t *testing.T) {
	l := NewBaseLooper(&config.LooperConfig{Endpoint: "http://localhost:8000"})

	agg := &AggregatedResponse{
		Models:          []string{"a", "b"},
		Responses:       []*ModelResponse{resp(100, 20, 120), resp(50, 10, 60)},
		CombinedContent: "combined",
		FinalModel:      "b",
	}

	out, err := l.formatJSONResponse(agg, agg.Models, 2)
	if err != nil {
		t.Fatalf("formatJSONResponse failed: %v", err)
	}

	wantUsage := TokenUsage{PromptTokens: 150, CompletionTokens: 30, TotalTokens: 180}
	if out.Usage != wantUsage {
		t.Errorf("Response.Usage = %+v, want %+v", out.Usage, wantUsage)
	}

	var parsed struct {
		Usage TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(wireResponseForTest(t, out), &parsed); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if parsed.Usage != wantUsage {
		t.Errorf("body usage = %+v, want %+v", parsed.Usage, wantUsage)
	}
}
