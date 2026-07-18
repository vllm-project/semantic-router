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
	"testing"
)

// Streaming responses carry token counts in a final SSE chunk (emitted only
// when the request sets stream_options.include_usage). The looper client must
// both request and parse it, or streamed calls report {0,0,0}.

func TestParseStreamingUsage_ExtractsFinalUsageChunk(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n" +
		"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8,\"total_tokens\":20}}\n" +
		"data: [DONE]\n"

	got := parseStreamingUsage([]byte(body))

	want := TokenUsage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20}
	if got != want {
		t.Errorf("parseStreamingUsage() = %+v, want %+v", got, want)
	}
}

func TestParseStreamingUsage_NoUsageChunkReturnsZero(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n" +
		"data: [DONE]\n"

	if got := parseStreamingUsage([]byte(body)); got != (TokenUsage{}) {
		t.Errorf("parseStreamingUsage() = %+v, want zero", got)
	}
}

func TestParseStreamingUsage_IgnoresNullUsageChunks(t *testing.T) {
	body := "data: {\"usage\":null,\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n" +
		"data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n" +
		"data: [DONE]\n"

	if got := parseStreamingUsage([]byte(body)); got.TotalTokens != 8 {
		t.Errorf("TotalTokens = %d, want 8", got.TotalTokens)
	}
}

func TestParseStreamingResponse_PopulatesUsage(t *testing.T) {
	c := &Client{}
	body := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n" +
		"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8,\"total_tokens\":20}}\n" +
		"data: [DONE]\n")

	resp, err := c.parseStreamingResponse(body, "model-a")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Usage.TotalTokens != 20 {
		t.Errorf("resp.Usage = %+v, want total 20", resp.Usage)
	}
}

func TestSetStreamParam_StreamingRequestsIncludeUsage(t *testing.T) {
	out, err := setStreamParam([]byte(`{"model":"x"}`), true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	so, ok := m["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("stream_options missing: %s", out)
	}
	if so["include_usage"] != true {
		t.Errorf("include_usage = %v, want true", so["include_usage"])
	}
}

func TestSetStreamParam_NonStreamingDropsStreamOptions(t *testing.T) {
	out, err := setStreamParam([]byte(`{"model":"x","stream_options":{"include_usage":true}}`), false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := m["stream_options"]; ok {
		t.Errorf("stream_options should be dropped when not streaming: %s", out)
	}
}
