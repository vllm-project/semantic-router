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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func readLimitTestRequest() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
}

// TestCallModel_RejectsOversizedResponseBody drives the full CallModel path
// against a backend that returns a *valid* but oversized completion. Without a
// read ceiling the body parses cleanly (no error); with the ceiling CallModel
// must reject it rather than buffer the whole thing into memory.
func TestCallModel_RejectsOversizedResponseBody(t *testing.T) {
	bigContent := strings.Repeat("a", 2*1024*1024) // 2 MiB, valid JSON, over the 1 MiB cap
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-big",
			"object": "chat.completion",
			"model":  "stub-backend",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": bigContent},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	c := NewClient(&config.LooperConfig{Endpoint: server.URL, MaxResponseBytesMB: 1})

	if _, err := c.CallModel(context.Background(), readLimitTestRequest(), "model-a", false, 1, nil, ""); err == nil {
		t.Fatal("expected CallModel to reject a response body larger than the configured cap, got nil error")
	}
}

// TestCallModel_AcceptsResponseWithinCap is the positive control: a normal,
// within-cap completion must still succeed after the ceiling is added.
func TestCallModel_AcceptsResponseWithinCap(t *testing.T) {
	server, _ := newUsageBackend(t, 1, 2, 3)
	defer server.Close()

	c := NewClient(&config.LooperConfig{Endpoint: server.URL})

	resp, err := c.CallModel(context.Background(), readLimitTestRequest(), "model-a", false, 1, nil, "")
	if err != nil {
		t.Fatalf("unexpected error for a within-cap response: %v", err)
	}
	if resp.Content != "stub answer" {
		t.Errorf("resp.Content = %q, want %q", resp.Content, "stub answer")
	}
}

// TestCallModel_RejectsOversizedStreamingBody proves the ceiling applies to the
// streaming path too, not just non-streaming JSON.
func TestCallModel_RejectsOversizedStreamingBody(t *testing.T) {
	oversized := strings.Repeat("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n", 60000) // >1 MiB of SSE frames
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(oversized))
	}))
	defer server.Close()

	c := NewClient(&config.LooperConfig{Endpoint: server.URL, MaxResponseBytesMB: 1})

	if _, err := c.CallModel(context.Background(), readLimitTestRequest(), "model-a", true, 1, nil, ""); err == nil {
		t.Fatal("expected CallModel to reject an oversized streaming body, got nil error")
	}
}

// TestCallModel_TruncatesOversizedErrorBody proves a non-2xx response with a
// huge body still surfaces the status code, is bounded (not fully buffered),
// and is marked as truncated for diagnostics.
func TestCallModel_TruncatesOversizedErrorBody(t *testing.T) {
	hugeErr := strings.Repeat("E", 64*1024) // 64 KiB, far over the 8 KiB error cap
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(hugeErr))
	}))
	defer server.Close()

	c := NewClient(&config.LooperConfig{Endpoint: server.URL})

	_, err := c.CallModel(context.Background(), readLimitTestRequest(), "model-a", false, 1, nil, "")
	if err == nil {
		t.Fatal("expected an error for a 500 response, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "500") {
		t.Errorf("error should surface the status code, got %q", msg)
	}
	if !strings.Contains(msg, "truncated") {
		t.Errorf("oversized error body should be marked truncated, got %q", msg)
	}
	if len(msg) > 16*1024 {
		t.Errorf("error message not bounded: %d bytes (error body was not truncated)", len(msg))
	}
}
