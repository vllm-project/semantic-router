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
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestConnectorClientPreservesEndpointAndCallHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if got := request.URL.Path; got != "/v1/chat/completions" {
			t.Errorf("path = %q, want /v1/chat/completions", got)
		}
		wantHeaders := map[string]string{
			"Authorization":            "Bearer secret-a",
			"X-Static":                 "static-a",
			headers.VSRLooperDecision:  "decision-a",
			headers.VSRLooperIteration: "2",
			headers.VSRFusionDepth:     "1",
			headers.VSRLooperRequest:   "true",
		}
		for name, want := range wantHeaders {
			if got := request.Header.Get(name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-connector",
			"object":  "chat.completion",
			"created": 1,
			"model":   "backend-model",
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "ok"},
				"finish_reason": "stop",
			}},
		})
	}))
	defer server.Close()

	client, err := NewConnectorClient(&config.LooperConfig{
		Endpoint: server.URL + "/v1/chat/completions",
		Headers:  map[string]string{"X-Static": "static-a"},
	})
	if err != nil {
		t.Fatalf("NewConnectorClient() error = %v", err)
	}
	defer client.Close()

	response, err := client.CallModelWithOptions(
		context.Background(),
		openai.ChatCompletionNewParams{},
		ModelTarget{Name: "model-a", AccessKey: "secret-a"},
		CallOptions{
			DecisionName: "decision-a",
			Iteration:    2,
			FusionDepth:  1,
		},
	)
	if err != nil {
		t.Fatalf("CallModelWithOptions() error = %v", err)
	}
	if response.Content != "ok" || response.Model != "model-a" {
		t.Fatalf("response = content %q model %q", response.Content, response.Model)
	}
}

func TestConnectorClientDoesNotRetryGenerativeCalls(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		http.Error(w, "provider detail must stay hidden", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client, err := NewConnectorClient(&config.LooperConfig{
		Endpoint:   server.URL + "/v1/chat/completions",
		RetryCount: 5,
	})
	if err != nil {
		t.Fatalf("NewConnectorClient() error = %v", err)
	}
	defer client.Close()

	_, err = client.CallModelWithOptions(
		context.Background(),
		openai.ChatCompletionNewParams{},
		ModelTarget{Name: "model-a"},
		CallOptions{Iteration: 1},
	)
	if err == nil {
		t.Fatal("CallModelWithOptions() error = nil")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
	if strings.Contains(err.Error(), "provider detail") {
		t.Fatalf("error exposed provider body: %v", err)
	}
}

func TestConnectorClientRejectsNoContentStreamingResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client, err := NewConnectorClient(&config.LooperConfig{Endpoint: server.URL})
	if err != nil {
		t.Fatalf("NewConnectorClient() error = %v", err)
	}
	defer client.Close()

	_, err = client.CallModelWithOptions(
		context.Background(),
		openai.ChatCompletionNewParams{},
		ModelTarget{Name: "model-a"},
		CallOptions{Iteration: 1, Mode: ResponseSSE},
	)
	if err == nil {
		t.Fatal("CallModelWithOptions() error = nil")
	}
	if !strings.Contains(err.Error(), "status 204") {
		t.Fatalf("CallModelWithOptions() error = %q, want status 204", err)
	}
}
