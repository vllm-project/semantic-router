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
	"testing"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestCallModelWithOptionsUsesRequestScopedMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get(headers.VSRLooperDecision); got != "decision-a" {
			t.Errorf("%s = %q, want decision-a", headers.VSRLooperDecision, got)
		}
		if got := request.Header.Get(headers.VSRLooperIteration); got != "2" {
			t.Errorf("%s = %q, want 2", headers.VSRLooperIteration, got)
		}
		if got := request.Header.Get(headers.VSRFusionDepth); got != "1" {
			t.Errorf("%s = %q, want 1", headers.VSRFusionDepth, got)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer secret-a" {
			t.Errorf("Authorization = %q, want Bearer secret-a", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-options",
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

	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	request := openai.ChatCompletionNewParams{
		Model:    "original-model",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
	response, err := client.CallModelWithOptions(
		context.Background(),
		request,
		ModelTarget{Name: "model-a", AccessKey: "secret-a"},
		CallOptions{
			DecisionName: "decision-a",
			Iteration:    2,
			FusionDepth:  1,
			Mode:         ResponseJSON,
		},
	)
	if err != nil {
		t.Fatalf("CallModelWithOptions() error = %v", err)
	}
	if response.Content != "ok" || response.Model != "model-a" {
		t.Fatalf("response = content %q model %q", response.Content, response.Model)
	}
	if request.Model != "original-model" {
		t.Fatalf("CallModelWithOptions() mutated caller model to %q", request.Model)
	}
}

func TestCallModelWithOptionsIsolatesConcurrentMetadata(t *testing.T) {
	type observedRequest struct {
		model         string
		decision      string
		iteration     string
		fusionDepth   string
		authorization string
	}

	observed := make(chan observedRequest, 2)
	release := make(chan struct{})
	releaseRequests := func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}
	defer releaseRequests()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		observed <- observedRequest{
			model:         body.Model,
			decision:      request.Header.Get(headers.VSRLooperDecision),
			iteration:     request.Header.Get(headers.VSRLooperIteration),
			fusionDepth:   request.Header.Get(headers.VSRFusionDepth),
			authorization: request.Header.Get("Authorization"),
		}
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()

	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	type call struct {
		target  ModelTarget
		options CallOptions
	}
	calls := []call{
		{
			target:  ModelTarget{Name: "model-a", AccessKey: "secret-a"},
			options: CallOptions{DecisionName: "decision-a", Iteration: 1, Mode: ResponseJSON},
		},
		{
			target:  ModelTarget{Name: "model-b", AccessKey: "secret-b"},
			options: CallOptions{DecisionName: "decision-b", Iteration: 2, FusionDepth: 1, Mode: ResponseJSON},
		},
	}

	errs := make(chan error, len(calls))
	for _, call := range calls {
		call := call
		go func() {
			_, err := client.CallModelWithOptions(
				context.Background(),
				openai.ChatCompletionNewParams{},
				call.target,
				call.options,
			)
			errs <- err
		}()
	}

	got := make(map[string]observedRequest, len(calls))
	for range calls {
		select {
		case request := <-observed:
			got[request.model] = request
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for concurrent Looper requests")
		}
	}
	releaseRequests()

	want := map[string]observedRequest{
		"model-a": {model: "model-a", decision: "decision-a", iteration: "1", authorization: "Bearer secret-a"},
		"model-b": {model: "model-b", decision: "decision-b", iteration: "2", fusionDepth: "1", authorization: "Bearer secret-b"},
	}
	for model, wantRequest := range want {
		if gotRequest := got[model]; gotRequest != wantRequest {
			t.Errorf("request for %s = %+v, want %+v", model, gotRequest, wantRequest)
		}
	}
	for range calls {
		if err := <-errs; err != nil {
			t.Errorf("CallModelWithOptions() error = %v", err)
		}
	}
}

func TestCallModelWithOptionsValidatesRequiredFields(t *testing.T) {
	client := NewClient(&config.LooperConfig{Endpoint: "http://unused.invalid"})
	request := openai.ChatCompletionNewParams{}

	tests := []struct {
		name    string
		target  ModelTarget
		options CallOptions
	}{
		{
			name:    "missing target",
			options: CallOptions{Iteration: 1, Mode: ResponseJSON},
		},
		{
			name:    "missing iteration",
			target:  ModelTarget{Name: "model-a"},
			options: CallOptions{Mode: ResponseJSON},
		},
		{
			name:    "unsupported mode",
			target:  ModelTarget{Name: "model-a"},
			options: CallOptions{Iteration: 1, Mode: ResponseMode(99)},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := client.CallModelWithOptions(
				context.Background(), request, test.target, test.options,
			); err == nil {
				t.Fatal("CallModelWithOptions() error = nil")
			}
		})
	}
}
