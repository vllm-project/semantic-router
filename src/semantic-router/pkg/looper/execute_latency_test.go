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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func latencyTestBackend(delay time.Duration) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-x",
			"object": "chat.completion",
			"model":  "stub-backend",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": "ok"},
					"finish_reason": "stop",
				},
			},
		})
	}))
}

// ExecuteWithLatency must record the wall-clock latency of the full looper
// execution on the returned Response, regardless of whether the underlying
// algorithm dispatches its model calls sequentially (BaseLooper) or
// concurrently (fusion, ratings, remom, workflows). It is the single
// instrumentation point for every Looper implementation, so BaseLooper here
// stands in for all of them (issue #2694).
func TestExecuteWithLatency_RecordsWallClockLatency(t *testing.T) {
	const delay = 40 * time.Millisecond
	server := latencyTestBackend(delay)
	defer server.Close()

	l := NewBaseLooper(&config.LooperConfig{Endpoint: server.URL})
	req := &Request{
		executionRequest: readLimitTestRequest(),
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
	}

	resp, err := ExecuteWithLatency(context.Background(), l, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// BaseLooper calls its two models sequentially, so the wall clock should
	// cover both sleeps. Lower-bound only, to avoid flakiness.
	want := (2 * delay).Milliseconds()
	if resp.LatencyMs < want {
		t.Errorf("LatencyMs = %d, want >= %d (2 sequential calls sleeping %v each)", resp.LatencyMs, want, delay)
	}
}

// ExecuteWithLatency must propagate the underlying Execute error unchanged
// (not mask it behind a latency-recording failure).
func TestExecuteWithLatency_PropagatesError(t *testing.T) {
	l := NewBaseLooper(&config.LooperConfig{Endpoint: "http://127.0.0.1:0"})
	req := &Request{
		executionRequest: readLimitTestRequest(),
		ModelRefs:        []config.ModelRef{{Model: "model-a"}},
	}

	_, err := ExecuteWithLatency(context.Background(), l, req)
	if err == nil {
		t.Fatal("expected error from unreachable endpoint, got nil")
	}
}

// nilResponseLooper is a Looper whose Execute returns (nil, nil), which the
// interface's contract does not forbid even though no real implementation
// does this today.
type nilResponseLooper struct{}

func (nilResponseLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	return nil, nil
}

// ExecuteWithLatency must not panic when Execute returns a nil Response
// alongside a nil error.
func TestExecuteWithLatency_NilResponseDoesNotPanic(t *testing.T) {
	req := &Request{executionRequest: readLimitTestRequest()}

	resp, err := ExecuteWithLatency(context.Background(), nilResponseLooper{}, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != nil {
		t.Fatalf("expected nil response, got %+v", resp)
	}
}
