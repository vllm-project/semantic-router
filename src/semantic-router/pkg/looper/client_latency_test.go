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

// CallModel must record the wall-clock latency of the upstream round-trip on
// the response so per-candidate overhead of Looper-family algorithms is
// measurable. A backend that sleeps a known duration proves the field carries
// real elapsed time (lower-bound assertion only, to avoid flakiness).
func TestCallModel_RecordsLatency(t *testing.T) {
	const delay = 40 * time.Millisecond
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	defer server.Close()

	c := NewClient(&config.LooperConfig{Endpoint: server.URL})

	resp, err := c.CallModel(context.Background(), readLimitTestRequest(), "model-a", false, 1, nil, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.LatencyMs < 30 {
		t.Errorf("LatencyMs = %d, want >= 30 (backend slept %v)", resp.LatencyMs, delay)
	}
}
