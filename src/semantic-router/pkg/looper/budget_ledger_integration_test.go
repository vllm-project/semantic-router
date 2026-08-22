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
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// budgetTestBackend returns a server that always answers with a fixed usage
// block and counts how many requests it received, so tests can assert a
// budget stopped escalation before every possible call was made.
func budgetTestBackend(promptTokens, completionTokens int) (*httptest.Server, *int32) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-budget",
			"object": "chat.completion",
			"model":  "stub-backend",
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "ok"},
				"finish_reason": "stop",
			}},
			"usage": map[string]interface{}{
				"prompt_tokens":     promptTokens,
				"completion_tokens": completionTokens,
				"total_tokens":      promptTokens + completionTokens,
			},
		})
	}))
	return server, &calls
}

// ReMoM's default schedule ([]int{4}) plus the appended synthesis round
// becomes two rounds; a budget that's exhausted by round one's usage must
// stop the loop in runReMoMSchedule (remom.go) before the synthesis round
// dispatches, deterministically, regardless of on_error. This is the
// integration-level proof that CheckBudget/RecordBudgetUsageForResponses
// actually stop a real multi-round execution, not just the unit-level ledger
// arithmetic covered by TestBudgetLedgerExhaustionReasons.
func TestReMoMStopsEscalatingWhenBudgetExhausted(t *testing.T) {
	server, calls := budgetTestBackend(100, 50)
	defer server.Close()

	budget := &ComputeBudget{MaxTotalTokens: 150}
	req := &Request{
		OriginalRequest: readLimitTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "model-a"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmReMoM,
			ReMoM: &config.ReMoMAlgorithmConfig{
				BreadthSchedule: []int{1},
				OnError:         "skip",
			},
		},
		Budget: budget,
		Ledger: NewBudgetLedger(budget),
	}

	l := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL})
	resp, err := l.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected a non-nil response")
	}

	// Round one (breadth 1) consumes exactly 150 tokens, meeting
	// MaxTotalTokens=150 and tripping the ledger before the synthesis round
	// (schedule's appended second entry) can dispatch.
	if got := atomic.LoadInt32(calls); got != 1 {
		t.Fatalf("backend received %d calls, want exactly 1 (round two must be skipped once the budget is exhausted)", got)
	}
	reason, exhausted := req.Ledger.Exhausted()
	if !exhausted || reason != BudgetReasonTotalTokens {
		t.Fatalf("ledger reason = %q, exhausted = %v, want total_tokens_exhausted", reason, exhausted)
	}
}

func TestReMoMRunsFullScheduleWhenBudgetNotExhausted(t *testing.T) {
	server, calls := budgetTestBackend(10, 5)
	defer server.Close()

	budget := &ComputeBudget{MaxTotalTokens: 1_000_000}
	req := &Request{
		OriginalRequest: readLimitTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "model-a"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmReMoM,
			ReMoM: &config.ReMoMAlgorithmConfig{
				BreadthSchedule: []int{1},
				OnError:         "skip",
			},
		},
		Budget: budget,
		Ledger: NewBudgetLedger(budget),
	}

	l := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL})
	if _, err := l.Execute(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Schedule is [1, 1] (breadth 1 + the appended synthesis round): a
	// generous budget must let both rounds dispatch.
	if got := atomic.LoadInt32(calls); got != 2 {
		t.Fatalf("backend received %d calls, want exactly 2 (both rounds should run under a generous budget)", got)
	}
}
