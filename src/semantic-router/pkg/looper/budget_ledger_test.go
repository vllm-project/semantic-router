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
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewBudgetLedgerReturnsNilForNilOrZeroBudget(t *testing.T) {
	if l := NewBudgetLedger(nil); l != nil {
		t.Fatalf("nil budget should produce a nil ledger, got %+v", l)
	}
	if l := NewBudgetLedger(&ComputeBudget{}); l != nil {
		t.Fatalf("all-zero budget should produce a nil ledger, got %+v", l)
	}
}

func TestBudgetLedgerNilSafety(t *testing.T) {
	var ledger *BudgetLedger
	if reason, exhausted := ledger.Exhausted(); exhausted || reason != BudgetReasonNone {
		t.Fatalf("nil ledger should never be exhausted, got (%q, %v)", reason, exhausted)
	}
	ledger.Record(TokenUsage{PromptTokens: 100}, 1.0) // must not panic
}

func TestBudgetLedgerExhaustionReasons(t *testing.T) {
	cases := []struct {
		name   string
		budget ComputeBudget
		usage  TokenUsage
		cost   float64
		want   BudgetExhaustionReason
	}{
		{"prompt tokens", ComputeBudget{MaxPromptTokens: 100}, TokenUsage{PromptTokens: 100}, 0, BudgetReasonPromptTokens},
		{"completion tokens", ComputeBudget{MaxCompletionTokens: 50}, TokenUsage{CompletionTokens: 50}, 0, BudgetReasonCompletionTokens},
		{"total tokens", ComputeBudget{MaxTotalTokens: 10}, TokenUsage{TotalTokens: 10}, 0, BudgetReasonTotalTokens},
		{"estimated cost", ComputeBudget{MaxEstimatedCost: 0.5}, TokenUsage{}, 0.5, BudgetReasonEstimatedCost},
		{"under every limit", ComputeBudget{MaxPromptTokens: 1000, MaxEstimatedCost: 10}, TokenUsage{PromptTokens: 1}, 0.01, BudgetReasonNone},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ledger := NewBudgetLedger(&tc.budget)
			if ledger == nil {
				t.Fatalf("expected a non-nil ledger for budget %+v", tc.budget)
			}
			ledger.Record(tc.usage, tc.cost)
			reason, exhausted := ledger.Exhausted()
			if reason != tc.want {
				t.Fatalf("reason = %q, want %q", reason, tc.want)
			}
			if exhausted != (tc.want != BudgetReasonNone) {
				t.Fatalf("exhausted = %v, want %v", exhausted, tc.want != BudgetReasonNone)
			}
		})
	}
}

func TestBudgetLedgerFirstReasonSticks(t *testing.T) {
	ledger := NewBudgetLedger(&ComputeBudget{MaxPromptTokens: 10, MaxCompletionTokens: 10})
	ledger.Record(TokenUsage{PromptTokens: 10}, 0)
	ledger.Record(TokenUsage{CompletionTokens: 10}, 0)
	reason, exhausted := ledger.Exhausted()
	if !exhausted || reason != BudgetReasonPromptTokens {
		t.Fatalf("reason = %q, exhausted = %v, want the first tripped reason (%q) to stick", reason, exhausted, BudgetReasonPromptTokens)
	}
}

func TestBudgetLedgerRecordConcurrentAccess(t *testing.T) {
	ledger := NewBudgetLedger(&ComputeBudget{MaxTotalTokens: 1_000_000})
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ledger.Record(TokenUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}, 0.001)
		}()
	}
	wg.Wait()
	if reason, exhausted := ledger.Exhausted(); exhausted {
		t.Fatalf("ledger should not be exhausted yet, got reason %q", reason)
	}
}

func TestCheckBudgetNilSafety(t *testing.T) {
	if stop, reason := CheckBudget(nil); stop || reason != BudgetReasonNone {
		t.Fatalf("nil request should never stop, got (%v, %q)", stop, reason)
	}
	req := &Request{}
	if stop, reason := CheckBudget(req); stop || reason != BudgetReasonNone {
		t.Fatalf("request with no ledger should never stop, got (%v, %q)", stop, reason)
	}
}

func TestCheckBudgetReflectsExhaustion(t *testing.T) {
	budget := &ComputeBudget{MaxPromptTokens: 10}
	req := &Request{Budget: budget, Ledger: NewBudgetLedger(budget)}
	req.Ledger.Record(TokenUsage{PromptTokens: 10}, 0)
	stop, reason := CheckBudget(req)
	if !stop || reason != BudgetReasonPromptTokens {
		t.Fatalf("stop = %v, reason = %q, want exhausted with prompt_tokens_exhausted", stop, reason)
	}
}

func TestRecordBudgetUsageEstimatesCostFromModelParams(t *testing.T) {
	budget := &ComputeBudget{MaxEstimatedCost: 1.0}
	req := &Request{
		Budget: budget,
		Ledger: NewBudgetLedger(budget),
		ModelParams: map[string]config.ModelParams{
			"gpt-test": {Pricing: config.ModelPricing{PromptPer1M: 1_000_000, CompletionPer1M: 1_000_000}},
		},
	}
	// 1 prompt token + 1 completion token at $1/token (1,000,000 per 1M) = $2, over the $1 budget.
	RecordBudgetUsage(req, TokenUsage{PromptTokens: 1, CompletionTokens: 1}, "gpt-test")
	if reason, exhausted := req.Ledger.Exhausted(); !exhausted || reason != BudgetReasonEstimatedCost {
		t.Fatalf("reason = %q, exhausted = %v, want estimated_cost_exhausted", reason, exhausted)
	}
}

func TestRecordBudgetUsageNoPricingTracksTokensOnlyWithZeroCost(t *testing.T) {
	budget := &ComputeBudget{MaxEstimatedCost: 1.0}
	req := &Request{Budget: budget, Ledger: NewBudgetLedger(budget)}
	RecordBudgetUsage(req, TokenUsage{PromptTokens: 1000}, "unpriced-model")
	if reason, exhausted := req.Ledger.Exhausted(); exhausted {
		t.Fatalf("missing pricing should not trip the cost budget, got reason %q", reason)
	}
}

func TestRecordBudgetUsageForResponsesSkipsNilAndUsesEachModel(t *testing.T) {
	budget := &ComputeBudget{MaxTotalTokens: 5}
	req := &Request{Budget: budget, Ledger: NewBudgetLedger(budget)}
	responses := []*ModelResponse{
		{Model: "a", Usage: TokenUsage{TotalTokens: 2}},
		nil,
		{Model: "b", Usage: TokenUsage{TotalTokens: 3}},
	}
	RecordBudgetUsageForResponses(req, responses)
	if reason, exhausted := req.Ledger.Exhausted(); !exhausted || reason != BudgetReasonTotalTokens {
		t.Fatalf("reason = %q, exhausted = %v, want total_tokens_exhausted after summing 2+3", reason, exhausted)
	}
}
