package sessiontelemetry

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func labelsMatchModelDomain(labels []*dto.LabelPair, model, domain string) bool {
	var hasModel, hasDomain bool
	for _, l := range labels {
		switch l.GetName() {
		case "model":
			if l.GetValue() == model {
				hasModel = true
			}
		case "domain":
			if l.GetValue() == domain {
				hasDomain = true
			}
		}
	}
	return hasModel && hasDomain
}

func histogramSampleCount(metricName, model, domain string) uint64 {
	mf, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mf {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_HISTOGRAM {
			continue
		}
		for _, m := range fam.GetMetric() {
			if !labelsMatchModelDomain(m.GetLabel(), model, domain) {
				continue
			}
			h := m.GetHistogram()
			if h == nil || h.SampleCount == nil {
				continue
			}
			return h.GetSampleCount()
		}
	}
	return 0
}

func TestRecordTurn_ChatCompletions_Cumulative(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	model := "m-test"
	domain := "unknown"

	p := TurnParams{
		RequestID:        "r1",
		Model:            model,
		Domain:           domain,
		PromptTokens:     10,
		CompletionTokens: 5,
		Chat: &ChatInput{
			UserID: "user_a",
			Messages: []ChatMessage{
				{Role: "user", Content: "thread-x"},
			},
		},
	}

	beforeP := histogramSampleCount("llm_session_turn_prompt_tokens", model, domain)
	beforeC := histogramSampleCount("llm_session_turn_completion_tokens", model, domain)

	RecordTurn(p)
	p.RequestID = "r2"
	p.PromptTokens = 3
	p.CompletionTokens = 2
	RecordTurn(p)

	afterP := histogramSampleCount("llm_session_turn_prompt_tokens", model, domain)
	afterC := histogramSampleCount("llm_session_turn_completion_tokens", model, domain)
	assert.Equal(t, uint64(2), afterP-beforeP)
	assert.Equal(t, uint64(2), afterC-beforeC)

	mu.Lock()
	require.Len(t, store, 1)
	var st *turnState
	for _, v := range store {
		st = v
		break
	}
	mu.Unlock()
	require.NotNil(t, st)
	assert.Equal(t, int64(13), st.cumulativePrompt)
	assert.Equal(t, int64(7), st.cumulativeCompletion)
}

func TestRecordTurn_ResponseAPI(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	RecordTurn(TurnParams{
		RequestID:        "r1",
		Model:            "m",
		Domain:           "unknown",
		PromptTokens:     1,
		CompletionTokens: 1,
		ResponseAPI: &ResponseAPIInput{
			ConversationID: "conv_1",
			HistoryLen:     2,
		},
	})
	mu.Lock()
	defer mu.Unlock()
	st, ok := store["respapi:conv_1"]
	require.True(t, ok)
	assert.Equal(t, int64(1), st.cumulativePrompt)
}

func TestEviction(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	orig := nowFn
	t.Cleanup(func() { nowFn = orig })

	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	nowFn = func() time.Time { return base }

	RecordTurn(TurnParams{
		RequestID:        "r1",
		Model:            "m",
		PromptTokens:     1,
		CompletionTokens: 1,
		Chat: &ChatInput{
			UserID:   "u1",
			Messages: []ChatMessage{{Role: "user", Content: "a"}},
		},
	})
	mu.Lock()
	assert.Len(t, store, 1)
	mu.Unlock()

	nowFn = func() time.Time { return base.Add(ttl + time.Minute) }
	RecordTurn(TurnParams{
		RequestID:        "r2",
		Model:            "m",
		PromptTokens:     1,
		CompletionTokens: 1,
		Chat: &ChatInput{
			UserID:   "u2",
			Messages: []ChatMessage{{Role: "user", Content: "b"}},
		},
	})
	mu.Lock()
	defer mu.Unlock()
	assert.Len(t, store, 1)
}

// =====================================================================
// PR 2 — per-turn pricing metadata and cumulative_cost
// =====================================================================

func labelHasCurrency(labels []*dto.LabelPair, currency string) bool {
	for _, l := range labels {
		if l.GetName() == "currency" && l.GetValue() == currency {
			return true
		}
	}
	return false
}

func histogramSampleCountCost(metricName, model, domain, currency string) uint64 {
	mf, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mf {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_HISTOGRAM {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := m.GetLabel()
			if !labelsMatchModelDomain(labels, model, domain) || !labelHasCurrency(labels, currency) {
				continue
			}
			h := m.GetHistogram()
			if h == nil || h.SampleCount == nil {
				continue
			}
			return h.GetSampleCount()
		}
	}
	return 0
}

func TestRecordTurn_ChatCompletions_PricingCostAccumulation(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	model := "gpt-cost-test"
	domain := "finance"

	pricing := TurnPricing{
		Currency:        "USD",
		PromptPer1M:     5.0,
		CompletionPer1M: 15.0,
	}

	baseParams := TurnParams{
		RequestID:        "r1",
		Model:            model,
		Domain:           domain,
		PromptTokens:     1000,
		CompletionTokens: 500,
		Pricing:          pricing,
		Chat: &ChatInput{
			UserID:   "user_cost",
			Messages: []ChatMessage{{Role: "user", Content: "cost-thread"}},
		},
	}

	beforeCost := histogramSampleCountCost("llm_session_turn_cost", model, domain, "USD")

	RecordTurn(baseParams)

	// turn 2: more tokens, same session
	baseParams.RequestID = "r2"
	baseParams.PromptTokens = 2000
	baseParams.CompletionTokens = 1000
	RecordTurn(baseParams)

	afterCost := histogramSampleCountCost("llm_session_turn_cost", model, domain, "USD")
	assert.Equal(t, uint64(2), afterCost-beforeCost, "expected two cost histogram observations")

	mu.Lock()
	require.Len(t, store, 1)
	var st *turnState
	for _, v := range store {
		st = v
		break
	}
	mu.Unlock()
	require.NotNil(t, st)

	// turn1 cost: (1000*5 + 500*15)/1e6 = 0.01250
	// turn2 cost: (2000*5 + 1000*15)/1e6 = 0.02500
	// cumulative: 0.03750
	const wantCumCost = (1000*5.0+500*15.0)/1_000_000.0 + (2000*5.0+1000*15.0)/1_000_000.0
	assert.InDelta(t, wantCumCost, st.cumulativeCost, 1e-9)
}

func TestRecordTurn_ResponseAPI_PricingCostAccumulation(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	model := "resp-cost-model"
	domain := "coding"

	pricing := TurnPricing{
		Currency:         "USD",
		PromptPer1M:      3.0,
		CompletionPer1M:  12.0,
		CachedInputPer1M: 1.5,
	}

	RecordTurn(TurnParams{
		RequestID:        "r1",
		Model:            model,
		Domain:           domain,
		PromptTokens:     500,
		CompletionTokens: 200,
		Pricing:          pricing,
		ResponseAPI: &ResponseAPIInput{
			ConversationID: "conv_pricing_1",
			HistoryLen:     0,
		},
	})
	RecordTurn(TurnParams{
		RequestID:        "r2",
		Model:            model,
		Domain:           domain,
		PromptTokens:     800,
		CompletionTokens: 300,
		Pricing:          pricing,
		ResponseAPI: &ResponseAPIInput{
			ConversationID: "conv_pricing_1",
			HistoryLen:     2,
		},
	})

	mu.Lock()
	defer mu.Unlock()
	st, ok := store["respapi:conv_pricing_1"]
	require.True(t, ok)

	// cumulative cost: ((500*3+200*12) + (800*3+300*12)) / 1e6
	wantCost := (500*3.0+200*12.0)/1_000_000.0 + (800*3.0+300*12.0)/1_000_000.0
	assert.InDelta(t, wantCost, st.cumulativeCost, 1e-9)
}

func TestRecordTurn_NoPricingDoesNotRecordCostHistogram(t *testing.T) {
	ResetForTesting()
	t.Cleanup(ResetForTesting)

	model := "no-price-model"
	domain := "unknown"

	before := histogramSampleCountCost("llm_session_turn_cost", model, domain, "USD")

	RecordTurn(TurnParams{
		RequestID:        "r1",
		Model:            model,
		Domain:           domain,
		PromptTokens:     100,
		CompletionTokens: 50,
		// Pricing intentionally omitted
		Chat: &ChatInput{
			UserID:   "u_noprice",
			Messages: []ChatMessage{{Role: "user", Content: "no-price-thread"}},
		},
	})

	after := histogramSampleCountCost("llm_session_turn_cost", model, domain, "USD")
	assert.Equal(t, before, after, "no cost observation should be recorded when pricing is not configured")

	mu.Lock()
	defer mu.Unlock()
	for _, st := range store {
		assert.InDelta(t, 0.0, st.cumulativeCost, 1e-12)
	}
}

func TestComputeCost(t *testing.T) {
	tests := []struct {
		name     string
		prompt   int
		compl    int
		pricing  TurnPricing
		wantCost float64
	}{
		{
			name:     "zero pricing returns zero cost",
			prompt:   1000,
			compl:    500,
			pricing:  TurnPricing{},
			wantCost: 0,
		},
		{
			name:    "standard pricing",
			prompt:  1000,
			compl:   500,
			pricing: TurnPricing{Currency: "USD", PromptPer1M: 5.0, CompletionPer1M: 15.0},
			// (1000*5 + 500*15) / 1e6 = 0.0125
			wantCost: 0.0125,
		},
		{
			name:     "only completion pricing",
			prompt:   0,
			compl:    1000,
			pricing:  TurnPricing{Currency: "USD", CompletionPer1M: 10.0},
			wantCost: 0.01,
		},
		{
			name:     "free model with explicit currency",
			prompt:   5000,
			compl:    2000,
			pricing:  TurnPricing{Currency: "USD", PromptPer1M: 0, CompletionPer1M: 0},
			wantCost: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := computeCost(tc.prompt, tc.compl, tc.pricing)
			assert.InDelta(t, tc.wantCost, got, 1e-9)
		})
	}
}
