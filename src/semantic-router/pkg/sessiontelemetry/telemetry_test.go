package sessiontelemetry

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func histogramSampleCount(metricName, model, domain string) uint64 {
	mf, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mf {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_HISTOGRAM {
			continue
		}
		for _, m := range fam.GetMetric() {
			matchModel, matchDomain := false, false
			for _, l := range m.GetLabel() {
				if l.GetName() == "model" && l.GetValue() == model {
					matchModel = true
				}
				if l.GetName() == "domain" && l.GetValue() == domain {
					matchDomain = true
				}
			}
			if matchModel && matchDomain {
				if h := m.GetHistogram(); h != nil && h.SampleCount != nil {
					return h.GetSampleCount()
				}
			}
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
