package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// noopMemoryStore satisfies memory.Store for response-memory tests.
type noopMemoryStore struct{}

func (s *noopMemoryStore) Store(_ context.Context, _ *memory.Memory) error { return nil }
func (s *noopMemoryStore) Retrieve(_ context.Context, _ memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}

func (s *noopMemoryStore) Get(_ context.Context, _ string) (*memory.Memory, error) {
	return nil, nil
}

func (s *noopMemoryStore) Update(_ context.Context, _ string, _ *memory.Memory) error { return nil }

func (s *noopMemoryStore) List(_ context.Context, _ memory.ListOptions) (*memory.ListResult, error) {
	return nil, nil
}

func (s *noopMemoryStore) Forget(_ context.Context, _ string) error                    { return nil }
func (s *noopMemoryStore) ForgetByScope(_ context.Context, _ memory.MemoryScope) error { return nil }
func (s *noopMemoryStore) IsEnabled() bool                                             { return true }
func (s *noopMemoryStore) CheckConnection(_ context.Context) error                     { return nil }
func (s *noopMemoryStore) Close() error                                                { return nil }

func TestScheduleResponseMemoryStore_NoOpWithoutMemoryExtractor(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor: nil,
	}

	reqCtx := &RequestContext{
		RequestID: "req-noop",
		ResponseObjectState: &ResponseObjectState{
			ConversationID: "conv-noop",
		},
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
}

func TestScheduleResponseMemoryStore_SkippedWhenAutoStoreDisabled(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: false},
		},
		MemoryExtractor: nil,
	}

	reqCtx := &RequestContext{
		RequestID: "req-disabled",
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
}

func TestScheduleResponseMemoryStore_SkippedWhenJailbreakDetected(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}),
	}

	reqCtx := &RequestContext{
		RequestID:                 "req-jailbreak",
		ResponseJailbreakDetected: true,
	}

	// Should return early at the jailbreak check — no goroutine launched.
	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
}

func TestScheduleResponseMemoryStore_FallsBackToRouterAutoStore(t *testing.T) {
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		// Non-nil extractor so the function reaches past the nil check
		MemoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
		memoryPersistence: runner,
	}

	// No per-decision plugin → extractAutoStore returns false
	// Router AutoStore=true -> fallback kicks in -> function does NOT return early
	// The goroutine runs but extractMemoryInfo fails gracefully (no ResponseObjectState)
	reqCtx := &RequestContext{
		RequestID:    "req-router-fallback",
		TraceContext: context.Background(),
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
	require.NoError(t, runner.RetireAndWait(time.Second))
}

func TestScheduleResponseMemoryStore_AppendsScheduledAndTerminalReplayOutcomes(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	const replayID = "memory-persistence-replay"
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: replayID})
	require.NoError(t, err)

	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
		ReplayRecorder:    recorder,
		memoryPersistence: runner,
	}
	reqCtx := &RequestContext{
		RequestID:    "req-memory-persistence-replay",
		Headers:      map[string]string{headers.AuthzUserID: "user-1"},
		SessionID:    "session-1",
		TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Explain bounded memory persistence workers."}},
		}}},
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("Use a bounded queue and report completion after the store write."))
	require.NoError(t, runner.RetireAndWait(time.Second))

	record, found := recorder.GetRecord(replayID)
	require.True(t, found)
	require.Len(t, record.Outcomes, 2)

	assert.Equal(t, "scheduled", record.Outcomes[0].Metadata["phase"])
	assert.Equal(t, "scheduled", record.Outcomes[0].Verdict)
	assert.Equal(t, "queue_accepted", record.Outcomes[0].Reason)
	assert.Equal(t, "terminal", record.Outcomes[1].Metadata["phase"])
	assert.Equal(t, "completed", record.Outcomes[1].Verdict)
	assert.Equal(t, "persisted", record.Outcomes[1].Reason)
	assert.Equal(t, "memory_persistence_receipt", record.Outcomes[1].Metadata["kind"])
}

func TestScheduleResponseMemoryStore_SkippedWhenBothAutoStoresDisabled(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: false},
		},
		MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}),
	}

	// extractAutoStore returns false + router AutoStore=false -> autoStoreEnabled stays false -> skip
	reqCtx := &RequestContext{
		RequestID: "req-both-disabled",
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
}

func TestScheduleResponseMemoryStore_RejectedWriteIsItsOwnMetricStatus(t *testing.T) {
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	require.NoError(t, runner.RetireAndWait(time.Second))

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
		memoryPersistence: runner,
	}
	reqCtx := &RequestContext{
		RequestID:               "req-memory-persistence-rejected",
		VSRSelectedDecisionName: "balance",
		TraceContext:            context.Background(),
	}

	rejected := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", "balance", "rejected")
	skipped := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", "balance", "skipped")
	beforeRejected := testutil.ToFloat64(rejected)
	beforeSkipped := testutil.ToFloat64(skipped)

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("dropped before the worker pool accepted it"))

	assert.Equal(t, beforeRejected+1, testutil.ToFloat64(rejected),
		"a refused write must be countable on its own status label")
	assert.Equal(t, beforeSkipped, testutil.ToFloat64(skipped),
		"a refused write must not be counted as a content-level skip")
}

func memoryTestResponse(text string) *llmprotocol.Response {
	return &llmprotocol.Response{
		Generation: 1,
		ID:         "response_test",
		Model:      "model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_test", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
}
