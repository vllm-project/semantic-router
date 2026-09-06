package extproc

import (
	"context"
	"fmt"
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

type blockingMemoryStore struct {
	noopMemoryStore
	storeStarted chan struct{}
	allowStore   chan struct{}
	closed       chan struct{}
}

func (s *blockingMemoryStore) Store(ctx context.Context, _ *memory.Memory) error {
	close(s.storeStarted)
	select {
	case <-s.allowStore:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *blockingMemoryStore) Close() error {
	close(s.closed)
	return nil
}

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
	require.NoError(t, recorder.DrainOutcomes())

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

func TestResponseMemoryStoreHoldsGenerationUntilBackgroundWriteCompletes(t *testing.T) {
	store := &blockingMemoryStore{
		storeStarted: make(chan struct{}),
		allowStore:   make(chan struct{}),
		closed:       make(chan struct{}),
	}
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	t.Cleanup(func() {
		select {
		case <-store.allowStore:
		default:
			close(store.allowStore)
		}
		_ = runner.RetireAndWait(time.Second)
	})
	resources := newResourceScope()
	resources.add(store.Close)
	resources.add(func() error { return runner.RetireAndWait(time.Second) })
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
		MemoryExtractor:   memory.NewMemoryChunkStore(store),
		memoryPersistence: runner,
		resources:         resources,
	}
	service := NewRouterService(router)
	reqCtx := &RequestContext{
		Headers: map[string]string{headers.AuthzUserID: "user-1"},
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentText,
					Text: "Please remember the detailed itinerary for next month's conference trip.",
				}},
			}},
		},
	}

	reqCtx.TraceContext = context.Background()
	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("The conference itinerary has been saved for later reference."))
	select {
	case <-store.storeStarted:
	case <-time.After(time.Second):
		t.Fatal("background memory write did not start")
	}

	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- service.Shutdown(context.Background())
	}()
	select {
	case <-store.closed:
		t.Fatal("generation resources closed while the background memory write was active")
	case <-time.After(50 * time.Millisecond):
	}

	close(store.allowStore)
	select {
	case err := <-shutdownDone:
		if err != nil {
			t.Fatalf("Shutdown() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("shutdown did not finish after the background memory write completed")
	}
	select {
	case <-store.closed:
	default:
		t.Fatal("generation resources remained open after shutdown")
	}
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

func TestResponseMemoryAutoStoreSurvivesProviderPreparation(t *testing.T) {
	for _, tc := range []struct {
		name            string
		control         string
		configAutoStore bool
		mutateRequest   bool
		wantStored      bool
	}{
		{"explicit_false", `,"auto_store":false`, true, false, false},
		{"explicit_true", `,"auto_store":true`, false, false, true},
		{"omitted_enabled", "", true, false, true},
		{"omitted_disabled", "", false, false, false},
		{"false_snapshot", `,"auto_store":false`, true, true, false},
		{"true_snapshot", `,"auto_store":true`, false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &persistenceRegressionStore{InMemoryStore: memory.NewInMemoryStore()}
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: tc.name})
			require.NoError(t, err)
			runner := memory.NewPersistenceRunner(5*time.Second, 1, 1)
			t.Cleanup(func() {
				assert.NoError(t, runner.RetireAndWait(5*time.Second))
				assert.NoError(t, recorder.DrainOutcomes())
			})
			router := &OpenAIRouter{
				Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: tc.configAutoStore}},
				MemoryExtractor:   memory.NewMemoryChunkStore(backend),
				ReplayRecorder:    recorder,
				memoryPersistence: runner,
			}
			ctx := persistenceRegressionContext(tc.name)
			ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
			ctx.RouterReplayID = tc.name
			body := []byte(fmt.Sprintf(`{"model":"test-model","input":"Explain how to deploy a backend service in eu-central-1."%s}`, tc.control))
			request, immediate := router.prepareProtocolRequest(body, ctx)
			require.Nil(t, immediate)
			require.NotNil(t, request)
			if tc.mutateRequest {
				// Routing mutations must not alter the original client preference.
				require.NotNil(t, request.AutoStore)
				*request.AutoStore = !*request.AutoStore
			}
			_, err = router.materializeResponseObjectContext(request, ctx)
			require.NoError(t, err)
			require.Nil(t, request.AutoStore, "router controls must be removed from the provider request")

			router.scheduleResponseMemoryStoreText(ctx, "Deploy the service using a regional cluster and a load balancer.")
			require.NoError(t, runner.RetireAndWait(5*time.Second))
			require.NoError(t, recorder.DrainOutcomes())
			record, found := recorder.GetRecord(tc.name)
			require.True(t, found)
			stored, err := backend.List(t.Context(), memory.ListOptions{UserID: "original-user", Limit: 10})
			require.NoError(t, err)
			if tc.wantStored {
				require.Len(t, stored.Memories, 1)
				require.Len(t, record.Outcomes, 2)
				assert.Equal(t, "scheduled", record.Outcomes[0].Verdict)
				assert.Equal(t, "completed", record.Outcomes[1].Verdict)
				assert.Equal(t, "persisted", record.Outcomes[1].Reason)
			} else {
				assert.Empty(t, stored.Memories, "disabled requests must not persist memory")
				require.Len(t, record.Outcomes, 1)
				assert.Equal(t, "disabled", record.Outcomes[0].Verdict)
				assert.Equal(t, "auto_store_off", record.Outcomes[0].Reason)
			}
			terminal := record.Outcomes[len(record.Outcomes)-1]
			assert.Equal(t, "terminal", terminal.Metadata["phase"])
			assert.Equal(t, "false", terminal.Metadata["fail_open"])
		})
	}
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
