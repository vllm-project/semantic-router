package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// noopMemoryStore satisfies memory.Store for tests that need a non-nil
// MemoryExtractor but never reach the actual store operations.
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
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		// Non-nil extractor so the function reaches past the nil check
		MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}),
	}

	// No per-decision plugin → extractAutoStore returns false
	// Router AutoStore=true -> fallback kicks in -> function does NOT return early
	// The goroutine runs but extractMemoryInfo fails gracefully (no ResponseObjectState)
	reqCtx := &RequestContext{
		RequestID: "req-router-fallback",
	}

	router.scheduleSemanticResponseMemoryStore(reqCtx, memoryTestResponse("test"))
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
	resources := newResourceScope()
	resources.add(store.Close)
	router := &OpenAIRouter{
		Config:          &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
		MemoryExtractor: memory.NewMemoryChunkStore(store),
		resources:       resources,
	}
	service := NewRouterService(router)
	reqCtx := &RequestContext{
		TrustedIdentity: authz.TrustedIdentity{UserID: "user-1"},
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

	router.scheduleResponseMemoryStoreText(reqCtx, "The conference itinerary has been saved for later reference.")
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
