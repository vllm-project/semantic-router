package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

func TestScheduleResponseMemoryStore_NoOpWithoutMemoryExtractor(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor: nil,
	}

	reqCtx := &RequestContext{
		RequestID: "req-noop",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-noop",
		},
	}

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
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

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
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
	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
}

func TestScheduleResponseMemoryStore_FallsBackToGlobalAutoStore(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		// Non-nil extractor so the function reaches past the nil check
		MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}),
	}

	// No per-decision plugin → extractAutoStore returns false
	// Global AutoStore=true → fallback kicks in → function does NOT return early
	// The goroutine runs but extractMemoryInfo fails gracefully (no ResponseAPICtx)
	reqCtx := &RequestContext{
		RequestID: "req-global-fallback",
	}

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
}

func TestScheduleResponseMemoryStore_SkippedWhenBothAutoStoresDisabled(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: false},
		},
		MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}),
	}

	// extractAutoStore returns false + global AutoStore=false → autoStoreEnabled stays false → skip
	reqCtx := &RequestContext{
		RequestID: "req-both-disabled",
	}

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
}
