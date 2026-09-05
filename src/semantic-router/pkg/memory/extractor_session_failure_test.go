package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type sessionFailureStore struct {
	*InMemoryStore
	cause error
}

func (s *sessionFailureStore) Store(ctx context.Context, mem *Memory) error {
	if mem.Source == "session_window" {
		return s.cause
	}
	// Avoid model inference when testing partial persistence.
	mem.Embedding = []float32{1}
	return s.InMemoryStore.Store(ctx, mem)
}

func TestProcessResponseWithHistory_SessionFailurePreservesSuccessfulWrites(t *testing.T) {
	for _, tc := range []struct {
		name       string
		user       string
		wantStored int
	}{
		{"turn stored", "Explain how Go concurrency works in backend services.", 1},
		{"turn skipped", "Thanks", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cause := errors.New("session backend unavailable")
			backend := &sessionFailureStore{InMemoryStore: NewInMemoryStore(), cause: cause}
			extractor := NewMemoryChunkStore(backend)
			history := []openai.ChatCompletionMessageParamUnion{
				sdkUserMessage("Which language should I use?"),
				sdkAssistantMessage("Go is a good choice for backend services."),
				sdkUserMessage("How does it handle concurrent requests?"),
				sdkAssistantMessage("It provides goroutines and channels."),
			}

			count, err := extractor.ProcessResponseWithHistory(context.Background(), "session", "user",
				tc.user, "Goroutines let backend services handle concurrent requests efficiently.", history)

			require.ErrorIs(t, err, cause)
			assert.Equal(t, tc.wantStored, count)
			stored, err := backend.List(context.Background(), ListOptions{UserID: "user", Limit: 10})
			require.NoError(t, err)
			require.Len(t, stored.Memories, tc.wantStored)
			if tc.wantStored > 0 {
				assert.Equal(t, "conversation", stored.Memories[0].Source)
				assert.Contains(t, stored.Memories[0].Content, tc.user)
			}
		})
	}
}
