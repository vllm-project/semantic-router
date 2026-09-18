package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"
)

type compatibilityStoreBase interface{ Store }

type compatibilityStore struct {
	compatibilityStoreBase
	writes int
	failAt int
	err    error
}

func (*compatibilityStore) IsEnabled() bool { return true }

func (s *compatibilityStore) Store(context.Context, *Memory) error {
	s.writes++
	if s.writes == s.failAt {
		return s.err
	}
	return nil
}

func TestMemoryExtractorLegacyMethods(t *testing.T) {
	const user = "My preferred deployment region is us-west-2."
	const assistant = "I will remember that preference."
	history := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Where should I deploy?"), openai.AssistantMessage("Use a regional cluster."),
		openai.UserMessage("Which language?"), openai.AssistantMessage("Use Go for the service."),
	}
	for _, method := range []struct {
		name   string
		call   func(*MemoryExtractor, context.Context) error
		writes int
	}{
		{"ProcessResponse", func(e *MemoryExtractor, ctx context.Context) error {
			return e.ProcessResponse(ctx, "session", "user", user, assistant)
		}, 1},
		{"ProcessResponseWithHistory", func(e *MemoryExtractor, ctx context.Context) error {
			return e.ProcessResponseWithHistory(ctx, "session", "user", user, assistant, history)
		}, 2},
	} {
		t.Run(method.name, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				failAt int
			}{
				{name: "success"},
				{name: "first write failure", failAt: 1},
				{name: "last write failure", failAt: method.writes},
			} {
				t.Run(tc.name, func(t *testing.T) {
					failure := errors.New("storage unavailable")
					store := &compatibilityStore{failAt: tc.failAt, err: failure}
					err := method.call(NewMemoryChunkStore(store), t.Context())
					if tc.failAt > 0 {
						require.ErrorIs(t, err, failure)
						require.Equal(t, tc.failAt, store.writes)
					} else {
						require.NoError(t, err)
						require.Equal(t, method.writes, store.writes)
					}
				})
			}
			t.Run("cancelled", func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				cancel()
				store := &compatibilityStore{}
				require.ErrorIs(t, method.call(NewMemoryChunkStore(store), ctx), context.Canceled)
				require.Zero(t, store.writes)
			})
			t.Run("nil extractor", func(t *testing.T) {
				require.NoError(t, method.call(nil, t.Context()))
			})
		})
	}
}
