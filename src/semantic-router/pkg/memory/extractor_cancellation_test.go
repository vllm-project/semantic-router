package memory

import (
	"context"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"
)

type cancelAfterTurnStore struct {
	*InMemoryStore
	cancel context.CancelFunc
	writes int
}

func (s *cancelAfterTurnStore) Store(context.Context, *Memory) error {
	s.writes++
	s.cancel()
	return nil
}

func TestExtractionCancellationPreventsSessionWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	backend := &cancelAfterTurnStore{InMemoryStore: NewInMemoryStore(), cancel: cancel}
	extractor := NewMemoryChunkStore(backend)
	history := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Which language should I use?"),
		openai.AssistantMessage("Go is a good choice for backend services."),
		openai.UserMessage("How does it handle concurrent requests?"),
		openai.AssistantMessage("It provides goroutines and channels."),
	}
	count, err := extractor.ProcessResponseWithHistory(ctx, "session", "user",
		"Explain how Go concurrency works in backend services.",
		"Goroutines let backend services handle concurrent requests efficiently.", history)
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, 1, count, "preserve the already accepted turn")
	require.Equal(t, 1, backend.writes, "do not start another write after cancellation")
	count, err = extractor.ProcessResponseWithHistory(ctx, "session", "user", "retry", "response", history)
	require.ErrorIs(t, err, context.Canceled)
	require.Zero(t, count)
	require.Equal(t, 1, backend.writes)
}
