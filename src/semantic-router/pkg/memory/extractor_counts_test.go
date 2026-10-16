package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"
)

type countsStore struct {
	*InMemoryStore
	writes int
	failAt int
	err    error
}

// Count writes without reaching the embedding provider.
func (s *countsStore) Store(context.Context, *Memory) error {
	s.writes++
	if s.writes == s.failAt {
		return s.err
	}
	return nil
}

// The stored-chunk count is what the persistence receipt distinguishes a
// zero-write skip by, so it must survive a partial write failure.
func TestProcessResponseWithHistoryCountReportsWritesAndErrors(t *testing.T) {
	history := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Where should I deploy?"), openai.AssistantMessage("Use a regional cluster."),
		openai.UserMessage("Which language?"), openai.AssistantMessage("Use Go for the service."),
	}
	for _, tc := range []struct {
		name            string
		history         []openai.ChatCompletionMessageParamUnion
		user, assistant string
		failAt, count   int
	}{
		{name: "zero"},
		{name: "one", user: "My preferred deployment region is us-west-2.", assistant: "I will remember that preference.", count: 1},
		{name: "two", history: history, user: "My preferred deployment region is us-west-2.", assistant: "I will remember that preference.", count: 2},
		{name: "partial failure", history: history, user: "My preferred deployment region is us-west-2.", assistant: "I will remember that preference.", failAt: 2, count: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			failure := errors.New("storage unavailable")
			store := &countsStore{InMemoryStore: NewInMemoryStore(), failAt: tc.failAt, err: failure}
			count, err := NewMemoryChunkStore(store).ProcessResponseWithHistoryCount(
				context.Background(), "session", "user", tc.user, tc.assistant, tc.history)
			require.Equal(t, tc.count, count)
			if tc.failAt > 0 {
				require.ErrorIs(t, err, failure)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
