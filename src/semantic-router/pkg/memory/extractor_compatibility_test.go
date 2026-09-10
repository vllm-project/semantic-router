package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"
)

// An external caller compiled against the original method set must still work.
type historyProcessor interface {
	ProcessResponseWithHistory(context.Context, string, string, string, string, []openai.ChatCompletionMessageParamUnion) error
}

var _ historyProcessor = (*MemoryExtractor)(nil)

type compatibilityUnusedStore interface{ Store }

type compatibilityStore struct {
	compatibilityUnusedStore
	writes int
	failAt int
	err    error
}

func (s *compatibilityStore) IsEnabled() bool { return true }
func (s *compatibilityStore) Store(context.Context, *Memory) error {
	s.writes++
	if s.writes == s.failAt {
		return s.err
	}
	return nil
}

func TestHistoryAPIsPreserveCountsAndErrors(t *testing.T) {
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
			oldStore := &compatibilityStore{failAt: tc.failAt, err: failure}
			var old historyProcessor = NewMemoryChunkStore(oldStore)
			err := old.ProcessResponseWithHistory(context.Background(), "session", "user", tc.user, tc.assistant, tc.history)
			newStore := &compatibilityStore{failAt: tc.failAt, err: failure}
			count, countedErr := NewMemoryChunkStore(newStore).ProcessResponseWithHistoryCount(
				context.Background(), "session", "user", tc.user, tc.assistant, tc.history)
			require.Equal(t, tc.count, count)
			require.Equal(t, oldStore.writes, newStore.writes)
			if tc.failAt > 0 {
				require.ErrorIs(t, err, failure)
				require.ErrorIs(t, countedErr, failure)
			} else {
				require.NoError(t, err)
				require.NoError(t, countedErr)
			}
		})
	}
}
