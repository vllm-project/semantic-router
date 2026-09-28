package extproc

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

type storedMemoryTurn struct {
	user      string
	assistant string
}

func TestMemoryRetrievalDropsSupersededFacts(t *testing.T) {
	boston := storedMemoryTurn{user: "I live in Boston, near the Charles River.", assistant: "Got it, you live in Boston."}
	denver := storedMemoryTurn{user: "I just moved to Denver, and I live there now.", assistant: "Welcome to Denver!"}
	dog := storedMemoryTurn{user: "My dog is a beagle named Biscuit.", assistant: "Biscuit the beagle, noted."}
	birthday := storedMemoryTurn{user: "Biscuit turned three today, so I bought my dog a new toy.", assistant: "Happy birthday to Biscuit!"}
	workout := storedMemoryTurn{user: "I work out every morning.", assistant: "Morning workouts, noted."}
	paramedic := storedMemoryTurn{user: "I changed jobs, and I work as a paramedic now.", assistant: "Congratulations!"}
	quotedMove := storedMemoryTurn{
		user:      "What does a stored session look like?",
		assistant: "Like this:\n---\nQ: I moved to Denver, and I live there now",
	}
	cityAndDog := storedMemoryTurn{user: "I live in Boston, my dog is Biscuit.", assistant: "Noted."}
	// A session window stored before quoted turn boundaries were escaped.
	oldWindow := memory.Memory{
		Source: "session_window",
		Content: "Q: My dog is a beagle named Biscuit.\nA: Biscuit the beagle, noted.\n---\n" +
			"Q: What does a stored session look like?\nA: Like this:\n---\nQ: I moved to Denver, and I live there now",
	}

	cases := []struct {
		name       string
		turns      []storedMemoryTurn
		stored     []memory.Memory
		query      string
		injected   []string
		superseded []string
	}{
		{
			name:       "a correction replaces the fact it supersedes",
			turns:      []storedMemoryTurn{boston, denver},
			query:      "Which city do I live in now?",
			injected:   []string{"Denver"},
			superseded: []string{"Boston"},
		},
		{
			name:     "a newer fact about the same dog keeps the older one",
			turns:    []storedMemoryTurn{dog, birthday},
			query:    "What is my dog's name?",
			injected: []string{"beagle named Biscuit", "Biscuit turned three"},
		},
		{
			name:       "a correction keeps unrelated memories",
			turns:      []storedMemoryTurn{dog, boston, denver},
			query:      "Which city do I live in now, and what is my dog's name?",
			injected:   []string{"Denver", "beagle named Biscuit"},
			superseded: []string{"Boston"},
		},
		{
			name:     "a job change keeps an unrelated work fact",
			turns:    []storedMemoryTurn{workout, paramedic},
			query:    "What do I do every morning, and what is my job now?",
			injected: []string{"work out every morning", "paramedic"},
		},
		{
			name:     "a turn quoted in a reply keeps the user's fact",
			turns:    []storedMemoryTurn{boston, quotedMove},
			query:    "Which city do I live in?",
			injected: []string{"I live in Boston"},
		},
		{
			name:     "a correction keeps a comma clause's own fact",
			turns:    []storedMemoryTurn{cityAndDog, denver},
			query:    "Which city do I live in, and what is my dog's name?",
			injected: []string{"my dog is Biscuit", "Denver"},
		},
		{
			name:     "a turn quoted in an old session window keeps the user's fact",
			turns:    []storedMemoryTurn{boston},
			stored:   []memory.Memory{oldWindow},
			query:    "Which city do I live in?",
			injected: []string{"I live in Boston"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "1")
			store := memory.NewInMemoryStoreWithConfig(memory.EmbeddingConfig{Model: memory.EmbeddingModelBERT})
			chunks := memory.NewMemoryChunkStore(store)
			bg := context.Background()
			for _, turn := range tc.turns {
				require.NoError(t, chunks.ProcessResponse(bg, "session", "user-1", turn.user, turn.assistant))
			}
			for i, mem := range tc.stored {
				mem.ID, mem.UserID, mem.Type, mem.CreatedAt = fmt.Sprintf("stored-%d", i), "user-1", memory.MemoryTypeEpisodic, time.Now()
				require.NoError(t, store.Store(bg, &mem))
			}

			// 0.10 is what the memory integration config pairs with deterministic embeddings.
			router := &OpenAIRouter{
				Config: &config.RouterConfig{Memory: config.MemoryConfig{
					Enabled: true, Backend: "memory", DefaultSimilarityThreshold: 0.10,
				}},
				MemoryStore: store,
			}
			request := testNeutralRequest("entrypoint", tc.query)
			ctx := &RequestContext{
				Headers:         map[string]string{"x-authz-user-id": "user-1"},
				TraceContext:    bg,
				SemanticRequest: request,
			}
			require.NoError(t, router.handleMemoryRetrieval(ctx, tc.query, request))

			t.Logf("status=%s/%s results=%d\n%s", ctx.MemoryStatus, ctx.MemoryReason, ctx.MemoryResultCount, ctx.MemoryContext)
			assert.Equal(t, "used", ctx.MemoryStatus)
			for _, fact := range tc.injected {
				assert.Contains(t, ctx.MemoryContext, fact)
			}
			for _, fact := range tc.superseded {
				assert.NotContains(t, ctx.MemoryContext, fact, "superseded fact injected beside its correction")
			}
		})
	}
}
