package extproc

import (
	"context"
	"testing"

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

	cases := []struct {
		name       string
		turns      []storedMemoryTurn
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
