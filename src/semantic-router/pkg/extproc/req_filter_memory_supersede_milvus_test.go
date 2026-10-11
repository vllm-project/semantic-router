package extproc

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// StorageIntegration: milvus
func TestMemoryRetrievalDropsSupersededFactsWithLiveMilvus(t *testing.T) {
	storagetest.Require(t, "milvus")
	address := os.Getenv("MILVUS_ADDRESS")
	if address == "" {
		address = "localhost:19530"
	}
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "1")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	milvusClient, err := client.NewClient(ctx, client.Config{Address: address})
	require.NoError(t, err)
	collection := fmt.Sprintf("memory_supersede_%x", time.Now().UnixNano())
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cleanupCancel()
		time.Sleep(time.Second)
		if dropErr := milvusClient.DropCollection(cleanupCtx, collection); dropErr != nil {
			t.Logf("drop test collection %s: %v", collection, dropErr)
		}
		require.NoError(t, milvusClient.Close())
	})

	memoryConfig := memory.DefaultMemoryConfig()
	embeddingConfig := memory.EmbeddingConfig{
		Model:     memory.EmbeddingModelMMBERT,
		Dimension: memoryConfig.Milvus.Dimension,
	}
	store, err := memory.NewMilvusStore(memory.MilvusStoreOptions{
		Client:          milvusClient,
		CollectionName:  collection,
		Config:          memoryConfig,
		Enabled:         true,
		EmbeddingConfig: &embeddingConfig,
	})
	require.NoError(t, err)
	chunks := memory.NewMemoryChunkStore(store)
	scenarios := []struct {
		name    string
		userID  string
		turns   []storedMemoryTurn
		query   string
		present []string
		absent  []string
	}{
		{
			name:   "quoted translation preserves residence",
			userID: "pr4260-quote",
			turns: []storedMemoryTurn{
				{user: "I live in Boston, near the Charles River.", assistant: "Got it, you live in Boston."},
				{user: "Please translate this sentence: ‘I just moved to Denver, and I live there now.’"},
			},
			query:   "Does my Boston residence change after asking to translate a Denver move?",
			present: []string{"I live in Boston"},
		},
		{
			name:   "straight single-quoted translation preserves residence",
			userID: "pr4260-straight-quote",
			turns: []storedMemoryTurn{
				{user: "I live in Boston, near the Charles River.", assistant: "Got it, you live in Boston."},
				{user: "Please translate this sentence: 'Yesterday I just moved to Denver, and I live there now.'"},
			},
			query:   "Does my Boston residence change after asking to translate a Denver move?",
			present: []string{"I live in Boston"},
		},
		{
			name:   "corrected residence preserves independent assistant fact",
			userID: "pr4260-assistant",
			turns: []storedMemoryTurn{
				{user: "I live in Boston.", assistant: "Your dog Biscuit is a beagle."},
				{user: "I just moved to Denver, and I live there now.", assistant: "Welcome to Denver!"},
			},
			query:   "Where do I live now, and what do you know about my dog?",
			present: []string{"Denver", "Your dog Biscuit is a beagle"},
			absent:  []string{"I live in Boston"},
		},
		{
			name:   "corrected residence preserves a dog fact that shares the city's name",
			userID: "pr4260-terrier",
			turns: []storedMemoryTurn{
				{user: "I live in Boston.", assistant: "Your dog Biscuit is a Boston terrier."},
				{user: "I just moved to Denver, and I live there now.", assistant: "Welcome to Denver!"},
			},
			query:   "Where do I live now, and what breed is my dog?",
			present: []string{"Denver", "Your dog Biscuit is a Boston terrier"},
			absent:  []string{"I live in Boston"},
		},
		{
			name:   "job change preserves another person's matching job",
			userID: "pr4260-sister",
			turns: []storedMemoryTurn{
				{user: "I work as a nurse.", assistant: "Got it, you work as a nurse. Your sister is a nurse too."},
				{user: "I changed jobs, and I work as a paramedic now.", assistant: "Congratulations!"},
			},
			query:   "What is my job now, and what does my sister do?",
			present: []string{"paramedic", "Your sister is a nurse too"},
			absent:  []string{"I work as a nurse", "you work as a nurse"},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for i, turn := range scenario.turns {
				require.NoError(t, chunks.ProcessResponse(ctx, "pr4260-session", scenario.userID, turn.user, turn.assistant))
				if i < len(scenario.turns)-1 {
					time.Sleep(time.Second + 100*time.Millisecond)
				}
			}
			time.Sleep(2 * time.Second)

			router := &OpenAIRouter{
				Config: &config.RouterConfig{Memory: config.MemoryConfig{
					Enabled: true, Backend: "milvus", DefaultSimilarityThreshold: 0.10,
				}},
				MemoryStore: store,
			}
			request := testNeutralRequest("entrypoint", scenario.query)
			requestContext := &RequestContext{
				Headers:         map[string]string{"x-authz-user-id": scenario.userID},
				TraceContext:    ctx,
				SemanticRequest: request,
			}
			require.NoError(t, router.handleMemoryRetrieval(requestContext, scenario.query, request))
			t.Logf("status=%s/%s results=%d\n%s", requestContext.MemoryStatus, requestContext.MemoryReason, requestContext.MemoryResultCount, requestContext.MemoryContext)
			require.Equal(t, "used", requestContext.MemoryStatus)
			for _, fact := range scenario.present {
				require.Contains(t, requestContext.MemoryContext, fact)
			}
			for _, fact := range scenario.absent {
				require.NotContains(t, requestContext.MemoryContext, fact)
			}
		})
	}
}
