package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHybridRAGVectorStorePrimaryReturnsMatchingContext(t *testing.T) {
	router, storeID := ragVectorStoreFixture(t, &plainRAGEmbedder{})
	topK := 2
	threshold := float32(0.5)
	ragConfig := &config.RAGPluginConfig{
		Enabled:             true,
		Backend:             "hybrid",
		TopK:                &topK,
		SimilarityThreshold: &threshold,
		BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{
			Primary: "vectorstore",
			PrimaryConfig: config.MustStructuredPayload(&config.VectorStoreRAGConfig{
				VectorStoreID: storeID,
			}),
		}),
	}

	got, err := router.retrieveFromHybrid(context.Background(), &RequestContext{UserContent: "how long does a refund take"}, ragConfig)
	if err != nil {
		t.Fatal(err)
	}
	if got != "refunds take 14 business days" {
		t.Fatalf("context = %q", got)
	}
}

func TestHybridRAGVectorStoreFallbackReturnsMatchingContext(t *testing.T) {
	router, storeID := ragVectorStoreFixture(t, &plainRAGEmbedder{})
	topK := 2
	threshold := float32(0.5)
	ragConfig := &config.RAGPluginConfig{
		Enabled:             true,
		Backend:             "hybrid",
		TopK:                &topK,
		SimilarityThreshold: &threshold,
		BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{
			Primary:  "milvus",
			Fallback: "vectorstore",
			PrimaryConfig: config.MustStructuredPayload(&config.MilvusRAGConfig{
				Collection: "docs",
			}),
			FallbackConfig: config.MustStructuredPayload(&config.VectorStoreRAGConfig{
				VectorStoreID: storeID,
			}),
		}),
	}

	got, err := router.retrieveFromHybrid(context.Background(), &RequestContext{UserContent: "how long does a refund take"}, ragConfig)
	if err != nil {
		t.Fatal(err)
	}
	if got != "refunds take 14 business days" {
		t.Fatalf("context = %q", got)
	}
}
