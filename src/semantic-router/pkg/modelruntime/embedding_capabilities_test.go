package modelruntime

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// A consumer whose deployment has no prepared provider fails preparation; a
// consumer that reads the full output is not probed.
func TestPreparedEmbeddingsServeEveryConsumer(t *testing.T) {
	calls := 0
	provider, err := embedding.NewFuncProvider(config.EmbeddingBackendModelRuntime, 8, func(context.Context, string) ([]float32, error) {
		calls++
		return make([]float32, 8), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	providers := map[string]embedding.Provider{config.EmbeddingDefaultDeployment: provider}
	served := []config.EmbeddingRequirement{{Deployment: config.EmbeddingDefaultDeployment, Consumer: config.EmbeddingConsumerAPI}}
	if err := validatePreparedEmbeddings(context.Background(), served, providers); err != nil || calls != 0 {
		t.Fatalf("full-output consumer: %v, %d probes", err, calls)
	}
	missing := []config.EmbeddingRequirement{served[0], {Deployment: "docs-embed", Consumer: config.EmbeddingConsumerRAG}}
	if err := validatePreparedEmbeddings(context.Background(), missing, providers); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a consumer without its deployment's provider was accepted: %v", err)
	}
}
