package cache

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// cacheFixtureProvider models an embedding operation that completes before the
// caller observes cancellation. No process-global native state is involved.
type cacheFixtureProvider struct{ storagetest.Vectors }

func (p cacheFixtureProvider) Embed(_ context.Context, text string) ([]float32, error) {
	return p.Vectors.Embed(context.Background(), text)
}

// FitsInputWithOptions reads at most 512 words whole, at any view.
func (p cacheFixtureProvider) FitsInputWithOptions(_ context.Context, text string, _ embedding.Options) (bool, error) {
	return len(strings.Fields(text)) <= 512, nil
}

func cacheTestEmbeddingProvider() embedding.Provider {
	return cacheFixtureProvider{Vectors: storagetest.Vectors{Size: 384}}
}
