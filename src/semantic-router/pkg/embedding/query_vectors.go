package embedding

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// DefaultQueryWindowLimit caps how many windows of one query are searched. Each
// window costs an embedding and a store round trip, and a query long enough to
// need more than this has already been sampled from end to end.
const DefaultQueryWindowLimit = 8

// TextEmbedder is the embed half of Provider, so a caller holding a narrower
// interface, such as the vector store's Embedder, reuses window embedding
// without depending on the whole provider surface.
type TextEmbedder interface {
	Embed(context.Context, string) ([]float32, error)
}

// QueryVectors returns one embedding per window of the query.
//
// Embedding a long query once reads only its first window, so two questions
// behind the same long preamble produce the same vector and retrieve the same
// documents. Every sampled window is embedded instead, and the caller searches
// with each and keeps a document's best score, which is the aggregation that
// ranks best over passages of a long input. An embedder that exposes no token
// windows keeps its single-embedding behaviour, because a remote model owns its
// own truncation and the offsets here would not describe it.
func QueryVectors(ctx context.Context, embedder TextEmbedder, text string, limit int) ([][]float32, error) {
	if embedder == nil {
		return nil, fmt.Errorf("embedding provider was not prepared")
	}
	tokenizer, ok := embedder.(WindowProvider)
	if !ok {
		return singleQueryVector(ctx, embedder, text)
	}
	windows, err := tokenizer.Windows(ctx, text, 0)
	if err != nil {
		// A prepared provider always satisfies WindowProvider, because Set.Get
		// wraps it, so the capability only shows when the tokenizer is asked
		// for. A provider that has none keeps its single embedding, while a
		// tokenizer that actually failed is still reported.
		if errors.Is(err, binding.ErrCapability) {
			return singleQueryVector(ctx, embedder, text)
		}
		return nil, err
	}
	if len(windows) <= 1 {
		return singleQueryVector(ctx, embedder, text)
	}

	sampled := SampleWindows(windows, limit)
	vectors := make([][]float32, 0, len(sampled))
	for _, window := range sampled {
		vector, embedErr := embedder.Embed(ctx, text[window.Start:window.End])
		if embedErr != nil {
			return nil, embedErr
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func singleQueryVector(ctx context.Context, embedder TextEmbedder, text string) ([][]float32, error) {
	vector, err := embedder.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	return [][]float32{vector}, nil
}

// SampleWindows keeps at most limit windows, evenly spaced and always including
// the first and the last, because the question a long query asks sits at either
// end of it about as often.
func SampleWindows(windows []Window, limit int) []Window {
	if limit <= 0 || len(windows) <= limit {
		return windows
	}
	if limit == 1 {
		return windows[:1]
	}
	kept := make([]Window, limit)
	for i := range kept {
		kept[i] = windows[i*(len(windows)-1)/(limit-1)]
	}
	return kept
}
