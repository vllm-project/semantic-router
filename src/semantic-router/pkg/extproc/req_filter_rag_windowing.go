package extproc

import (
	"context"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// ragQueryWindowLimit caps how many windows of one query are searched. Each
// window costs an embedding and a store round trip, and a query long enough to
// need more than this has already been sampled from end to end.
const ragQueryWindowLimit = embedding.DefaultQueryWindowLimit

// ragQueryEmbeddings returns one embedding per window of the query.
//
// Embedding a long query once reads only its first window, so two questions
// behind the same long preamble produce the same vector and retrieve the same
// documents. Every window is embedded instead, and the caller searches with each
// and keeps a document's best score, which is the aggregation that ranks best
// over passages of a long input.
//
// A provider that exposes no token windows, such as a remote embedding service,
// keeps its single embedding, because the model owns its own truncation there.
func ragQueryEmbeddings(ctx context.Context, provider embedding.Provider, query string) ([][]float32, error) {
	if provider == nil {
		return nil, fmt.Errorf("RAG embedding provider was not prepared")
	}
	return embedding.QueryVectors(ctx, provider, query, ragQueryWindowLimit)
}

// ragHits collects the documents the windows of one query retrieved, keeping the
// best score each document reached.
type ragHits struct {
	best map[string]float32
}

func (h *ragHits) add(contents []string, scores []float32) {
	if h.best == nil {
		h.best = make(map[string]float32, len(contents))
	}
	for i, content := range contents {
		var score float32
		if i < len(scores) {
			score = scores[i]
		}
		if current, seen := h.best[content]; !seen || score > current {
			h.best[content] = score
		}
	}
}

// top returns the documents by descending best score, at most topK of them.
func (h *ragHits) top(topK int) ([]string, []float32) {
	contents := make([]string, 0, len(h.best))
	for content := range h.best {
		contents = append(contents, content)
	}
	sort.Slice(contents, func(i, j int) bool {
		if h.best[contents[i]] == h.best[contents[j]] {
			return contents[i] < contents[j]
		}
		return h.best[contents[i]] > h.best[contents[j]]
	})
	if topK > 0 && len(contents) > topK {
		contents = contents[:topK]
	}
	scores := make([]float32, len(contents))
	for i, content := range contents {
		scores[i] = h.best[content]
	}
	return contents, scores
}

func (r *OpenAIRouter) ragQueryEmbeddings(ctx context.Context, query string, request *RequestContext) ([][]float32, error) {
	provider, err := r.embeddingsForRequest(request).Get("bert", 0, 0)
	if err != nil {
		return nil, err
	}
	return ragQueryEmbeddings(ctx, provider, query)
}

// Requests already hold the enclosing router generation lease. Named recipes
// resolve only their prepared snapshot, never the default recipe's binding.
func (r *OpenAIRouter) embeddingsForRequest(request *RequestContext) *embedding.Set {
	if r == nil {
		return nil
	}
	if request == nil || request.Routing.SelectedRecipe() == nil {
		return r.Embeddings
	}
	classifier := r.classifierForRequest(request)
	if classifier == nil {
		return nil
	}
	return classifier.PreparedEmbeddings()
}
