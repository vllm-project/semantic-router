package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// ragQueryEmbedding embeds the RAG query once. A query past the embedding
// model's input budget is cut by the model's own input policy.
func (r *OpenAIRouter) ragQueryEmbedding(ctx context.Context, query string, request *RequestContext) ([]float32, error) {
	provider, err := r.embeddingsForRequest(request).Get(config.RAGQueryEmbeddingModel, 0, 0)
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return nil, fmt.Errorf("RAG embedding provider was not prepared")
	}
	return provider.Embed(ctx, query)
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
