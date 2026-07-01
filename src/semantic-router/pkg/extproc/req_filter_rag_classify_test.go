package extproc

import (
	"errors"
	"fmt"
	"testing"
)

func TestClassifyRAGError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, "none"},
		{"timeout", errors.New("context deadline exceeded"), "timeout"},
		{"embedding", errors.New("failed to generate embedding"), "embedding"},
		{"no_results_milvus", fmt.Errorf("no results above similarity threshold %.3f", 0.7), "no_results"},
		{"no_results_openai", errors.New("no content found in search results"), "no_results"},
		{"no_results_pinecone", errors.New("no matches in Pinecone response"), "no_results"},
		{"backend_unavailable", errors.New("milvus connection not available (reuse_cache_connection=true)"), "backend_unavailable"},
		{"not_initialized", errors.New("vector store manager not initialized"), "backend_unavailable"},
		{"config", errors.New("invalid Milvus RAG config: missing collection"), "config"},
		{"required", errors.New("qdrant collection name is required"), "config"},
		{"backend_error", errors.New("qdrant search failed: rpc error"), "backend_error"},
		{"backend_error_status", errors.New("API returned status 500: boom"), "backend_error"},
		{"other", errors.New("totally unexpected condition"), "other"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyRAGError(tc.err); got != tc.want {
				t.Fatalf("classifyRAGError(%v) = %q, want %q", tc.err, got, tc.want)
			}
		})
	}
}

// Wrapped errors (the orchestrator wraps backend errors with a prefix) must
// still classify to the underlying cause.
func TestClassifyRAGError_Wrapped(t *testing.T) {
	wrapped := fmt.Errorf("backend %q retrieval failed: %w", "milvus", errors.New("failed to generate embedding"))
	if got := classifyRAGError(wrapped); got != "embedding" {
		t.Fatalf("wrapped classifyRAGError = %q, want embedding", got)
	}
}
