//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestBuildBatchSimilarityMatchesRejectsInvalidNativeIndex(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 2, Similarity: 0.9},
		},
	}

	if _, err := buildBatchSimilarityMatches(result, []string{"a", "b"}); err == nil {
		t.Fatalf("expected invalid native match index to return an error")
	}
}

func TestBuildBatchSimilarityMatchesIncludesCandidateText(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 1, Similarity: 0.9},
			{Index: 0, Similarity: 0.7},
		},
	}

	matches, err := buildBatchSimilarityMatches(result, []string{"first", "second"})
	if err != nil {
		t.Fatalf("expected valid native matches, got %v", err)
	}

	if matches[0].Text != "second" || matches[1].Text != "first" {
		t.Fatalf("expected candidate text to follow native indexes, got %+v", matches)
	}
}

func TestNormalizeBatchSimilarityLimitCapsTopKAtCandidateCount(t *testing.T) {
	req := BatchSimilarityRequest{
		Candidates: []string{"a", "b"},
		TopK:       10,
	}

	normalizeBatchSimilarityLimit(&req)

	if req.TopK != 2 {
		t.Fatalf("expected top_k to be capped at candidate count, got %d", req.TopK)
	}
}

func TestValidateBatchSimilarityRequestRejectsNegativeTopK(t *testing.T) {
	req := BatchSimilarityRequest{
		Query:      "query",
		Candidates: []string{"a", "b"},
		TopK:       -1,
		Dimension:  defaultEmbeddingDimension,
	}

	code, message, ok := validateBatchSimilarityRequest(req)
	if ok {
		t.Fatalf("expected negative top_k to be invalid")
	}
	if code != "INVALID_INPUT" || message != "top_k cannot be negative" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}

func TestEmbeddingEndpointsReturn503WhenNotReady(t *testing.T) {
	if candle_binding.IsEmbeddingReady() {
		t.Skip("test requires embedding models to be uninitialized")
	}

	s := &ClassificationAPIServer{}

	tests := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{
			"embeddings",
			"/api/v1/embeddings",
			`{"texts":["hi"]}`,
			s.handleEmbeddings,
		},
		{
			"similarity",
			"/api/v1/similarity",
			`{"text1":"hello","text2":"world"}`,
			s.handleSimilarity,
		},
		{
			"batch similarity",
			"/api/v1/batch-similarity",
			`{"query":"hello","candidates":["world"]}`,
			s.handleBatchSimilarity,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()

			tc.handler(rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected 503, got %d: %s", rr.Code, rr.Body.String())
			}

			if !strings.Contains(rr.Body.String(), "EMBEDDING_NOT_READY") {
				t.Fatalf("expected EMBEDDING_NOT_READY, got: %s", rr.Body.String())
			}
		})
	}
}