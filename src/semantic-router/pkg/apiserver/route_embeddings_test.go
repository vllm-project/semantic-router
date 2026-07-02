//go:build !windows && cgo

package apiserver

import (
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

func TestValidateSimilarityRequest(t *testing.T) {
	cases := []struct {
		name     string
		req      SimilarityRequest
		wantOK   bool
		wantCode string
	}{
		{"valid", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension}, true, ""},
		{"empty_text1", SimilarityRequest{Text1: "", Text2: "b", Dimension: defaultEmbeddingDimension}, false, "INVALID_INPUT"},
		{"whitespace_text2", SimilarityRequest{Text1: "a", Text2: "   ", Dimension: defaultEmbeddingDimension}, false, "INVALID_INPUT"},
		{"bad_dimension", SimilarityRequest{Text1: "a", Text2: "b", Dimension: 999}, false, "INVALID_DIMENSION"},
		{"dimension_64_allowed", SimilarityRequest{Text1: "a", Text2: "b", Dimension: 64}, true, ""},
		{"quality_priority_too_high", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension, QualityPriority: 1.5}, false, "INVALID_PARAMETER"},
		{"latency_priority_negative", SimilarityRequest{Text1: "a", Text2: "b", Dimension: defaultEmbeddingDimension, LatencyPriority: -0.1}, false, "INVALID_PARAMETER"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			code, _, ok := validateSimilarityRequest(tc.req)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if code != tc.wantCode {
				t.Fatalf("code = %q, want %q", code, tc.wantCode)
			}
		})
	}
}

func TestValidateBatchSimilarityRequestRejectsBlankAndOutOfRange(t *testing.T) {
	base := func() BatchSimilarityRequest {
		return BatchSimilarityRequest{Query: "q", Candidates: []string{"a", "b"}, Dimension: defaultEmbeddingDimension}
	}
	cases := []struct {
		name     string
		mutate   func(*BatchSimilarityRequest)
		wantOK   bool
		wantCode string
	}{
		{"valid", func(*BatchSimilarityRequest) {}, true, ""},
		{"whitespace_query", func(r *BatchSimilarityRequest) { r.Query = "  " }, false, "INVALID_INPUT"},
		{"blank_candidate", func(r *BatchSimilarityRequest) { r.Candidates = []string{"a", " "} }, false, "INVALID_INPUT"},
		{"quality_priority_too_high", func(r *BatchSimilarityRequest) { r.QualityPriority = 2 }, false, "INVALID_PARAMETER"},
		{"latency_priority_negative", func(r *BatchSimilarityRequest) { r.LatencyPriority = -1 }, false, "INVALID_PARAMETER"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := base()
			tc.mutate(&req)
			code, _, ok := validateBatchSimilarityRequest(req)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if code != tc.wantCode {
				t.Fatalf("code = %q, want %q", code, tc.wantCode)
			}
		})
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
