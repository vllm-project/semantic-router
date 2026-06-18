//go:build !windows && cgo

package apiserver

import (
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestApplyRerankDefaultsSetsModelDimensionAndTopN(t *testing.T) {
	req := RerankRequest{Documents: []string{"a", "b", "c"}}

	applyRerankDefaults(&req)

	if req.Model != "auto" {
		t.Fatalf("expected default model 'auto', got %q", req.Model)
	}
	if req.Dimension != defaultEmbeddingDimension {
		t.Fatalf("expected default dimension %d, got %d", defaultEmbeddingDimension, req.Dimension)
	}
	if req.TopN != len(req.Documents) {
		t.Fatalf("expected default top_n to equal document count %d, got %d", len(req.Documents), req.TopN)
	}
	if req.QualityPriority != defaultEmbeddingPriority || req.LatencyPriority != defaultEmbeddingPriority {
		t.Fatalf("expected default priorities %.2f, got q=%.2f l=%.2f",
			defaultEmbeddingPriority, req.QualityPriority, req.LatencyPriority)
	}
}

func TestValidateRerankRequestRejectsEmptyQuery(t *testing.T) {
	req := RerankRequest{Documents: []string{"a"}, Dimension: defaultEmbeddingDimension}

	code, message, ok := validateRerankRequest(req)
	if ok {
		t.Fatalf("expected empty query to be invalid")
	}
	if code != "INVALID_INPUT" || message != "query must be provided" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}

func TestValidateRerankRequestRejectsEmptyDocuments(t *testing.T) {
	req := RerankRequest{Query: "q", Dimension: defaultEmbeddingDimension}

	code, _, ok := validateRerankRequest(req)
	if ok {
		t.Fatalf("expected empty documents to be invalid")
	}
	if code != "INVALID_INPUT" {
		t.Fatalf("expected INVALID_INPUT, got %q", code)
	}
}

func TestValidateRerankRequestRejectsNegativeTopN(t *testing.T) {
	req := RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b"},
		TopN:      -1,
		Dimension: defaultEmbeddingDimension,
	}

	code, message, ok := validateRerankRequest(req)
	if ok {
		t.Fatalf("expected negative top_n to be invalid")
	}
	if code != "INVALID_INPUT" || message != "top_n cannot be negative" {
		t.Fatalf("unexpected validation error %q: %q", code, message)
	}
}

func TestNormalizeRerankLimitCapsTopNAtDocumentCount(t *testing.T) {
	req := RerankRequest{Documents: []string{"a", "b"}, TopN: 10}

	normalizeRerankLimit(&req)

	if req.TopN != 2 {
		t.Fatalf("expected top_n to be capped at document count, got %d", req.TopN)
	}
}

func TestBuildRerankResultsRejectsInvalidIndex(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 5, Similarity: 0.9},
		},
	}

	if _, err := buildRerankResults(result, []string{"a", "b"}, false); err == nil {
		t.Fatalf("expected out-of-range index to return an error")
	}
}

func TestBuildRerankResultsPreservesScoreOrderAndOmitsDocsByDefault(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 2, Similarity: 0.91},
			{Index: 0, Similarity: 0.40},
		},
	}

	results, err := buildRerankResults(result, []string{"doc0", "doc1", "doc2"}, false)
	if err != nil {
		t.Fatalf("expected valid results, got %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Index != 2 || results[0].RelevanceScore != 0.91 {
		t.Fatalf("expected top result index=2 score=0.91, got %+v", results[0])
	}
	if results[0].Document != nil {
		t.Fatalf("expected no document echoed when return_documents=false, got %+v", results[0].Document)
	}
}

func TestBuildRerankResultsEchoesDocumentsWhenRequested(t *testing.T) {
	result := &candle_binding.BatchSimilarityOutput{
		Matches: []candle_binding.BatchSimilarityMatch{
			{Index: 1, Similarity: 0.8},
		},
	}

	results, err := buildRerankResults(result, []string{"first", "second"}, true)
	if err != nil {
		t.Fatalf("expected valid results, got %v", err)
	}
	if results[0].Document == nil || results[0].Document.Text != "second" {
		t.Fatalf("expected echoed document 'second', got %+v", results[0].Document)
	}
}

func TestRequestsCrossEncoderMatchesAliasAndServedName(t *testing.T) {
	prev := crossEncoderServedName
	defer func() { crossEncoderServedName = prev }()

	crossEncoderServedName = ""
	if !requestsCrossEncoder("cross-encoder") {
		t.Fatalf("expected the 'cross-encoder' alias to route to the cross-encoder backend")
	}
	if requestsCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") {
		t.Fatalf("expected a real model id to fall through when no served name is configured")
	}
	if requestsCrossEncoder("auto") {
		t.Fatalf("expected 'auto' to use the bi-encoder path")
	}

	crossEncoderServedName = "cross-encoder/ms-marco-MiniLM-L-6-v2"
	if !requestsCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") {
		t.Fatalf("expected the configured served model id to route to the cross-encoder backend")
	}
	if !requestsCrossEncoder("cross-encoder") {
		t.Fatalf("expected the alias to keep working alongside a configured served name")
	}
	if requestsCrossEncoder("BAAI/bge-reranker-v2-m3") {
		t.Fatalf("expected an unconfigured model id to use the bi-encoder path")
	}
}

func TestServedRerankModelNamePrefersConfiguredName(t *testing.T) {
	prev := crossEncoderServedName
	defer func() { crossEncoderServedName = prev }()

	crossEncoderServedName = ""
	if got := servedRerankModelName(); got != "cross-encoder" {
		t.Fatalf("expected alias fallback 'cross-encoder', got %q", got)
	}

	crossEncoderServedName = "cross-encoder/ms-marco-MiniLM-L-6-v2"
	if got := servedRerankModelName(); got != crossEncoderServedName {
		t.Fatalf("expected served name %q, got %q", crossEncoderServedName, got)
	}
}

func TestBuildRerankResultsFromCrossEncoderRejectsInvalidIndex(t *testing.T) {
	out := &candle_binding.RerankOutput{
		Matches: []candle_binding.RerankMatch{{Index: 7, Score: 0.9}},
	}

	if _, err := buildRerankResultsFromCrossEncoder(out, []string{"a", "b"}, false); err == nil {
		t.Fatalf("expected out-of-range index to return an error")
	}
}

func TestBuildRerankResultsFromCrossEncoderPreservesScoreOrderAndEchoesDocs(t *testing.T) {
	out := &candle_binding.RerankOutput{
		Matches: []candle_binding.RerankMatch{
			{Index: 2, Score: 0.99},
			{Index: 0, Score: 0.02},
		},
	}

	results, err := buildRerankResultsFromCrossEncoder(out, []string{"doc0", "doc1", "doc2"}, true)
	if err != nil {
		t.Fatalf("expected valid results, got %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Index != 2 || results[0].RelevanceScore != 0.99 {
		t.Fatalf("expected top result index=2 score=0.99, got %+v", results[0])
	}
	if results[0].Document == nil || results[0].Document.Text != "doc2" {
		t.Fatalf("expected echoed document 'doc2', got %+v", results[0].Document)
	}
}
