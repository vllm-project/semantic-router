package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestRecordRAGRetrieval_IncrementsAttemptsByStatus(t *testing.T) {
	RAGRetrievalAttempts.Reset()

	RecordRAGRetrieval("milvus", "factual_lookup", "success", 0.012)
	RecordRAGRetrieval("milvus", "factual_lookup", "success", 0.020)
	RecordRAGRetrieval("milvus", "factual_lookup", "error", 0)

	got := testutil.ToFloat64(RAGRetrievalAttempts.WithLabelValues("milvus", "factual_lookup", "success"))
	if got != 2 {
		t.Fatalf("expected success attempts=2, got %v", got)
	}
	got = testutil.ToFloat64(RAGRetrievalAttempts.WithLabelValues("milvus", "factual_lookup", "error"))
	if got != 1 {
		t.Fatalf("expected error attempts=1, got %v", got)
	}
}

func TestRecordRAGCacheHitMiss(t *testing.T) {
	RAGCacheHits.Reset()
	RAGCacheMisses.Reset()

	RecordRAGCacheHit("qdrant")
	RecordRAGCacheHit("qdrant")
	RecordRAGCacheMiss("qdrant")

	if got := testutil.ToFloat64(RAGCacheHits.WithLabelValues("qdrant")); got != 2 {
		t.Fatalf("expected cache hits=2, got %v", got)
	}
	if got := testutil.ToFloat64(RAGCacheMisses.WithLabelValues("qdrant")); got != 1 {
		t.Fatalf("expected cache misses=1, got %v", got)
	}
}

func TestRecordRAGResultCount_ObservesHistogram(t *testing.T) {
	RAGResultCount.Reset()

	RecordRAGResultCount("vectorstore", "research", 3)
	RecordRAGResultCount("vectorstore", "research", 7)

	count := testutil.CollectAndCount(RAGResultCount)
	if count != 1 {
		t.Fatalf("expected 1 result-count series, got %d", count)
	}
}

func TestRecordRAGContextTruncation_IncrementsCounter(t *testing.T) {
	RAGContextTruncations.Reset()

	RecordRAGContextTruncation("milvus", "domain_expertise")
	RecordRAGContextTruncation("milvus", "domain_expertise")

	got := testutil.ToFloat64(RAGContextTruncations.WithLabelValues("milvus", "domain_expertise"))
	if got != 2 {
		t.Fatalf("expected truncations=2, got %v", got)
	}
}

func TestRecordRAGRetrievalError_BreaksDownByType(t *testing.T) {
	RAGRetrievalErrors.Reset()

	RecordRAGRetrievalError("milvus", "factual_lookup", "embedding")
	RecordRAGRetrievalError("milvus", "factual_lookup", "no_results")
	RecordRAGRetrievalError("milvus", "factual_lookup", "no_results")

	if got := testutil.ToFloat64(RAGRetrievalErrors.WithLabelValues("milvus", "factual_lookup", "embedding")); got != 1 {
		t.Fatalf("expected embedding errors=1, got %v", got)
	}
	if got := testutil.ToFloat64(RAGRetrievalErrors.WithLabelValues("milvus", "factual_lookup", "no_results")); got != 2 {
		t.Fatalf("expected no_results errors=2, got %v", got)
	}
}
