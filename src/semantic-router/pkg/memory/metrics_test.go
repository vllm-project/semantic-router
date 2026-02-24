package memory

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// TestMemoryMetricsInitialization verifies that all metrics are properly initialized
func TestMemoryMetricsInitialization(t *testing.T) {
	tests := []struct {
		name   string
		metric prometheus.Collector
	}{
		{"MemoryRetrievalLatency", MemoryRetrievalLatency},
		{"MemoryRetrievalCount", MemoryRetrievalCount},
		{"MemoryRetrievalResults", MemoryRetrievalResults},
		{"MemoryExtractionCount", MemoryExtractionCount},
		{"MemoryExtractionLatency", MemoryExtractionLatency},
		{"MemoryExtractionFactsCount", MemoryExtractionFactsCount},
		{"MemoryStoreOperations", MemoryStoreOperations},
		{"MemoryStoreSize", MemoryStoreSize},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.metric == nil {
				t.Errorf("%s should not be nil", tt.name)
			}
		})
	}
}

// TestRecordMemoryRetrieval tests the retrieval metrics recording
func TestRecordMemoryRetrieval(t *testing.T) {
	tests := []struct {
		name        string
		backend     string
		operation   string
		status      string
		userID      string
		duration    float64
		resultCount int
	}{
		{
			name:        "successful retrieval with results",
			backend:     "milvus",
			operation:   "retrieve",
			status:      "hit",
			userID:      "user_123",
			duration:    0.025,
			resultCount: 5,
		},
		{
			name:        "retrieval with no results",
			backend:     "milvus",
			operation:   "retrieve",
			status:      "miss",
			userID:      "user_456",
			duration:    0.015,
			resultCount: 0,
		},
		{
			name:        "retrieval error",
			backend:     "milvus",
			operation:   "retrieve",
			status:      "error",
			userID:      "user_789",
			duration:    0.001,
			resultCount: -1, // No result count on error
		},
		{
			name:        "empty values use defaults",
			backend:     "",
			operation:   "",
			status:      "",
			userID:      "",
			duration:    0.010,
			resultCount: 3,
		},
		{
			name:        "zero result count is recorded",
			backend:     "milvus",
			operation:   "retrieve",
			status:      "miss",
			userID:      "user_zero",
			duration:    0.001,
			resultCount: 0,
		},
		{
			name:        "large duration",
			backend:     "milvus",
			operation:   "retrieve",
			status:      "hit",
			userID:      "user_slow",
			duration:    30.0,
			resultCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RecordMemoryRetrieval(tt.backend, tt.operation, tt.status, tt.userID, tt.duration, tt.resultCount)
		})
	}
}

// TestRecordMemoryExtraction tests the extraction metrics recording
func TestRecordMemoryExtraction(t *testing.T) {
	tests := []struct {
		name       string
		status     string
		duration   float64
		factsCount int
		factType   string
	}{
		{
			name:       "successful extraction with semantic facts",
			status:     "success",
			duration:   1.5,
			factsCount: 3,
			factType:   "semantic",
		},
		{
			name:       "successful extraction with procedural facts",
			status:     "success",
			duration:   2.0,
			factsCount: 2,
			factType:   "procedural",
		},
		{
			name:       "extraction with no facts",
			status:     "success",
			duration:   0.8,
			factsCount: 0,
			factType:   "all",
		},
		{
			name:       "extraction error",
			status:     "error",
			duration:   0.5,
			factsCount: -1,
			factType:   "",
		},
		{
			name:       "skipped extraction",
			status:     "skipped",
			duration:   0.001,
			factsCount: 0,
			factType:   "all",
		},
		{
			name:       "empty factType does not observe facts histogram",
			status:     "success",
			duration:   0.5,
			factsCount: 2,
			factType:   "",
		},
		{
			name:       "negative factsCount does not observe facts histogram",
			status:     "error",
			duration:   0.1,
			factsCount: -1,
			factType:   "semantic",
		},
		{
			name:       "zero duration",
			status:     "success",
			duration:   0,
			factsCount: 0,
			factType:   "all",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RecordMemoryExtraction(tt.status, tt.duration, tt.factsCount, tt.factType)
		})
	}
}

// TestRecordMemoryStoreOperation tests the store operation metrics recording
func TestRecordMemoryStoreOperation(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		operation string
		status    string
		duration  float64
	}{
		{
			name:      "successful store",
			backend:   "milvus",
			operation: "store",
			status:    "success",
			duration:  0.050,
		},
		{
			name:      "successful update",
			backend:   "milvus",
			operation: "update",
			status:    "success",
			duration:  0.075,
		},
		{
			name:      "successful forget",
			backend:   "milvus",
			operation: "forget",
			status:    "success",
			duration:  0.020,
		},
		{
			name:      "successful forget_by_scope",
			backend:   "milvus",
			operation: "forget_by_scope",
			status:    "success",
			duration:  0.100,
		},
		{
			name:      "store error",
			backend:   "milvus",
			operation: "store",
			status:    "error",
			duration:  0.005,
		},
		{
			name:      "empty backend and status use defaults",
			backend:   "",
			operation: "",
			status:    "",
			duration:  0.010,
		},
		{
			name:      "zero duration",
			backend:   "milvus",
			operation: "store",
			status:    "success",
			duration:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RecordMemoryStoreOperation(tt.backend, tt.operation, tt.status, tt.duration)
		})
	}
}

// TestUpdateMemoryStoreSize tests the store size gauge update
func TestUpdateMemoryStoreSize(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		userID  string
		count   int
	}{
		{
			name:    "user with 10 memories",
			backend: "milvus",
			userID:  "user_123",
			count:   10,
		},
		{
			name:    "user with 0 memories",
			backend: "milvus",
			userID:  "user_456",
			count:   0,
		},
		{
			name:    "user with many memories",
			backend: "milvus",
			userID:  "user_789",
			count:   1000,
		},
		{
			name:    "empty values use defaults",
			backend: "",
			userID:  "",
			count:   5,
		},
		{
			name:    "negative count sets gauge to negative",
			backend: "milvus",
			userID:  "user_neg",
			count:   -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			UpdateMemoryStoreSize(tt.backend, tt.userID, tt.count)
		})
	}
}

// TestMetricLabels verifies that metrics have the correct labels
func TestMetricLabels(t *testing.T) {
	tests := []struct {
		name           string
		metric         prometheus.Collector
		expectedLabels []string
	}{
		{
			name:           "MemoryRetrievalLatency labels",
			metric:         MemoryRetrievalLatency,
			expectedLabels: []string{"backend", "operation"},
		},
		{
			name:           "MemoryRetrievalCount labels",
			metric:         MemoryRetrievalCount,
			expectedLabels: []string{"backend", "status", "user_id"},
		},
		{
			name:           "MemoryRetrievalResults labels",
			metric:         MemoryRetrievalResults,
			expectedLabels: []string{"backend"},
		},
		{
			name:           "MemoryExtractionCount labels",
			metric:         MemoryExtractionCount,
			expectedLabels: []string{"status"},
		},
		{
			name:           "MemoryExtractionLatency labels",
			metric:         MemoryExtractionLatency,
			expectedLabels: []string{"status"},
		},
		{
			name:           "MemoryExtractionFactsCount labels",
			metric:         MemoryExtractionFactsCount,
			expectedLabels: []string{"type"},
		},
		{
			name:           "MemoryStoreOperations labels",
			metric:         MemoryStoreOperations,
			expectedLabels: []string{"backend", "operation", "status"},
		},
		{
			name:           "MemoryStoreSize labels",
			metric:         MemoryStoreSize,
			expectedLabels: []string{"backend", "user_id"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			desc := make(chan *prometheus.Desc, 1)
			tt.metric.Describe(desc)
			close(desc)

			for d := range desc {
				if d == nil {
					t.Error("Metric descriptor should not be nil")
				}
				// Verify expected label names appear in the descriptor string
				descStr := d.String()
				for _, label := range tt.expectedLabels {
					if !strings.Contains(descStr, label) {
						t.Errorf("expected descriptor to contain label %q, got: %s", label, descStr)
					}
				}
			}
		})
	}
}

// TestRecordMemoryRetrievalIncrementsCounter verifies that recording retrieval updates the counter
func TestRecordMemoryRetrievalIncrementsCounter(t *testing.T) {
	backend, status, userID := "milvus", "hit", "test_retrieval_counter_user"
	before := testutil.ToFloat64(MemoryRetrievalCount.WithLabelValues(backend, status, userID))
	RecordMemoryRetrieval(backend, "retrieve", status, userID, 0.01, 2)
	after := testutil.ToFloat64(MemoryRetrievalCount.WithLabelValues(backend, status, userID))
	if after != before+1 {
		t.Errorf("expected counter to increment by 1, got before=%.0f after=%.0f", before, after)
	}
}

// TestRecordMemoryExtractionIncrementsCounter verifies that recording extraction updates the counter
func TestRecordMemoryExtractionIncrementsCounter(t *testing.T) {
	status := "success"
	before := testutil.ToFloat64(MemoryExtractionCount.WithLabelValues(status))
	RecordMemoryExtraction(status, 0.5, 1, "semantic")
	after := testutil.ToFloat64(MemoryExtractionCount.WithLabelValues(status))
	if after != before+1 {
		t.Errorf("expected extraction counter to increment by 1, got before=%.0f after=%.0f", before, after)
	}
}

// TestRecordMemoryStoreOperationIncrementsCounter verifies that recording store ops updates the counter
func TestRecordMemoryStoreOperationIncrementsCounter(t *testing.T) {
	backend, operation, status := "milvus", "store", "success"
	before := testutil.ToFloat64(MemoryStoreOperations.WithLabelValues(backend, operation, status))
	RecordMemoryStoreOperation(backend, operation, status, 0.05)
	after := testutil.ToFloat64(MemoryStoreOperations.WithLabelValues(backend, operation, status))
	if after != before+1 {
		t.Errorf("expected store operations counter to increment by 1, got before=%.0f after=%.0f", before, after)
	}
}

// TestUpdateMemoryStoreSizeSetsGauge verifies that UpdateMemoryStoreSize sets the gauge
func TestUpdateMemoryStoreSizeSetsGauge(t *testing.T) {
	backend, userID := "milvus", "test_gauge_user"
	count := 42
	UpdateMemoryStoreSize(backend, userID, count)
	got := testutil.ToFloat64(MemoryStoreSize.WithLabelValues(backend, userID))
	if got != float64(count) {
		t.Errorf("expected gauge %.0f, got %.0f", float64(count), got)
	}
}

// TestMetricsExported verifies that all memory metrics are registered and exported
func TestMetricsExported(t *testing.T) {
	expectedNames := []string{
		"llm_memory_retrieval_latency_seconds",
		"llm_memory_retrieval_total",
		"llm_memory_retrieval_results",
		"llm_memory_extraction_total",
		"llm_memory_extraction_latency_seconds",
		"llm_memory_extraction_facts_count",
		"llm_memory_store_operations_total",
		"llm_memory_store_size",
	}

	mfs, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Gather failed: %v", err)
	}

	seen := make(map[string]bool)
	for _, mf := range mfs {
		if mf.Name != nil {
			seen[*mf.Name] = true
		}
	}

	for _, name := range expectedNames {
		if !seen[name] {
			t.Errorf("metric %q not found in gathered metrics", name)
		}
	}
}

// TestMetricsHaveSamplesAfterRecording verifies that after recording, metrics have at least one sample
func TestMetricsHaveSamplesAfterRecording(t *testing.T) {
	// Record one of each so that all three categories have data
	RecordMemoryRetrieval("milvus", "retrieve", "hit", "sample_user", 0.01, 1)
	RecordMemoryExtraction("success", 0.1, 1, "semantic")
	RecordMemoryStoreOperation("milvus", "store", "success", 0.01)

	mfs, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Gather failed: %v", err)
	}

	// Check that our memory metrics appear and have at least one metric each
	wantFamilies := map[string]bool{
		"llm_memory_retrieval_total":        false,
		"llm_memory_extraction_total":       false,
		"llm_memory_store_operations_total": false,
	}
	for _, mf := range mfs {
		if mf.Name == nil {
			continue
		}
		if _, ok := wantFamilies[*mf.Name]; ok && len(mf.Metric) > 0 {
			wantFamilies[*mf.Name] = true
		}
	}
	for name, found := range wantFamilies {
		if !found {
			t.Errorf("metric family %q missing or has no samples after recording", name)
		}
	}
}

// TestMetricsIntegration is an integration test that exercises the full metrics flow
func TestMetricsIntegration(t *testing.T) {
	// Simulate a retrieval operation
	t.Run("retrieval flow", func(t *testing.T) {
		RecordMemoryRetrieval("milvus", "retrieve", "hit", "test_user", 0.025, 5)
		RecordMemoryRetrieval("milvus", "retrieve", "miss", "test_user", 0.010, 0)
		RecordMemoryRetrieval("milvus", "retrieve", "error", "test_user", 0.001, -1)
	})

	// Simulate an extraction operation
	t.Run("extraction flow", func(t *testing.T) {
		RecordMemoryExtraction("success", 1.5, 3, "semantic")
		RecordMemoryExtraction("success", 0, 2, "procedural")
		RecordMemoryExtraction("success", 0, 1, "episodic")
	})

	// Simulate store operations
	t.Run("store operations flow", func(t *testing.T) {
		RecordMemoryStoreOperation("milvus", "store", "success", 0.050)
		RecordMemoryStoreOperation("milvus", "update", "success", 0.075)
		RecordMemoryStoreOperation("milvus", "forget", "success", 0.020)
		UpdateMemoryStoreSize("milvus", "test_user", 10)
	})
}
