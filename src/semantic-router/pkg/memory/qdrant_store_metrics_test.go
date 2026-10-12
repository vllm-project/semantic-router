package memory

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
)

// qdrantStoreWithUnreachableBackend builds an enabled store whose client
// refuses every call, so each write path takes its error branch.
func qdrantStoreWithUnreachableBackend(t *testing.T) *QdrantStore {
	t.Helper()

	client, err := qdrant.NewClient(&qdrant.Config{Host: "127.0.0.1", Port: 1})
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })

	return &QdrantStore{
		client:         client,
		collectionName: "agentic_memory",
		enabled:        true,
	}
}

// qdrantStoreLatencySeconds returns the summed latency observed so far for one
// (backend, operation) pair of the store-operation histogram.
func qdrantStoreLatencySeconds(t *testing.T, operation string) float64 {
	t.Helper()

	families, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)

	for _, family := range families {
		if family.GetName() != "llm_memory_retrieval_latency_seconds" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := make(map[string]string, 2)
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["backend"] == "qdrant" && labels["operation"] == operation {
				return metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return 0
}

// TestQdrantStoreWriteFailuresAreTimedAndCounted covers the write methods that
// used to record a single hardcoded success with a zero duration, so failures
// were invisible and successful writes looked instant.
func TestQdrantStoreWriteFailuresAreTimedAndCounted(t *testing.T) {
	store := qdrantStoreWithUnreachableBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cases := []struct {
		operation string
		call      func() error
	}{
		{
			operation: "store",
			call: func() error {
				return store.Store(ctx, &Memory{
					ID:        "memory-1",
					UserID:    "user-1",
					Content:   "a memory",
					Embedding: []float32{0.1, 0.2},
				})
			},
		},
		{
			operation: "update",
			call: func() error {
				return store.Update(ctx, "memory-1", &Memory{
					Content:   "a memory",
					Embedding: []float32{0.1, 0.2},
				})
			},
		},
		{
			operation: "forget",
			call:      func() error { return store.Forget(ctx, "memory-1") },
		},
		{
			operation: "forget_by_scope",
			call: func() error {
				return store.ForgetByScope(ctx, MemoryScope{UserID: "user-1"})
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.operation, func(t *testing.T) {
			errorCounter := MemoryStoreOperations.WithLabelValues("qdrant", tc.operation, "error")
			countBefore := testutil.ToFloat64(errorCounter)
			latencyBefore := qdrantStoreLatencySeconds(t, tc.operation)

			require.Error(t, tc.call(), "the unreachable backend must fail the write")

			require.Equal(t, countBefore+1, testutil.ToFloat64(errorCounter),
				"a failed %s must be counted as an error", tc.operation)

			require.Greater(t, qdrantStoreLatencySeconds(t, tc.operation), latencyBefore,
				"a failed %s must record how long it took, not zero", tc.operation)
		})
	}
}
