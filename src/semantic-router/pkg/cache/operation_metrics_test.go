//go:build !windows && cgo

package cache

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const measuredUpsertDelay = 5 * time.Millisecond

type cacheMetricsPointsServer struct {
	qdrant.UnimplementedPointsServer
	fail bool
}

func (s *cacheMetricsPointsServer) Upsert(ctx context.Context, _ *qdrant.UpsertPoints) (*qdrant.PointsOperationResponse, error) {
	timer := time.NewTimer(measuredUpsertDelay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
	}
	if s.fail {
		return nil, status.Error(codes.Internal, "fixture write failure")
	}
	return &qdrant.PointsOperationResponse{Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed}}, nil
}

func (*cacheMetricsPointsServer) Scroll(context.Context, *qdrant.ScrollPoints) (*qdrant.ScrollResponse, error) {
	return &qdrant.ScrollResponse{Result: []*qdrant.RetrievedPoint{{
		Id: qdrant.NewIDNum(1),
		Vectors: &qdrant.VectorsOutput{VectorsOptions: &qdrant.VectorsOutput_Vector{
			Vector: &qdrant.VectorOutput{Vector: &qdrant.VectorOutput_Dense{Dense: &qdrant.DenseVector{Data: []float32{1, 0}}}},
		}},
		Payload: qdrant.NewValueMap(map[string]any{"model": "fixture", "query": "fixture", "request_body": "{}"}),
	}}}, nil
}

func newMetricsQdrantCache(t *testing.T, fail bool) *QdrantCache {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	qdrant.RegisterPointsServer(server, &cacheMetricsPointsServer{fail: fail})
	t.Cleanup(server.Stop)
	go func() { _ = server.Serve(listener) }()
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: "127.0.0.1", Port: listener.Addr().(*net.TCPAddr).Port,
		PoolSize: 1, SkipCompatibilityCheck: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return &QdrantCache{
		enabled: true, collectionName: "fixture", client: client,
		embeddingProvider: cacheTestEmbeddingProvider(), embeddingModel: "bert",
	}
}

func cacheOperationHistogram(t *testing.T, backend, operation string) *dto.Histogram {
	t.Helper()
	var value dto.Metric
	if err := metrics.CacheOperationDuration.WithLabelValues(backend, operation).(prometheus.Metric).Write(&value); err != nil {
		t.Fatal(err)
	}
	return value.GetHistogram()
}

func TestQdrantWritesObserveMeasuredDuration(t *testing.T) {
	for _, failed := range []bool{false, true} {
		outcome := "success"
		if failed {
			outcome = "error"
		}
		t.Run(outcome, func(t *testing.T) {
			cache := newMetricsQdrantCache(t, failed)
			operations := []struct {
				name string
				call func() error
			}{
				{"add_pending", func() error { return cache.AddPendingRequest("pending", "fixture", "fixture", []byte(`{}`), 60) }},
				{"update_response", func() error { return cache.UpdateWithResponse("pending", []byte(`{}`), 60) }},
				{"add_entry", func() error {
					return cache.AddEntry(context.Background(), "entry", "fixture", "fixture", []byte(`{}`), []byte(`{}`), 60)
				}},
			}
			for _, operation := range operations {
				t.Run(operation.name, func(t *testing.T) {
					before := cacheOperationHistogram(t, "qdrant", operation.name)
					counter := metrics.CacheOperationTotal.WithLabelValues("qdrant", operation.name, outcome)
					count := testutil.ToFloat64(counter)
					err := operation.call()
					if (err != nil) != failed {
						t.Fatalf("cache operation error = %v, want failure %v", err, failed)
					}
					after := cacheOperationHistogram(t, "qdrant", operation.name)
					if after.GetSampleCount()-before.GetSampleCount() != 1 || testutil.ToFloat64(counter)-count != 1 {
						t.Fatal("cache operation must count and observe exactly once")
					}
					if elapsed := after.GetSampleSum() - before.GetSampleSum(); elapsed < measuredUpsertDelay.Seconds() {
						t.Fatalf("cache latency %gs does not include the actual RPC delay", elapsed)
					}
				})
			}
		})
	}
}

func TestInMemoryEvictionObservesMeasuredDuration(t *testing.T) {
	cache := NewInMemoryCache(InMemoryCacheOptions{EmbeddingProvider: cacheTestEmbeddingProvider(), Enabled: true, MaxEntries: 1})
	t.Cleanup(func() { _ = cache.Close() })
	if err := cache.AddEntry(context.Background(), "first", "fixture", "first", nil, []byte(`{}`), 60); err != nil {
		t.Fatal(err)
	}
	before := cacheOperationHistogram(t, "memory", "evict")
	if err := cache.AddEntry(context.Background(), "second", "fixture", "second", nil, []byte(`{}`), 60); err != nil {
		t.Fatal(err)
	}
	after := cacheOperationHistogram(t, "memory", "evict")
	if after.GetSampleCount()-before.GetSampleCount() != 1 || after.GetSampleSum() <= before.GetSampleSum() {
		t.Fatal("eviction must observe its actual positive duration once")
	}
}
