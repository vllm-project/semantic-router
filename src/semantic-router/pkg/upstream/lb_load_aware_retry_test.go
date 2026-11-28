package upstream

import (
	"math"
	"net/http"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestLoadAwareRetryPrefersAnUntriedReplica(t *testing.T) {
	failed := endpointOf(t, "a", backend(t, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	healthy := endpointOf(t, "b", backend(t, func(w http.ResponseWriter, _ *http.Request) {}))
	failed.Telemetry, healthy.Telemetry = telemetryID("a"), telemetryID("b")
	spec := retrying("load-aware-retry", "5xx", 1, failed, healthy)
	spec.LBPolicy = LBLoadAware
	set := newSet(t, Options{Clock: &waitRecorder{}, Telemetry: fakeTelemetry{
		telemetryID("a"): observation("a", 0), telemetryID("b"): observation("b", 5),
	}}, spec)
	before := testutil.ToFloat64(metrics.UpstreamLoadAwarePicksTotal.WithLabelValues(spec.Name, loadAwareTelemetry))
	resp, err := set.Do(t.Context(), post(spec.Name))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if len(resp.Attempts) != 2 || resp.Attempts[0].Endpoint != "a" || resp.Endpoint.Name != "b" || resp.StatusCode != http.StatusOK {
		t.Fatalf("retry returned to the lowest-load failed replica: %+v", resp)
	}
	after := testutil.ToFloat64(metrics.UpstreamLoadAwarePicksTotal.WithLabelValues(spec.Name, loadAwareTelemetry))
	if after-before != 2 {
		t.Fatalf("pick count = %v, want one per attempt", after-before)
	}
}

func TestLoadAwareRetryChecksTheWholePoolAndCanReuseExhaustedHosts(t *testing.T) {
	hosts := loadAwareEndpoints(2)
	balancer := newLoadAware()
	telemetry := fakeTelemetry{telemetryID("a"): observation("a", 0), telemetryID("b"): observation("b", 5)}
	if got, _ := balancer.pickByTelemetryAvoiding(newHostSet(hosts), telemetry, hosts); got != hosts[0] {
		t.Fatal("exhausting the pool should permit another attempt on the best host")
	}
	delete(telemetry, telemetryID("a"))
	if got, outcome := balancer.pickByTelemetryAvoiding(newHostSet(hosts), telemetry, hosts[:1]); got != nil || outcome != loadAwareStale {
		t.Fatal("excluding a tried host hid incomplete telemetry for the pool")
	}
}

func TestLoadAwareScoreCannotWrapToAnIdleReplica(t *testing.T) {
	hosts := loadAwareEndpoints(2)
	hosts[0].active.Store(2)
	telemetry := fakeTelemetry{telemetryID("a"): observation("a", math.MaxInt64), telemetryID("b"): observation("b", 5)}
	if got := newLoadAware().pick(newHostSet(hosts), telemetry); got != hosts[1] {
		t.Fatal("an overflowing busy score was treated as idle")
	}
}
