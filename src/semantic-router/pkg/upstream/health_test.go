package upstream

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// switchable answers health checks with a status the test sets, or drops the
// connection when the status is zero, and counts the checks it saw.
type switchable struct {
	status atomic.Int64
	checks atomic.Int32
	host   atomic.Value
}

func newSwitchable(status int) *switchable {
	s := &switchable{}
	s.status.Store(int64(status))
	return s
}

func (s *switchable) handler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/health" {
		return
	}
	s.checks.Add(1)
	s.host.Store(r.Host + " " + r.Method + " " + r.Header.Get("User-Agent"))
	status := int(s.status.Load())
	if status == 0 {
		hijackAndClose(w)
		return
	}
	w.WriteHeader(status)
}

func healthChecked(name string, endpoints ...EndpointSpec) ClusterSpec {
	spec := clusterOf(name, endpoints...)
	spec.HealthCheck = &HealthCheckSpec{Path: "/health", Interval: 10 * time.Second}
	return spec
}

func TestHealthChecksStartFailedAndWarmOnTheFirstPass(t *testing.T) {
	up, down := newSwitchable(200), newSwitchable(503)
	a := endpointOf(t, "a", backend(t, up.handler))
	b := endpointOf(t, "b", backend(t, down.handler))
	clock := newManualClock()
	set := newSet(t, Options{Clock: clock}, healthChecked("hc-warm", a, b))
	c := set.clusters["hc-warm"]

	if inRotation(c) != "ab" {
		t.Fatalf("before any check every endpoint is pending and the cluster panics; rotation = %s", inRotation(c))
	}
	early, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	if err := set.Warm(early); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Warm before the first round = %v", err)
	}
	clock.Advance(0)
	if err := set.Warm(t.Context()); err != nil {
		t.Fatalf("Warm after the first round: %v", err)
	}
	if inRotation(c) != "a" {
		t.Fatalf("rotation = %s, want only the endpoint that passed", inRotation(c))
	}
	if got := up.host.Load(); got != a.Authority()+" GET vllm-semantic-router/health-check" {
		t.Fatalf("health check request = %v", got)
	}
	if testutil.ToFloat64(metrics.UpstreamHealthChecksTotal.WithLabelValues("hc-warm", "b", "status")) != 1 {
		t.Fatal("the failed check was not counted")
	}
}

func TestHealthCheckThresholdsFollowEnvoy(t *testing.T) {
	flaky, steady := newSwitchable(200), newSwitchable(200)
	a := endpointOf(t, "a", backend(t, flaky.handler))
	b := endpointOf(t, "b", backend(t, steady.handler))
	clock := newManualClock()
	set := newSet(t, Options{Clock: clock}, healthChecked("hc-thresholds", a, b))
	c := set.clusters["hc-thresholds"]
	clock.Advance(0)

	// Network failures need the unhealthy threshold (3) in a row. The
	// cluster has served nothing yet, so checks run every 60s.
	flaky.status.Store(0)
	for i := range 3 {
		if inRotation(c) != "ab" {
			t.Fatalf("after %d network failures rotation = %s, want both", i, inRotation(c))
		}
		clock.Advance(60 * time.Second)
	}
	if inRotation(c) != "b" {
		t.Fatalf("after 3 network failures rotation = %s, want only b", inRotation(c))
	}
	// One pass returns it (healthy threshold 1).
	flaky.status.Store(200)
	clock.Advance(60 * time.Second)
	if inRotation(c) != "ab" {
		t.Fatalf("after a pass rotation = %s", inRotation(c))
	}
	// Any status but 200 fails it at once.
	flaky.status.Store(500)
	clock.Advance(60 * time.Second)
	if inRotation(c) != "b" {
		t.Fatalf("after a 500 rotation = %s", inRotation(c))
	}
}

func TestHealthCheckIntervalShortensOnceTheClusterServes(t *testing.T) {
	probe := newSwitchable(200)
	server := backend(t, probe.handler)
	clock := newManualClock()
	set := newSet(t, Options{Clock: clock}, healthChecked("hc-interval", endpointOf(t, "a", server)))
	clock.Advance(0)
	resp, err := set.Do(t.Context(), post("hc-interval"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)

	clock.Advance(59 * time.Second)
	if n := probe.checks.Load(); n != 1 {
		t.Fatalf("checks = %d, want only the first before the 60s no-traffic interval", n)
	}
	clock.Advance(time.Second)
	clock.Advance(10 * time.Second)
	if n := probe.checks.Load(); n != 3 {
		t.Fatalf("checks = %d, want the 10s interval once the cluster served traffic", n)
	}
}

func TestHealthCheckPassLiftsAnOutlierEjection(t *testing.T) {
	probe := newSwitchable(200)
	a := endpointOf(t, "a", backend(t, probe.handler))
	b := endpointOf(t, "b", backend(t, newSwitchable(200).handler))
	spec := healthChecked("hc-uneject", a, b)
	// A long ejection, so only the passing check can bring a back.
	spec.Outlier = &OutlierSpec{Consecutive5xx: 2, BaseEjectionTime: 10 * time.Minute}
	clock := newManualClock()
	set := newSet(t, Options{Clock: clock}, spec)
	c := set.clusters["hc-uneject"]
	clock.Advance(0)

	observeN(c, "a", 500, 2)
	if inRotation(c) != "b" {
		t.Fatalf("rotation = %s, want a ejected", inRotation(c))
	}
	clock.Advance(60 * time.Second)
	if inRotation(c) != "ab" {
		t.Fatalf("rotation = %s, want a back after a passing check", inRotation(c))
	}
}
