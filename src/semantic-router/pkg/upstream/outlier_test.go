package upstream

import (
	"io"
	"math/rand/v2"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// detachedCluster builds a cluster on a manual clock without any network.
func detachedCluster(t *testing.T, spec ClusterSpec, clock Clock) *cluster {
	t.Helper()
	c := newCluster(spec, newShared(newPoolRegistry(nil, nil)), globalRand{}, clock)
	c.retain()
	c.start()
	t.Cleanup(c.release)
	return c
}

func endpointSpecs(names ...string) []EndpointSpec {
	specs := make([]EndpointSpec, len(names))
	for i, name := range names {
		specs[i] = EndpointSpec{Name: name, Scheme: "http", Host: "10.0.0.1", Port: 8000 + i, Weight: 1}
	}
	return specs
}

func inRotation(c *cluster) string {
	var names string
	for _, ep := range c.hosts.Load().hosts {
		names += ep.spec.Name
	}
	return names
}

func observeN(c *cluster, name string, code, n int) {
	for _, ep := range c.endpoints {
		if ep.spec.Name == name {
			for range n {
				c.observe(ep, code)
			}
		}
	}
}

func TestOutlierEjectsAfterConsecutiveFailuresUpToTheCap(t *testing.T) {
	spec := ClusterSpec{
		Name: "outlier-consecutive", Endpoints: endpointSpecs("a", "b", "c", "d"), LBPolicy: LBRoundRobin,
		Outlier: &OutlierSpec{Consecutive5xx: 3, MaxEjectionPercent: 50},
	}
	c := detachedCluster(t, spec, newManualClock())

	observeN(c, "a", 500, 2)
	observeN(c, "a", 200, 1)
	observeN(c, "a", 503, 2)
	if inRotation(c) != "abcd" {
		t.Fatalf("a success between failures must reset the count; rotation = %s", inRotation(c))
	}
	observeN(c, "a", 503, 1)
	if inRotation(c) != "bcd" {
		t.Fatalf("three failures in a row must eject a; rotation = %s", inRotation(c))
	}
	// Local failures count as Envoy records them: 503 for a broken
	// connection, 504 for a timeout.
	connect, _ := outlierCode(&Error{Kind: KindConnectFailure})
	timeout, _ := outlierCode(&Error{Kind: KindTimeout, Stage: StagePerTry})
	observeN(c, "b", connect, 2)
	observeN(c, "b", timeout, 1)
	if inRotation(c) != "cd" {
		t.Fatalf("local failures must eject b; rotation = %s", inRotation(c))
	}
	observeN(c, "c", 500, 3)
	if inRotation(c) != "cd" {
		t.Fatalf("a third ejection would exceed 50%%; rotation = %s", inRotation(c))
	}
	got := testutil.ToFloat64(metrics.UpstreamEjectionsTotal.WithLabelValues("outlier-consecutive", "a", "consecutive_5xx"))
	if got != 1 {
		t.Fatalf("ejection metric = %v, want 1", got)
	}
}

func TestOutlierFailuresThatSayNothingAboutTheEndpointAreNotCharged(t *testing.T) {
	for _, failure := range []*Error{
		nil,
		{Kind: KindTimeout, Stage: StageIdle},
		{Kind: KindCanceled},
		{Kind: KindOverflow},
		{Kind: KindClosed},
	} {
		if _, charged := outlierCode(failure); charged {
			t.Fatalf("%v was charged to the endpoint", failure)
		}
	}
}

func TestOutlierEjectionTimeBacksOffAndRecovers(t *testing.T) {
	clock := newManualClock()
	spec := ClusterSpec{
		Name: "outlier-backoff", Endpoints: endpointSpecs("a", "b"), LBPolicy: LBRoundRobin,
		Outlier: &OutlierSpec{Consecutive5xx: 3, BaseEjectionTime: 30 * time.Second},
	}
	c := detachedCluster(t, spec, clock)
	a := c.endpoints[0]

	observeN(c, "a", 500, 3)
	clock.Advance(20 * time.Second)
	if inRotation(c) != "b" {
		t.Fatalf("a returned before its base ejection time; rotation = %s", inRotation(c))
	}
	clock.Advance(10 * time.Second)
	if inRotation(c) != "ab" {
		t.Fatalf("a did not return at the sweep after 30s; rotation = %s", inRotation(c))
	}
	// Ejected again at once: twice the base time, so the sweep at +60s.
	observeN(c, "a", 500, 3)
	clock.Advance(50 * time.Second)
	if inRotation(c) != "b" {
		t.Fatalf("a returned before 60s; rotation = %s", inRotation(c))
	}
	clock.Advance(10 * time.Second)
	if inRotation(c) != "ab" || a.outlier.backoff != 2 {
		t.Fatalf("rotation = %s backoff = %d, want ab and 2", inRotation(c), a.outlier.backoff)
	}
	// Each interval in rotation takes one step off the back-off.
	clock.Advance(20 * time.Second)
	if a.outlier.backoff != 0 {
		t.Fatalf("backoff = %d after two healthy intervals, want 0", a.outlier.backoff)
	}
}

func TestOutlierSuccessRateEjectsTheStatisticalOutlier(t *testing.T) {
	clock := newManualClock()
	spec := ClusterSpec{
		Name: "outlier-success-rate", Endpoints: endpointSpecs("a", "b", "c", "d", "e", "f"), LBPolicy: LBRoundRobin,
		Outlier: &OutlierSpec{Consecutive5xx: 1000},
	}
	c := detachedCluster(t, spec, clock)
	// Envoy's own example: rates {50, 100, 100, 100, 100} give mean 90, stdev
	// 20 and a threshold of 52. f saw too few requests to be judged.
	for _, name := range []string{"b", "c", "d", "e"} {
		observeN(c, name, 200, 100)
	}
	observeN(c, "a", 200, 50)
	observeN(c, "a", 500, 50)
	observeN(c, "f", 500, 99)
	clock.Advance(10 * time.Second)
	if inRotation(c) != "bcdef" {
		t.Fatalf("rotation = %s, want a ejected for its success rate", inRotation(c))
	}
}

func TestPanicThresholdBalancesOverEveryEndpoint(t *testing.T) {
	c := detachedCluster(t, ClusterSpec{Name: "panic", Endpoints: endpointSpecs("a", "b", "c", "d"), LBPolicy: LBRoundRobin},
		newManualClock())
	setFailed := func(names ...string) {
		c.mu.Lock()
		defer c.mu.Unlock()
		for _, ep := range c.endpoints {
			ep.health.failed = false
			for _, name := range names {
				ep.health.failed = ep.health.failed || ep.spec.Name == name
			}
		}
		c.refreshHostsLocked()
	}
	setFailed("a", "b")
	if inRotation(c) != "cd" {
		t.Fatalf("half healthy is not below the threshold; rotation = %s", inRotation(c))
	}
	setFailed("a", "b", "c")
	if inRotation(c) != "abcd" {
		t.Fatalf("below half healthy the cluster must balance over all; rotation = %s", inRotation(c))
	}
	if testutil.ToFloat64(metrics.UpstreamClusterPanic.WithLabelValues("panic")) != 1 {
		t.Fatal("panic gauge not set")
	}
}

func TestDoEjectsAFailingEndpointAndRoutesAround(t *testing.T) {
	failing := endpointOf(t, "failing", backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	healthy := endpointOf(t, "healthy", backend(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	spec := clusterOf("eject-e2e", failing, healthy)
	spec.Outlier = &OutlierSpec{Consecutive5xx: 2}
	clock := newManualClock()
	set := newSet(t, Options{Clock: clock, Rand: rand.New(rand.NewPCG(5, 6))}, spec)

	served := map[string]int{}
	for range 8 {
		resp, err := set.Do(t.Context(), post("eject-e2e"))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = readAll(t, resp)
		served[resp.Endpoint.Name]++
	}
	// Round robin alternates until the second 503 ejects the failing one.
	if served["failing"] != 2 || served["healthy"] != 6 {
		t.Fatalf("served = %v, want the failing endpoint ejected after two 503s", served)
	}
	clock.Advance(30 * time.Second)
	if inRotation(set.clusters["eject-e2e"]) != "failinghealthy" {
		t.Fatal("the ejected endpoint did not return after the base ejection time")
	}
}
