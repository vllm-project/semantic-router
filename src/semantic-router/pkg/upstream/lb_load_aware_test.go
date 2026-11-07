package upstream

import (
	"io"
	"math/rand/v2"
	"net/http"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

// fakeTelemetry deliberately uses a non-comparable reader, as the public
// interface permits. An identity it holds is fresh.
type fakeTelemetry map[backendtelemetry.Identity]backendtelemetry.Observation

func (f fakeTelemetry) Fresh(id backendtelemetry.Identity) (backendtelemetry.Observation, bool) {
	o, ok := f[id]
	return o, ok
}

func telemetryID(name string) backendtelemetry.Identity {
	return backendtelemetry.Identity{Model: "m", BackendRef: name, Engine: backendtelemetry.EngineVLLM}
}

func loadAwareEndpoints(n int) []*endpoint {
	endpoints := testEndpoints(make([]int, n)...)
	for _, ep := range endpoints {
		ep.spec.Weight = 1
		ep.spec.Telemetry = telemetryID(ep.spec.Name)
	}
	return endpoints
}

func observation(name string, queued int64) backendtelemetry.Observation {
	return backendtelemetry.Observation{
		Schema: backendtelemetry.SchemaVersion, Identity: telemetryID(name),
		Source: "test", CollectedAt: time.Unix(1, 0), QueueDepth: backendtelemetry.Observed(queued),
	}
}

func newLoadAware() *loadAware {
	return newBalancer("m", LBLoadAware, rand.New(rand.NewPCG(1, 2))).(*loadAware)
}

func TestLoadAwarePicksTheLeastQueuedEndpoint(t *testing.T) {
	telemetry := fakeTelemetry{
		telemetryID("a"): observation("a", 9),
		telemetryID("b"): observation("b", 1),
		telemetryID("c"): observation("c", 4),
	}
	if counts := pickCounts(newLoadAware(), newHostSet(loadAwareEndpoints(3)), 50, telemetry); counts["b"] != 50 {
		t.Fatalf("counts = %v, want every pick on the least-queued b", counts)
	}
}

func TestLoadAwareCountsRequestsInFlight(t *testing.T) {
	endpoints := loadAwareEndpoints(2)
	endpoints[0].active.Store(5) // a: 0 queued + 5 in flight; b: 3 queued
	telemetry := fakeTelemetry{telemetryID("a"): observation("a", 0), telemetryID("b"): observation("b", 3)}
	if got := newLoadAware().pick(newHostSet(endpoints), telemetry); got.spec.Name != "b" {
		t.Fatalf("picked %s, want b: a's in-flight requests make it busier", got.spec.Name)
	}
}

func TestLoadAwareBreaksTiesByWeightThenEvenly(t *testing.T) {
	endpoints := loadAwareEndpoints(3)
	endpoints[2].spec.Weight = 5
	tied := fakeTelemetry{
		telemetryID("a"): observation("a", 2),
		telemetryID("b"): observation("b", 2),
		telemetryID("c"): observation("c", 2),
	}
	if got := newLoadAware().pick(newHostSet(endpoints), tied); got.spec.Name != "c" {
		t.Fatalf("picked %s, want the heavier c among equal loads", got.spec.Name)
	}
	counts := pickCounts(newLoadAware(), newHostSet(loadAwareEndpoints(3)), 3000, tied)
	for _, name := range []string{"a", "b", "c"} {
		if counts[name] < 850 || counts[name] > 1150 {
			t.Fatalf("counts = %v, want equal loads spread about evenly", counts)
		}
	}
}

func TestLoadAwareSkipsEndpointsReportedUnhealthy(t *testing.T) {
	sick := observation("a", 0)
	sick.Health = backendtelemetry.HealthUnhealthy
	telemetry := fakeTelemetry{telemetryID("a"): sick, telemetryID("b"): observation("b", 7)}
	if counts := pickCounts(newLoadAware(), newHostSet(loadAwareEndpoints(2)), 20, telemetry); counts["b"] != 20 {
		t.Fatalf("counts = %v, want the unhealthy a skipped", counts)
	}
}

func TestLoadAwareFallsBackWithoutCompleteTelemetry(t *testing.T) {
	missingQueue := observation("b", 0)
	missingQueue.QueueDepth = backendtelemetry.Count{}
	a, b := observation("a", 0), observation("b", 0)
	a.Health, b.Health = backendtelemetry.HealthUnhealthy, backendtelemetry.HealthUnhealthy
	cases := []struct {
		name      string
		telemetry TelemetryReader
		noEngine  bool
		want      string
	}{
		{"no reader", nil, false, loadAwareNoTelemetry},
		{"stale endpoint", fakeTelemetry{telemetryID("a"): observation("a", 0)}, false, loadAwareStale},
		{"missing queue depth", fakeTelemetry{telemetryID("a"): observation("a", 0), telemetryID("b"): missingQueue}, false, loadAwareMissingMetric},
		{"endpoint without engine", fakeTelemetry{telemetryID("a"): observation("a", 0), telemetryID("b"): observation("b", 0)}, true, loadAwareNoEngine},
		{"every endpoint unhealthy", fakeTelemetry{telemetryID("a"): a, telemetryID("b"): b}, false, loadAwareAllUnhealthy},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			endpoints := loadAwareEndpoints(2)
			if tc.noEngine {
				endpoints[1].spec.Telemetry = backendtelemetry.Identity{}
			}
			set := newHostSet(endpoints)
			balancer := newLoadAware()
			if ep, outcome := balancer.pickByTelemetry(set, tc.telemetry); ep != nil || outcome != tc.want {
				t.Fatalf("pickByTelemetry() = %v, %q; want a fallback for %q", ep, outcome, tc.want)
			}
			endpoints[0].active.Store(10)
			if counts := pickCounts(balancer, set, 40, tc.telemetry); counts["b"] < 30 {
				t.Fatalf("fallback counts = %v, want least-request to favor the idle b", counts)
			}
		})
	}
}

func TestReusedClusterReadsEachSetsTelemetry(t *testing.T) {
	a := backend(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "a") })
	b := backend(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "b") })
	spec := clusterOf("m", endpointOf(t, "a", a), endpointOf(t, "b", b))
	spec.LBPolicy = LBLoadAware
	for i := range spec.Endpoints {
		spec.Endpoints[i].Telemetry = telemetryID(spec.Endpoints[i].Name)
	}
	first := newSet(t, Options{Telemetry: fakeTelemetry{
		telemetryID("a"): observation("a", 0), telemetryID("b"): observation("b", 9),
	}}, spec)
	second := newSet(t, Options{Previous: first, Telemetry: fakeTelemetry{
		telemetryID("a"): observation("a", 9), telemetryID("b"): observation("b", 0),
	}}, spec)
	if first.clusters["m"] != second.clusters["m"] {
		t.Fatal("an unchanged cluster was not reused")
	}
	for _, tc := range []struct {
		set  *Set
		want string
	}{{first, "a"}, {second, "b"}, {first, "a"}} {
		resp, err := tc.set.Do(t.Context(), post("m"))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = readAll(t, resp)
		if resp.Endpoint.Name != tc.want {
			t.Fatalf("picked %s, want %s from this Set's reader", resp.Endpoint.Name, tc.want)
		}
	}
	disabled := newSet(t, Options{Previous: second}, spec)
	if disabled.clusters["m"] != second.clusters["m"] {
		t.Fatal("disabling telemetry discarded reusable cluster state")
	}
	counts := map[string]int{}
	for range 60 {
		resp, err := disabled.Do(t.Context(), post("m"))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = readAll(t, resp)
		counts[resp.Endpoint.Name]++
	}
	if counts["a"] == 0 || counts["b"] == 0 {
		t.Fatalf("nil reader retained old telemetry: %v", counts)
	}
}

func TestCompileMapsLoadAwarePolicyAndEndpointTelemetry(t *testing.T) {
	topology := compileYAML(t, `version: v0.3
providers:
  defaults:
    model: m
  models:
    - name: m
      reliability:
        lb_policy: load_aware
      backend_refs:
        - name: gpu-a
          provider: vllm
          endpoint: 10.0.0.1:8000
          engine: vllm
        - name: gpu-b
          provider: vllm
          endpoint: 10.0.0.2:8000
routing: {}
`)
	cluster := topology.Clusters[0]
	if cluster.LBPolicy != LBLoadAware {
		t.Fatalf("lb policy = %q, want %q", cluster.LBPolicy, LBLoadAware)
	}
	want := backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM}
	if got := cluster.Endpoints[0].Telemetry; got != want {
		t.Fatalf("gpu-a telemetry = %+v, want %+v", got, want)
	}
	if got := cluster.Endpoints[1].Telemetry; got != (backendtelemetry.Identity{}) {
		t.Fatalf("gpu-b names no engine but has telemetry identity %+v", got)
	}
}
