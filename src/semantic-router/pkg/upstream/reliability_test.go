package upstream

import (
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func durationOf(d time.Duration) *time.Duration { return &d }

func countOf(n int) *int { return &n }

func TestMergeKeepsThePolicyWithoutAnOverride(t *testing.T) {
	base := Policy{Timeouts: Timeouts{Total: time.Minute}, Retry: &RetryPolicy{NumRetries: 2, On: Retry5xx}}
	if got := base.merge(nil); !reflect.DeepEqual(got, base) {
		t.Fatalf("merge(nil) = %+v, want %+v", got, base)
	}
	if got := base.merge(&routing.Reliability{}); !reflect.DeepEqual(got, base) {
		t.Fatalf("merge(empty) = %+v, want %+v", got, base)
	}
}

func TestMergeReplacesTimeoutsAndZeroDisablesThem(t *testing.T) {
	base := Policy{Timeouts: Timeouts{Connect: time.Second, Total: time.Minute, PerTry: 5 * time.Second, Idle: time.Minute}}
	got := base.merge(&routing.Reliability{
		TotalTimeout:     durationOf(10 * time.Second),
		PerTryTimeout:    durationOf(0),
		FirstByteTimeout: durationOf(2 * time.Second),
	})
	want := Timeouts{Connect: time.Second, Total: 10 * time.Second, PerTry: NoTimeout, Idle: time.Minute, FirstByte: 2 * time.Second}
	if got.Timeouts != want {
		t.Fatalf("timeouts = %+v, want %+v", got.Timeouts, want)
	}
	if got.Retry != nil {
		t.Fatalf("a timeout override added a retry policy: %+v", got.Retry)
	}
}

// Envoy's router ORs x-envoy-retry-on into the route's retry_on, appends
// x-envoy-retriable-status-codes and lets x-envoy-max-retries replace the
// count; the native merge must agree.
func TestMergeAddsRetryConditionsAsEnvoyHeadersDo(t *testing.T) {
	codes := []int{503}
	base := Policy{Retry: &RetryPolicy{
		NumRetries: 2, On: RetryConnectFailure | RetryRetriableStatusCodes, RetriableStatusCodes: codes,
		BackOffBase: 25 * time.Millisecond,
	}}
	got := base.merge(&routing.Reliability{
		RetryCount:           countOf(4),
		RetryOn:              []string{"reset", "5xx"},
		RetriableStatusCodes: []int{429},
		RetryBackOffBase:     durationOf(100 * time.Millisecond),
	})
	want := &RetryPolicy{
		NumRetries:           4,
		On:                   RetryConnectFailure | RetryRetriableStatusCodes | RetryReset | Retry5xx,
		RetriableStatusCodes: []int{503, 429},
		BackOffBase:          100 * time.Millisecond,
	}
	if !reflect.DeepEqual(got.Retry, want) {
		t.Fatalf("retry = %+v, want %+v", got.Retry, want)
	}
	if len(codes) != 1 || base.Retry.NumRetries != 2 {
		t.Fatal("merge changed the cluster's own policy")
	}
}

func TestMergeStartsRetriesLikeEnvoyWithoutARoutePolicy(t *testing.T) {
	cases := map[string]struct {
		override routing.Reliability
		want     RetryPolicy
	}{
		"count alone retries on the default conditions": {
			override: routing.Reliability{RetryCount: countOf(2)},
			want:     RetryPolicy{NumRetries: 2, On: RetryConnectFailure | RetryRefusedStream},
		},
		"conditions alone retry once": {
			override: routing.Reliability{RetryOn: []string{"gateway-error"}},
			want:     RetryPolicy{NumRetries: 1, On: RetryGatewayError},
		},
		"a zero count disables retries": {
			override: routing.Reliability{RetryCount: countOf(0)},
			want:     RetryPolicy{NumRetries: 0, On: RetryConnectFailure | RetryRefusedStream},
		},
	}
	for name, tc := range cases {
		got := Policy{}.merge(&tc.override)
		if got.Retry == nil || !reflect.DeepEqual(*got.Retry, tc.want) {
			t.Fatalf("%s: retry = %+v, want %+v", name, got.Retry, tc.want)
		}
	}
}

// With a per-try timeout and no retries, the template still renders a route
// retry policy, so a decision's retry count merges into its conditions.
// A request-graph step's override is one more layer over its decision's:
// the call's layers merge in turn, so the decision may start retries on the
// default conditions and the step then add its own.
func TestMergeAppliesAStepsOverrideAsAnotherLayer(t *testing.T) {
	decision := &routing.Reliability{RetryCount: countOf(1), TotalTimeout: durationOf(time.Minute)}
	step := &routing.Reliability{RetryOn: []string{"reset"}, TotalTimeout: durationOf(5 * time.Second)}
	got := Policy{}.merge(decision, step)
	want := Policy{
		Timeouts: Timeouts{Total: 5 * time.Second},
		Retry:    &RetryPolicy{NumRetries: 1, On: ParseRetryOn(config.DefaultProviderRetryOn) | RetryReset},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merge(decision, step) = %+v, want %+v", got, want)
	}
	if got := (Policy{}).merge(nil, step); !reflect.DeepEqual(got, (Policy{}).merge(step)) {
		t.Fatalf("a step over no decision is the step alone: %+v", got)
	}
}

func TestCompileKeepsTheTemplatesZeroRetryPolicy(t *testing.T) {
	topology := compileYAML(t, `
version: v0.3
providers:
  models:
    - name: bounded
      provider_model_id: bounded-model
      api_format: openai
      reliability:
        per_try_timeout: 5s
        retry_on: reset
      backend_refs:
        - {name: a, provider: vllm, endpoint: 10.0.0.1:8000}
routing: {}
`)
	got := topology.cluster("bounded").Policy.Retry
	if got == nil || got.NumRetries != 0 || got.On != RetryReset {
		t.Fatalf("retry = %+v, want the template's zero-retry policy on reset", got)
	}
	merged := topology.cluster("bounded").Policy.merge(&routing.Reliability{RetryCount: countOf(2)})
	if merged.Retry.NumRetries != 2 || merged.Retry.On != RetryReset {
		t.Fatalf("merged retry = %+v, want two retries on the route's reset", merged.Retry)
	}
}

func TestCallPolicyLayersTheDecisionBetweenClusterAndCaller(t *testing.T) {
	spec := clusterOf("layered", EndpointSpec{Name: "e", Scheme: "http", Host: "127.0.0.1", Port: 1, Weight: 1})
	spec.Policy = Policy{Timeouts: Timeouts{Total: time.Minute, PerTry: 20 * time.Second}}
	set := newSet(t, Options{}, spec)
	c, _ := set.route("layered")
	req := post("layered")
	req.Reliability = []*routing.Reliability{{TotalTimeout: durationOf(30 * time.Second), PerTryTimeout: durationOf(10 * time.Second)}}
	req.Policy = &Policy{Timeouts: Timeouts{PerTry: 5 * time.Second}}

	got := set.policy(req, c).Timeouts
	if got.Total != 30*time.Second || got.PerTry != 5*time.Second {
		t.Fatalf("timeouts = %+v, want the decision's total and the caller's per-try", got)
	}
}

func TestDoTimesOutAtTheDecisionsTotalTimeout(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) { awaitDisconnect(r) })
	set := newSet(t, Options{}, clusterOf("slow", endpointOf(t, "e", server)))
	req := post("slow")
	req.Reliability = []*routing.Reliability{{TotalTimeout: durationOf(100 * time.Millisecond)}}

	started := time.Now()
	resp, err := set.Do(t.Context(), req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	elapsed := time.Since(started)
	assertLocalReply(t, resp, http.StatusGatewayTimeout, "upstream request timeout")
	if elapsed < 100*time.Millisecond || elapsed > 2*time.Second {
		t.Fatalf("timed out after %v, want the decision's 100ms", elapsed)
	}
}

func TestDoRetriesUnderTheDecisionsRetryOverride(t *testing.T) {
	var calls atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	set := newSet(t, Options{}, clusterOf("flaky", endpointOf(t, "e", server)))
	req := post("flaky")
	req.Reliability = []*routing.Reliability{{RetryCount: countOf(1), RetryOn: []string{"5xx"}, RetryBackOffBase: durationOf(time.Millisecond)}}

	resp, err := set.Do(t.Context(), req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_, _ = readAll(t, resp)
	if resp.StatusCode != http.StatusOK || len(resp.Attempts) != 2 {
		t.Fatalf("status %d after %d attempts, want 200 after a retry", resp.StatusCode, len(resp.Attempts))
	}
}
