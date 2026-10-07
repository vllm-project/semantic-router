package upstream

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// scriptedFallback answers each Next from its steps, then ends the chain, and
// records the outcomes it was shown.
type scriptedFallback struct {
	steps    []routing.FallbackStep
	err      error
	onNext   func()
	outcomes []routing.Outcome
}

func (f *scriptedFallback) Next(_ context.Context, outcome routing.Outcome) (routing.FallbackStep, error) {
	f.outcomes = append(f.outcomes, outcome)
	if f.onNext != nil {
		f.onNext()
	}
	if f.err != nil || len(f.steps) == 0 {
		return routing.FallbackStep{}, f.err
	}
	step := f.steps[0]
	f.steps = f.steps[1:]
	return step, nil
}

func plannedCall(route string, fallback routing.Fallback) *routing.Call {
	return &routing.Call{
		Route: route,
		Request: routing.Request{
			Header: routing.Header{
				{Name: ":method", Value: http.MethodPost},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: "content-type", Value: "application/json"},
			},
			Body: []byte(`{"model":"m"}`),
		},
		Fallback: fallback,
	}
}

func answering(status int, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// fallbackSet serves a failing primary cluster and a candidate cluster.
func fallbackSet(t *testing.T, primary, candidate http.HandlerFunc) *Set {
	t.Helper()
	return newSet(t, Options{},
		clusterOf("primary", endpointOf(t, "p", backend(t, primary))),
		clusterOf("candidate", endpointOf(t, "c", backend(t, candidate))))
}

func execute(t *testing.T, set *Set, call *routing.Call) *Result {
	t.Helper()
	result, err := set.Execute(t.Context(), call, "")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	return result
}

func TestExecuteDoesNotConsultTheFallbackAfterASuccess(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusOK, "primary ok"), answering(http.StatusOK, "unused"))
	fallback := &scriptedFallback{}
	result := execute(t, set, plannedCall("primary", fallback))
	if body, _ := readAll(t, result.Response); body != "primary ok" || result.Hops != 1 {
		t.Fatalf("body = %q after %d hops, want the primary's answer", body, result.Hops)
	}
	if len(fallback.outcomes) != 0 {
		t.Fatalf("fallback saw %d outcomes, want none", len(fallback.outcomes))
	}
}

func TestExecuteSendsTheCandidateTheFallbackPrepares(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, "primary down"), answering(http.StatusOK, "candidate ok"))
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Call: plannedCall("candidate", nil)}}}
	result := execute(t, set, plannedCall("primary", fallback))

	if body, _ := readAll(t, result.Response); body != "candidate ok" || result.Response.Cluster != "candidate" {
		t.Fatalf("body = %q from %s, want the candidate's answer", body, result.Response.Cluster)
	}
	if result.Hops != 2 || len(fallback.outcomes) != 1 {
		t.Fatalf("hops = %d, outcomes = %d; want 2 and 1", result.Hops, len(fallback.outcomes))
	}
	outcome := fallback.outcomes[0]
	if outcome.Route != "primary" || outcome.Status != http.StatusServiceUnavailable ||
		string(outcome.Body) != "primary down" || outcome.Local || outcome.Header.Get("content-length") == "" {
		t.Fatalf("outcome = %+v, want the primary's 503 with its body and headers", outcome)
	}
}

func TestExecuteReturnsThePrimaryFailureWholeWhenTheChainEnds(t *testing.T) {
	large := strings.Repeat("x", fallbackBodyLimit+4096)
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, large), answering(http.StatusBadGateway, "candidate down"))
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Call: plannedCall("candidate", nil)}}}
	result := execute(t, set, plannedCall("primary", fallback))

	body, err := readAll(t, result.Response)
	if err != nil || body != large || result.Response.Cluster != "primary" ||
		result.Response.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("got %d from %s with %d body bytes (%v), want the primary's whole failure",
			result.Response.StatusCode, result.Response.Cluster, len(body), err)
	}
	if len(fallback.outcomes) != 2 || len(fallback.outcomes[0].Body) != fallbackBodyLimit ||
		fallback.outcomes[1].Route != "candidate" || fallback.outcomes[1].Status != http.StatusBadGateway {
		t.Fatalf("outcomes = %d, want the bounded primary and the candidate", len(fallback.outcomes))
	}
}

// The response-phase fallback accepts only a 2xx from a candidate, so a
// candidate's redirect is a failed attempt, not an answer.
func TestExecuteTreatsACandidatesRedirectAsAFailure(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, "primary down"), answering(http.StatusFound, "moved"))
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Call: plannedCall("candidate", nil)}}}
	result := execute(t, set, plannedCall("primary", fallback))

	if len(fallback.outcomes) != 2 || fallback.outcomes[1].Status != http.StatusFound {
		t.Fatalf("outcomes = %+v, want the candidate's 302 reported as a failure", fallback.outcomes)
	}
	if body, _ := readAll(t, result.Response); body != "primary down" {
		t.Fatalf("body = %q, want the primary's failure once the chain ends", body)
	}
}

func TestExecuteReturnsThePrimaryFailureWhenTheFallbackFails(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, "primary down"), answering(http.StatusOK, "unused"))
	fallback := &scriptedFallback{err: errors.New("session ended")}
	result := execute(t, set, plannedCall("primary", fallback))
	if body, _ := readAll(t, result.Response); body != "primary down" || result.Hops != 1 {
		t.Fatalf("body = %q after %d hops, want the primary's failure", body, result.Hops)
	}
}

func TestExecuteAnswersWithTheFallbacksImmediateResponse(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, "primary down"), answering(http.StatusOK, "unused"))
	immediate := &routing.Response{Status: http.StatusTooManyRequests, Body: []byte("slow down")}
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Immediate: immediate}}}
	result := execute(t, set, plannedCall("primary", fallback))
	if result.Immediate != immediate || result.Response != nil {
		t.Fatalf("result = %+v, want the immediate response alone", result)
	}
}

func TestExecuteDescribesALocalReplyToTheFallback(t *testing.T) {
	refused := EndpointSpec{Name: "refused", Scheme: "http", Host: "127.0.0.1", Port: closedPort(t), Weight: 1}
	set := newSet(t, Options{}, clusterOf("primary", refused))
	fallback := &scriptedFallback{}
	result := execute(t, set, plannedCall("primary", fallback))

	outcome := fallback.outcomes[0]
	if !outcome.Local || outcome.Failure != string(KindConnectFailure) || outcome.Status != http.StatusServiceUnavailable {
		t.Fatalf("outcome = %+v, want a local connect-failure reply", outcome)
	}
	if body, _ := readAll(t, result.Response); result.Response.Local == nil ||
		!strings.HasPrefix(body, "upstream connect error") {
		t.Fatalf("response %q, want Envoy's local reply", body)
	}
}

func TestExecuteNeverFallsBackOnceTheResponseIsReturned(t *testing.T) {
	set := fallbackSet(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		_, _ = io.WriteString(w, "partial")
		w.(http.Flusher).Flush()
		hijackAndClose(w)
	}, answering(http.StatusOK, "unused"))
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Call: plannedCall("candidate", nil)}}}
	result := execute(t, set, plannedCall("primary", fallback))

	body, err := readAll(t, result.Response)
	if KindOf(err) != KindReset || body != "partial" {
		t.Fatalf("body = %q, err = %v; want the partial body then a reset", body, err)
	}
	if result.Hops != 1 || len(fallback.outcomes) != 0 {
		t.Fatalf("hops = %d, outcomes = %d; a drop after the headers must not fall back",
			result.Hops, len(fallback.outcomes))
	}
}

func TestExecuteStopsWhenTheClientGoesAway(t *testing.T) {
	set := fallbackSet(t, answering(http.StatusServiceUnavailable, "primary down"), answering(http.StatusOK, "unused"))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	fallback := &scriptedFallback{steps: []routing.FallbackStep{{Call: plannedCall("candidate", nil)}}, onNext: cancel}
	if _, err := set.Execute(ctx, plannedCall("primary", fallback), ""); KindOf(err) != KindCanceled {
		t.Fatalf("error = %v, want %s", err, KindCanceled)
	}
}
