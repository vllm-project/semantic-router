package upstream

import (
	"io"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// waitRecorder is a Clock whose timers fire at once, recording each wait, so
// retry tests check back-off bounds without sleeping.
type waitRecorder struct {
	mu    sync.Mutex
	waits []time.Duration
}

func (r *waitRecorder) Now() time.Time { return time.Now() }

func (r *waitRecorder) AfterFunc(d time.Duration, f func()) Timer {
	r.mu.Lock()
	r.waits = append(r.waits, d)
	r.mu.Unlock()
	go f()
	return firedTimer{}
}

func (r *waitRecorder) recorded() []time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]time.Duration(nil), r.waits...)
}

type firedTimer struct{}

func (firedTimer) Stop() bool { return false }

func TestParseRetryOnIgnoresTokensItCannotAct(t *testing.T) {
	got := ParseRetryOn(" 5xx, connect-failure ,retriable-status-codes,unavailable,typo")
	if got != Retry5xx|RetryConnectFailure|RetryRetriableStatusCodes {
		t.Fatalf("ParseRetryOn = %b", got)
	}
}

func TestRetryDecisionsFollowEnvoy(t *testing.T) {
	response := func(code int, header ...string) *Response {
		h := http.Header{}
		for i := 0; i+1 < len(header); i += 2 {
			h.Set(header[i], header[i+1])
		}
		return &Response{StatusCode: code, Header: h}
	}
	tests := []struct {
		name    string
		on      string
		resp    *Response
		failure *Error
		sent    bool
		want    retryDecision
	}{
		{"5xx retries a 500", "5xx", response(500), nil, true, retryWithBackOff},
		{"gateway-error skips a 500", "gateway-error", response(500), nil, true, noRetry},
		{"gateway-error retries a 503", "gateway-error", response(503), nil, true, retryWithBackOff},
		{"listed status", "retriable-status-codes", response(429), nil, true, retryWithBackOff},
		{"unlisted status", "retriable-status-codes", response(418), nil, true, noRetry},
		{"409 under retriable-4xx", "retriable-4xx", response(409), nil, true, retryWithBackOff},
		{"rate limited without the token", "5xx", response(503, "X-Envoy-Ratelimited", "true"), nil, true, noRetry},
		{"connect failure", "connect-failure", nil, &Error{Kind: KindConnectFailure}, false, retryWithBackOff},
		{"connect failure under reset", "reset", nil, &Error{Kind: KindConnectFailure}, false, retryWithBackOff},
		{"connect-failure skips a reset", "connect-failure", nil, &Error{Kind: KindReset}, true, noRetry},
		{"reset before the request", "reset-before-request", nil, &Error{Kind: KindReset}, false, retryWithBackOff},
		{"reset after the request", "reset-before-request", nil, &Error{Kind: KindReset}, true, noRetry},
		{"refused stream at once", "refused-stream", nil, &Error{Kind: KindRefusedStream}, true, retryNow},
		{"per-try timeout under 5xx", "5xx", nil, &Error{Kind: KindTimeout, Stage: StagePerTry}, true, retryWithBackOff},
		{"first-byte timeout under reset", "reset", nil, &Error{Kind: KindTimeout, Stage: StageFirstByte}, true, retryWithBackOff},
		{"total timeout ends the call", "5xx", nil, &Error{Kind: KindTimeout, Stage: StageTotal}, true, noRetry},
		{"overflow is never retried", "5xx,reset", nil, &Error{Kind: KindOverflow}, false, noRetry},
		{"cancellation is never retried", "5xx,reset", nil, &Error{Kind: KindCanceled}, true, noRetry},
	}
	for _, tt := range tests {
		policy := &RetryPolicy{On: ParseRetryOn(tt.on), RetriableStatusCodes: []int{429}}
		if got := policy.decide(tt.resp, tt.failure, tt.sent); got != tt.want {
			t.Errorf("%s: decision = %d, want %d", tt.name, got, tt.want)
		}
	}
}

func TestBackOffIsJitteredExponentialWithEnvoysDefaults(t *testing.T) {
	waits := newBackOff(&RetryPolicy{}, rand.New(rand.NewPCG(7, 8)))
	bounds := []time.Duration{25, 50, 100, 200, 250, 250}
	for i, bound := range bounds {
		bound *= time.Millisecond
		if w := waits.wait(); w < 0 || w >= bound {
			t.Fatalf("wait %d = %v, want below %v", i, w, bound)
		}
	}
	// Intervals 100ms, 200ms, then the 300ms cap: each wait is half the draw
	// modulo the interval.
	custom := newBackOff(&RetryPolicy{BackOffBase: 100 * time.Millisecond, BackOffMax: 300 * time.Millisecond},
		&scriptedRand{values: []uint64{198, 398, 598}})
	for i, want := range []time.Duration{99, 199, 299} {
		if w := custom.wait(); w != want*time.Millisecond {
			t.Fatalf("custom wait %d = %v, want %v", i, w, want*time.Millisecond)
		}
	}
}

func TestRetryAfterIsHonoredWithinItsBound(t *testing.T) {
	policy := &RetryPolicy{RetryAfterMax: 30 * time.Second}
	rnd := rand.New(rand.NewPCG(1, 1))
	withHeader := func(v string) *Response { return &Response{Header: http.Header{"Retry-After": {v}}} }
	if wait, ok := policy.retryAfter(withHeader("2"), rnd); !ok || wait < 2*time.Second || wait >= 3*time.Second {
		t.Fatalf("Retry-After 2 waits %v (%v), want 2s to 3s", wait, ok)
	}
	for _, value := range []string{"60", "soon", "-1", ""} {
		if _, ok := policy.retryAfter(withHeader(value), rnd); ok {
			t.Fatalf("Retry-After %q was honored", value)
		}
	}
	if _, ok := (&RetryPolicy{}).retryAfter(withHeader("1"), rnd); ok {
		t.Fatal("Retry-After was honored without retry_after_max")
	}
}

func TestBreakerRetryBudgetScalesWithLoad(t *testing.T) {
	b := newBreaker("budget", Breakers{RetryBudget: &RetryBudget{Percent: 50, MinConcurrency: 1}})
	b.active = 6
	var releases []func()
	for range 3 {
		release, err := b.acquireRetry()
		if err != nil {
			t.Fatalf("retry within 50%% of 6 active: %v", err)
		}
		releases = append(releases, release)
	}
	if _, err := b.acquireRetry(); KindOf(err) != KindOverflow {
		t.Fatalf("a fourth retry = %v, want overflow", err)
	}
	for _, release := range releases {
		release()
	}
}

func retrying(name, on string, retries int, endpoints ...EndpointSpec) ClusterSpec {
	spec := clusterOf(name, endpoints...)
	spec.Policy.Retry = &RetryPolicy{NumRetries: retries, On: ParseRetryOn(on), RetriableStatusCodes: []int{429}}
	return spec
}

func TestDoRetriesUntilASuccess(t *testing.T) {
	var hits atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, "ok")
	})
	clock := &waitRecorder{}
	// Two draws seed the balancer; the next two pick the back-off waits.
	rnd := &scriptedRand{values: []uint64{0, 0, 14, 66}}
	set := newSet(t, Options{Clock: clock, Rand: rnd},
		retrying("retry-success", "gateway-error", 3, endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("retry-success"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); body != "ok" || len(resp.Attempts) != 3 {
		t.Fatalf("body = %q after %d attempts, want ok after 3", body, len(resp.Attempts))
	}
	// Envoy's defaults: the first wait is drawn below 25ms, the second below 50ms.
	if waits := clock.recorded(); len(waits) != 2 || waits[0] != 7*time.Millisecond || waits[1] != 33*time.Millisecond {
		t.Fatalf("back-off waits = %v, want 7ms then 33ms", waits)
	}
}

func TestDoReturnsTheLastOutcomeWhenRetriesRunOut(t *testing.T) {
	failing := backend(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusBadGateway) })
	refused := EndpointSpec{Name: "refused", Scheme: "http", Host: "127.0.0.1", Port: closedPort(t), Weight: 1}
	set := newSet(t, Options{Clock: &waitRecorder{}},
		retrying("retry-502", "5xx", 2, endpointOf(t, "e", failing)),
		retrying("retry-refused", "connect-failure", 1, refused))

	resp, err := set.Do(t.Context(), post("retry-502"))
	if err != nil || resp.StatusCode != http.StatusBadGateway || resp.Local != nil || len(resp.Attempts) != 3 {
		t.Fatalf("502 backend: %v %+v", err, resp)
	}
	_, _ = readAll(t, resp)

	resp, err = set.Do(t.Context(), post("retry-refused"))
	if err != nil {
		t.Fatal(err)
	}
	assertLocalReply(t, resp, http.StatusServiceUnavailable, "upstream connect error or disconnect/reset before headers. "+
		"retried and the latest reset reason: remote connection failure, transport failure reason: delayed connect error: 111")
}

func TestDoRetryPrefersAnotherEndpoint(t *testing.T) {
	refused := EndpointSpec{Name: "refused", Scheme: "http", Host: "127.0.0.1", Port: closedPort(t), Weight: 1}
	healthy := endpointOf(t, "healthy", backend(t, func(w http.ResponseWriter, r *http.Request) {}))
	// The rotation starts at the refused endpoint.
	set := newSet(t, Options{Clock: &waitRecorder{}, Rand: &scriptedRand{values: []uint64{0, 0}}},
		retrying("retry-other", "connect-failure", 1, refused, healthy))
	resp, err := set.Do(t.Context(), post("retry-other"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if len(resp.Attempts) != 2 || resp.Attempts[0].Endpoint != "refused" || resp.Endpoint.Name != "healthy" {
		t.Fatalf("attempts = %+v served by %s", resp.Attempts, resp.Endpoint.Name)
	}
}

func TestDoRetriesAStalledFirstByteBeforeCommitting(t *testing.T) {
	var hits atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		if hits.Add(1) == 1 {
			<-r.Context().Done()
			return
		}
		_, _ = io.WriteString(w, "data: token\n\n")
	})
	spec := retrying("retry-first-byte", "reset", 1, endpointOf(t, "e", server))
	spec.Policy.Timeouts.FirstByte = 100 * time.Millisecond
	set := newSet(t, Options{Clock: &waitRecorder{}}, spec)
	resp, err := set.Do(t.Context(), post("retry-first-byte"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); body != "data: token\n\n" {
		t.Fatalf("body = %q", body)
	}
	first := resp.Attempts[0].Err
	if len(resp.Attempts) != 2 || first == nil || first.Stage != StageFirstByte {
		t.Fatalf("attempts = %+v, want a first-byte timeout then a success", resp.Attempts)
	}
}

func TestDoRetriesAPerTryTimeout(t *testing.T) {
	var hits atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			awaitDisconnect(r)
			return
		}
		_, _ = io.WriteString(w, "ok")
	})
	spec := retrying("retry-per-try", "5xx", 1, endpointOf(t, "e", server))
	spec.Policy.Timeouts.PerTry = 100 * time.Millisecond
	set := newSet(t, Options{Clock: &waitRecorder{}}, spec)
	resp, err := set.Do(t.Context(), post("retry-per-try"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); body != "ok" || resp.Attempts[0].Err == nil || resp.Attempts[0].Err.Stage != StagePerTry {
		t.Fatalf("body = %q attempts = %+v", body, resp.Attempts)
	}
}

func TestDoNeverRetriesAfterBytesReachTheCaller(t *testing.T) {
	var hits atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Length", "1000")
		_, _ = io.WriteString(w, "data: partial")
		w.(http.Flusher).Flush()
		hijackAndClose(w)
	})
	spec := retrying("never-after-bytes", "5xx,reset,connect-failure,gateway-error", 5, endpointOf(t, "e", server))
	set := newSet(t, Options{Clock: &waitRecorder{}}, spec)
	resp, err := set.Do(t.Context(), post("never-after-bytes"))
	if err != nil {
		t.Fatal(err)
	}
	body, readErr := readAll(t, resp)
	if body != "data: partial" || KindOf(readErr) != KindReset {
		t.Fatalf("body = %q err = %v, want the partial body then a reset", body, readErr)
	}
	if hits.Load() != 1 || len(resp.Attempts) != 1 {
		t.Fatalf("backend hits = %d attempts = %d after the stream broke; want no retry", hits.Load(), len(resp.Attempts))
	}
}

func TestDoHonorsRetryAfterOnARetriableStatus(t *testing.T) {
	var hits atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = io.WriteString(w, "ok")
	})
	clock := &waitRecorder{}
	spec := retrying("retry-after", "retriable-status-codes", 1, endpointOf(t, "e", server))
	spec.Policy.Retry.RetryAfterMax = 5 * time.Second
	set := newSet(t, Options{Clock: clock}, spec)
	resp, err := set.Do(t.Context(), post("retry-after"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if waits := clock.recorded(); len(waits) != 1 || waits[0] < time.Second || waits[0] >= 1500*time.Millisecond {
		t.Fatalf("waits = %v, want one Retry-After wait of 1s plus jitter", waits)
	}
}

func TestDoTotalTimeoutCutsTheBackOffShort(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) })
	spec := retrying("retry-deadline", "5xx", 3, endpointOf(t, "e", server))
	spec.Policy.Retry.BackOffBase = 10 * time.Second
	spec.Policy.Retry.BackOffMax = 10 * time.Second
	spec.Policy.Timeouts.Total = 150 * time.Millisecond
	set := newSet(t, Options{Rand: &scriptedRand{values: []uint64{0, 0, 9000}}}, spec)
	start := time.Now()
	resp, err := set.Do(t.Context(), post("retry-deadline"))
	if err != nil {
		t.Fatal(err)
	}
	assertLocalReply(t, resp, http.StatusGatewayTimeout, "upstream request timeout")
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("the total timeout fired after %v", elapsed)
	}
}
