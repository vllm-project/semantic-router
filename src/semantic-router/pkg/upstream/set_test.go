package upstream

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestDoRoutesByKeyAndRecordsTheAttempt(t *testing.T) {
	var gotA, gotB atomic.Int32
	a := backend(t, func(w http.ResponseWriter, r *http.Request) { gotA.Add(1); _, _ = io.WriteString(w, "from-a") })
	b := backend(t, func(w http.ResponseWriter, r *http.Request) { gotB.Add(1); _, _ = io.WriteString(w, "from-b") })
	set := newSet(t, Options{}, clusterOf("route-a", endpointOf(t, "a1", a)), clusterOf("route-b", endpointOf(t, "b1", b)))

	resp, err := set.Do(t.Context(), post("route-b"))
	if err != nil {
		t.Fatal(err)
	}
	body, err := readAll(t, resp)
	if err != nil || body != "from-b" {
		t.Fatalf("body = %q, err = %v", body, err)
	}
	if resp.Cluster != "route-b" || resp.DefaultRoute || resp.Endpoint.Name != "b1" || resp.StatusCode != 200 {
		t.Fatalf("response = %+v", resp)
	}
	if len(resp.Attempts) != 1 || resp.Attempts[0].Endpoint != "b1" || resp.Attempts[0].StatusCode != 200 ||
		resp.Attempts[0].Err != nil {
		t.Fatalf("attempts = %+v", resp.Attempts)
	}
	if gotA.Load() != 0 || gotB.Load() != 1 {
		t.Fatalf("backend hits a=%d b=%d", gotA.Load(), gotB.Load())
	}
	if got := testutil.ToFloat64(metrics.UpstreamAttemptsTotal.WithLabelValues("route-b", "b1", "2xx")); got != 1 {
		t.Fatalf("attempt metric = %v, want 1", got)
	}
	if got := testutil.ToFloat64(metrics.UpstreamStreamsTotal.WithLabelValues("route-b", "b1", "complete")); got != 1 {
		t.Fatalf("stream metric = %v, want 1", got)
	}
}

func TestDoDefaultRouteRewritesPathAndAddsRouteHeaders(t *testing.T) {
	type seen struct{ uri, host, tenant, userAgent, encoding, looper string }
	got := make(chan seen, 4)
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		got <- seen{
			uri: r.RequestURI, host: r.Host, tenant: r.Header.Get("X-Tenant"),
			userAgent: r.Header.Get("User-Agent"), encoding: r.Header.Get("Accept-Encoding"),
			looper: r.Header.Get("X-Vsr-Looper-Request"),
		}
	})
	ep := endpointOf(t, "e", server)
	first := clusterOf("first", ep)
	first.PathPrefix = "/compatible-mode/v1"
	first.RouteHeaders = []Header{{Name: "X-Tenant", Value: "operator"}}
	second := clusterOf("second", ep)
	set := newSet(t, Options{}, first, second)

	for _, key := range []string{"", "unknown-model"} {
		req := post(key)
		req.Header.Set("X-Vsr-Looper-Request", "true")
		resp, err := set.Do(t.Context(), req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = readAll(t, resp)
		if !resp.DefaultRoute || resp.Cluster != "first" {
			t.Fatalf("key %q: cluster %q default %v", key, resp.Cluster, resp.DefaultRoute)
		}
		want := seen{uri: "/compatible-mode/v1/chat/completions", host: ep.Authority(), tenant: "operator"}
		if s := <-got; s != want {
			t.Fatalf("key %q: backend saw %+v, want %+v", key, s, want)
		}
	}

	// A routed request keeps the provider path the router set.
	resp, err := set.Do(t.Context(), post("second"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if s := <-got; s.uri != "/v1/chat/completions" || s.tenant != "" {
		t.Fatalf("routed request: backend saw %+v", s)
	}
}

func TestDoStreamsEachChunkAsItArrives(t *testing.T) {
	step := make(chan struct{})
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := range 3 {
			_, _ = fmt.Fprintf(w, "data: %d\n\n", i)
			w.(http.Flusher).Flush()
			if i < 2 {
				<-step
			}
		}
	})
	set := newSet(t, Options{}, clusterOf("stream", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("stream"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 64)
	for i := range 3 {
		n, err := io.ReadAtLeast(resp.Body, buf, len("data: 0\n\n"))
		if err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		if got, want := string(buf[:n]), fmt.Sprintf("data: %d\n\n", i); got != want {
			t.Fatalf("chunk %d = %q, want %q", i, got, want)
		}
		// The backend sends the next event only after this one was read, so
		// a body buffered until the end would deadlock here.
		if i < 2 {
			step <- struct{}{}
		}
	}
	if _, err := resp.Body.Read(buf); err != io.EOF {
		t.Fatalf("after the last chunk: %v, want EOF", err)
	}
}

func TestCopyFlushesEveryChunk(t *testing.T) {
	step := make(chan struct{})
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		for i := range 2 {
			_, _ = fmt.Fprintf(w, "data: %d\n\n", i)
			w.(http.Flusher).Flush()
			if i == 0 {
				<-step
			}
		}
	})
	set := newSet(t, Options{}, clusterOf("copy", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("copy"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	dst := &flushRecorder{flushed: make(chan string, 4)}
	done := make(chan error, 1)
	go func() { _, err := Copy(dst, resp.Body); done <- err }()
	if got := <-dst.flushed; got != "data: 0\n\n" {
		t.Fatalf("first flush = %q", got)
	}
	step <- struct{}{}
	if got := <-dst.flushed; got != "data: 1\n\n" {
		t.Fatalf("second flush = %q", got)
	}
	if err := <-done; err != nil {
		t.Fatalf("Copy: %v", err)
	}
}

type flushRecorder struct {
	mu      sync.Mutex
	pending []byte
	flushed chan string
}

func (f *flushRecorder) Write(p []byte) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pending = append(f.pending, p...)
	return len(p), nil
}

func (f *flushRecorder) Flush() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushed <- string(f.pending)
	f.pending = nil
}

func TestDoAppliesBackpressureToASlowReader(t *testing.T) {
	const total = 64 << 20
	var written atomic.Int64
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		chunk := make([]byte, 32<<10)
		for written.Load() < total {
			n, err := w.Write(chunk)
			written.Add(int64(n))
			if err != nil {
				return
			}
		}
	})
	set := newSet(t, Options{}, clusterOf("slow-reader", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("slow-reader"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	// Nothing reads for a while: only socket buffers may fill.
	time.Sleep(300 * time.Millisecond)
	if got := written.Load(); got >= 32<<20 {
		t.Fatalf("backend wrote %d bytes while nobody read; the body is buffered", got)
	}
	n, err := io.Copy(io.Discard, resp.Body)
	if err != nil || n != total {
		t.Fatalf("read %d bytes, err %v; want %d", n, err, total)
	}
}

func TestDoTimeoutsByStage(t *testing.T) {
	stall := func(before time.Duration, firstChunk bool, after time.Duration) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			select {
			case <-time.After(before):
			case <-r.Context().Done():
				return
			}
			w.WriteHeader(http.StatusOK)
			if firstChunk {
				_, _ = io.WriteString(w, "data: first\n\n")
			}
			w.(http.Flusher).Flush()
			select {
			case <-time.After(after):
			case <-r.Context().Done():
				return
			}
			_, _ = io.WriteString(w, "data: late\n\n")
		}
	}
	tests := []struct {
		name     string
		handler  http.HandlerFunc
		timeouts Timeouts
		// atDo is true when Do itself fails; otherwise the body read does.
		atDo  bool
		stage TimeoutStage
	}{
		{"total before headers", stall(2*time.Second, false, 0), Timeouts{Total: 100 * time.Millisecond}, true, StageTotal},
		{"per try before headers", stall(2*time.Second, false, 0), Timeouts{PerTry: 100 * time.Millisecond}, true, StagePerTry},
		{"first byte after headers", stall(0, false, 2*time.Second), Timeouts{FirstByte: 100 * time.Millisecond}, true, StageFirstByte},
		{"idle mid stream", stall(0, true, 2*time.Second), Timeouts{Idle: 100 * time.Millisecond}, false, StageIdle},
		{"total mid stream", stall(0, true, 2*time.Second), Timeouts{Total: 150 * time.Millisecond}, false, StageTotal},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name := fmt.Sprintf("timeouts-%d", i)
			set := newSet(t, Options{}, clusterOf(name, endpointOf(t, "e", backend(t, tt.handler))))
			start := time.Now()
			resp, err := set.Do(t.Context(), withTimeouts(post(name), tt.timeouts))
			if err != nil {
				t.Fatalf("Do: %v", err)
			}
			var failure *Error
			if tt.atDo {
				failure = resp.Local
				assertLocalReply(t, resp, http.StatusGatewayTimeout, "upstream request timeout")
			} else {
				_, readErr := readAll(t, resp)
				asUpstreamError(readErr, &failure)
			}
			if failure == nil || failure.Kind != KindTimeout || failure.Stage != tt.stage {
				t.Fatalf("failure = %v, want a %s timeout", failure, tt.stage)
			}
			if elapsed := time.Since(start); elapsed > time.Second {
				t.Fatalf("the %s timeout took %v", tt.stage, elapsed)
			}
		})
	}
}

// assertLocalReply checks a response is Envoy's local reply, byte for byte.
func assertLocalReply(t *testing.T, resp *Response, status int, body string) {
	t.Helper()
	if resp.Local == nil || resp.StatusCode != status {
		t.Fatalf("response = %d local %v, want a %d local reply", resp.StatusCode, resp.Local, status)
	}
	got, err := readAll(t, resp)
	if err != nil || got != body {
		t.Fatalf("local reply body = %q, err %v; want %q", got, err, body)
	}
	if resp.Header.Get("Content-Type") != "text/plain" || resp.Header.Get("Content-Length") != fmt.Sprint(len(body)) {
		t.Fatalf("local reply header = %v", resp.Header)
	}
}

func TestDoFirstByteWaitsForTheBodyBeforeReturning(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		time.Sleep(200 * time.Millisecond)
		_, _ = io.WriteString(w, "data: token\n\n")
	})
	set := newSet(t, Options{}, clusterOf("first-byte", endpointOf(t, "e", server)))
	start := time.Now()
	resp, err := set.Do(t.Context(), withTimeouts(post("first-byte"), Timeouts{FirstByte: 2 * time.Second}))
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("Do returned after %v, before the first byte", elapsed)
	}
	if body, err := readAll(t, resp); err != nil || body != "data: token\n\n" {
		t.Fatalf("body = %q, err = %v", body, err)
	}
}

func TestDoAnswersConnectionFailuresAsEnvoy(t *testing.T) {
	refused := EndpointSpec{Name: "refused", Scheme: "http", Host: "127.0.0.1", Port: closedPort(t), Weight: 1}
	reset := endpointOf(t, "reset", backend(t, func(w http.ResponseWriter, r *http.Request) { hijackAndClose(w) }))
	set := newSet(t, Options{}, clusterOf("refused", refused), clusterOf("reset", reset))

	tests := map[string]struct {
		kind Kind
		body string
	}{
		"refused": {KindConnectFailure, "upstream connect error or disconnect/reset before headers. " +
			"reset reason: remote connection failure, transport failure reason: delayed connect error: 111"},
		"reset": {KindReset, "upstream connect error or disconnect/reset before headers. " +
			"reset reason: connection termination"},
	}
	for key, want := range tests {
		resp, err := set.Do(t.Context(), post(key))
		if err != nil {
			t.Fatalf("%s: Do: %v", key, err)
		}
		failure := resp.Local
		if failure == nil || failure.Kind != want.kind || failure.Cluster != key || failure.Endpoint != key {
			t.Fatalf("%s: failure = %+v, want %s", key, failure, want.kind)
		}
		assertLocalReply(t, resp, http.StatusServiceUnavailable, want.body)
		if len(resp.Attempts) != 1 || resp.Attempts[0].Err != failure {
			t.Fatalf("%s: attempts = %+v", key, resp.Attempts)
		}
	}
}

func TestDoSurfacesAMidStreamDropOnRead(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		_, _ = io.WriteString(w, "partial")
		w.(http.Flusher).Flush()
		hijackAndClose(w)
	})
	set := newSet(t, Options{}, clusterOf("drop", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("drop"))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	body, err := readAll(t, resp)
	if KindOf(err) != KindReset || body != "partial" {
		t.Fatalf("body = %q, err = %v; want the partial body then a reset", body, err)
	}
	if len(resp.Attempts) != 1 {
		t.Fatalf("attempts = %+v, want one", resp.Attempts)
	}
}

func TestDoHonorsCallerCancellation(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) { awaitDisconnect(r) })
	set := newSet(t, Options{}, clusterOf("cancel", endpointOf(t, "e", server)))
	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(50*time.Millisecond, cancel)
	_, err := set.Do(ctx, post("cancel"))
	if KindOf(err) != KindCanceled {
		t.Fatalf("error = %v, want %s", err, KindCanceled)
	}
}

func TestDoRemovesHopByHopResponseHeaders(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Connection", "X-Hop")
		w.Header().Set("X-Hop", "1")
		w.Header().Set("Keep-Alive", "timeout=5")
		w.Header().Set("X-Kept", "yes")
	})
	set := newSet(t, Options{}, clusterOf("hop", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("hop"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if resp.Header.Get("X-Hop") != "" || resp.Header.Get("Keep-Alive") != "" || resp.Header.Get("X-Kept") != "yes" {
		t.Fatalf("response header = %v", resp.Header)
	}
}

func TestDoBalancesAcrossEndpoints(t *testing.T) {
	handler := func(name string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, name) }
	}
	a := endpointOf(t, "a", backend(t, handler("a")))
	b := endpointOf(t, "b", backend(t, handler("b")))
	set := newSet(t, Options{Rand: rand.New(rand.NewPCG(1, 2))}, clusterOf("rr", a, b))
	var order string
	for range 4 {
		resp, err := set.Do(t.Context(), post("rr"))
		if err != nil {
			t.Fatal(err)
		}
		body, _ := readAll(t, resp)
		order += body
	}
	if order != "abab" && order != "baba" {
		t.Fatalf("round robin order = %q", order)
	}
}

func TestDoLeastRequestAvoidsTheBusyEndpoint(t *testing.T) {
	release := make(chan struct{})
	busy := endpointOf(t, "busy", backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		<-release
	}))
	idle := endpointOf(t, "idle", backend(t, func(w http.ResponseWriter, r *http.Request) {}))
	spec := clusterOf("lr", busy, idle)
	spec.LBPolicy = LBLeastRequest
	// Both samples land on the busy endpoint first, then on each in turn.
	set := newSet(t, Options{Rand: &scriptedRand{values: []uint64{0, 0, 0, 0, 1}}}, spec)
	held, err := set.Do(t.Context(), post("lr"))
	if err != nil || held.Endpoint.Name != "busy" {
		t.Fatalf("first call: %v %+v", err, held)
	}
	defer func() { close(release); _, _ = readAll(t, held) }()
	resp, err := set.Do(t.Context(), post("lr"))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = readAll(t, resp)
	if resp.Endpoint.Name != "idle" {
		t.Fatalf("second call went to %s while busy had a request in flight", resp.Endpoint.Name)
	}
}

func TestDoConcurrentCallsReleaseEverything(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, "ok") })
	a := endpointOf(t, "a", server)
	b := a
	b.Name = "b"
	spec := clusterOf("concurrent", a, b)
	spec.LBPolicy = LBLeastRequest
	set := newSet(t, Options{}, spec)
	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 10 {
				resp, err := set.Do(context.Background(), post("concurrent"))
				if err != nil {
					t.Error(err)
					return
				}
				_, _ = readAll(t, resp)
			}
		}()
	}
	wg.Wait()
	for _, ep := range set.clusters["concurrent"].hosts.Load().hosts {
		if n := ep.active.Load(); n != 0 {
			t.Fatalf("endpoint %s still counts %d active requests", ep.spec.Name, n)
		}
	}
	if set.inflight != 0 {
		t.Fatalf("set still counts %d calls in flight", set.inflight)
	}
}

func asUpstreamError(err error, target **Error) bool {
	return errors.As(err, target)
}
