package upstream

import (
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestBreakerDefaultsAreEnvoys(t *testing.T) {
	got := Breakers{}.withDefaults()
	want := Breakers{MaxConnections: 1024, MaxPendingRequests: 1024, MaxRequests: 1024, MaxRetries: 3}
	if got != want {
		t.Fatalf("defaults = %+v, want %+v", got, want)
	}
}

func TestBreakerQueuesForAConnectionThenOverflows(t *testing.T) {
	overflows := metrics.UpstreamOverflowTotal.WithLabelValues("breaker-queue", "max_pending_requests")
	before := testutil.ToFloat64(overflows)
	b := newBreaker("breaker-queue", Breakers{MaxConnections: 2, MaxPendingRequests: 1, MaxRequests: 10})
	release1, err := b.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = b.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	admitted := make(chan func(), 1)
	go func() {
		release, waitErr := b.acquire(context.Background())
		if waitErr != nil {
			t.Error(waitErr)
		}
		admitted <- release
	}()
	waitUntil(t, func() bool { b.mu.Lock(); defer b.mu.Unlock(); return b.waiting.Len() == 1 })
	if _, err = b.acquire(context.Background()); KindOf(err) != KindOverflow {
		t.Fatalf("a request past max_pending_requests = %v, want overflow", err)
	}
	release1()
	select {
	case release := <-admitted:
		release()
	case <-time.After(5 * time.Second):
		t.Fatal("the queued request did not get the freed connection")
	}
	if got := testutil.ToFloat64(overflows) - before; got != 1 {
		t.Fatalf("overflow metric grew by %v, want 1", got)
	}
}

func TestBreakerMaxRequestsOverflowsWithoutQueueing(t *testing.T) {
	b := newBreaker("breaker-requests", Breakers{MaxConnections: 5, MaxRequests: 2})
	for range 2 {
		if _, err := b.acquire(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := b.acquire(context.Background()); KindOf(err) != KindOverflow {
		t.Fatalf("a request past max_requests = %v, want overflow", err)
	}
}

func TestBreakerQueuedRequestLeavesWhenItsDeadlinePasses(t *testing.T) {
	b := newBreaker("breaker-deadline", Breakers{MaxConnections: 1, MaxPendingRequests: 4})
	release, err := b.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err = b.acquire(ctx); KindOf(err) != KindTimeout {
		t.Fatalf("queued past its deadline = %v, want a timeout", err)
	}
	b.mu.Lock()
	waiting := b.waiting.Len()
	b.mu.Unlock()
	release()
	if waiting != 0 || b.active != 0 {
		t.Fatalf("waiting = %d active = %d after the request left", waiting, b.active)
	}
}

func TestBreakerLimitsConcurrentRetries(t *testing.T) {
	b := newBreaker("breaker-retries", Breakers{MaxRetries: 1})
	done, err := b.acquireRetry()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = b.acquireRetry(); KindOf(err) != KindOverflow {
		t.Fatalf("a second concurrent retry = %v, want overflow", err)
	}
	done()
	if _, err = b.acquireRetry(); err != nil {
		t.Fatalf("a retry after the first ended: %v", err)
	}
}

func TestDoAnswersOverflowAsEnvoy(t *testing.T) {
	release := make(chan struct{})
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		w.(http.Flusher).Flush()
		<-release
	})
	spec := clusterOf("overflow-e2e", endpointOf(t, "e", server))
	spec.Breakers = Breakers{MaxConnections: 1, MaxPendingRequests: 1}
	set := newSet(t, Options{}, spec)

	held, err := set.Do(t.Context(), post("overflow-e2e"))
	if err != nil || held.Local != nil {
		t.Fatalf("first call: %v %+v", err, held)
	}
	queued := make(chan *Response, 1)
	go func() {
		resp, doErr := set.Do(context.Background(), post("overflow-e2e"))
		if doErr != nil {
			t.Error(doErr)
		}
		queued <- resp
	}()
	c := set.clusters["overflow-e2e"]
	waitUntil(t, func() bool { c.breaker.mu.Lock(); defer c.breaker.mu.Unlock(); return c.breaker.waiting.Len() == 1 })

	rejected, err := set.Do(t.Context(), post("overflow-e2e"))
	if err != nil || KindOf(rejected.Local) != KindOverflow || rejected.Header.Get("X-Envoy-Overloaded") != "true" {
		t.Fatalf("third call: %v %+v", err, rejected)
	}
	assertLocalReply(t, rejected, http.StatusServiceUnavailable,
		"upstream connect error or disconnect/reset before headers. reset reason: overflow")

	close(release)
	_, _ = io.Copy(io.Discard, held.Body)
	_ = held.Body.Close()
	resp := <-queued
	if resp == nil || resp.Local != nil {
		t.Fatalf("the queued call did not get through: %+v", resp)
	}
	_, _ = readAll(t, resp)
}

func waitUntil(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition not reached")
		}
		time.Sleep(2 * time.Millisecond)
	}
}
