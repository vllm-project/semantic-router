package upstream

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"
)

func errorsAs[T error](err error, target *T) bool { return errors.As(err, target) }

func TestNewReusesUnchangedClustersAndSharesPools(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, "ok") })
	ep := endpointOf(t, "e", server)
	old := newSet(t, Options{}, clusterOf("kept", ep), clusterOf("changed", ep), clusterOf("removed", ep))

	changed := clusterOf("changed", ep)
	changed.Endpoints[0].Weight = 5
	next := newSet(t, Options{Previous: old}, clusterOf("kept", ep), changed, clusterOf("added", ep))

	if next.clusters["kept"] != old.clusters["kept"] {
		t.Fatal("an unchanged cluster was rebuilt")
	}
	if next.clusters["changed"] == old.clusters["changed"] {
		t.Fatal("a changed cluster was reused")
	}
	if next.clusters["added"].pool != old.clusters["kept"].pool {
		t.Fatal("a cluster in the same security domain did not share the connection pool")
	}
	if err := old.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	// The adopted cluster keeps serving after the old Set is gone.
	resp, err := next.Do(t.Context(), post("kept"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); body != "ok" {
		t.Fatalf("body = %q", body)
	}
	if _, err := old.Do(t.Context(), post("kept")); KindOf(err) != KindClosed {
		t.Fatalf("closed set answered: %v", err)
	}
}

func TestCloseDrainsInFlightStreams(t *testing.T) {
	release := make(chan struct{})
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: 1\n\n")
		w.(http.Flusher).Flush()
		<-release
		_, _ = io.WriteString(w, "data: 2\n\n")
	})
	set := newSet(t, Options{}, clusterOf("drain", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("drain"))
	if err != nil {
		t.Fatal(err)
	}
	closed := make(chan error, 1)
	go func() { closed <- set.Close(context.Background()) }()
	select {
	case err := <-closed:
		t.Fatalf("Close returned %v with a stream in flight", err)
	case <-time.After(100 * time.Millisecond):
	}
	close(release)
	if body, err := readAll(t, resp); err != nil || body != "data: 1\n\ndata: 2\n\n" {
		t.Fatalf("drained stream = %q, err = %v", body, err)
	}
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestCloseAbortsStreamsLeftAfterItsDeadline(t *testing.T) {
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: 1\n\n")
		w.(http.Flusher).Flush()
		awaitDisconnect(r)
	})
	set := newSet(t, Options{}, clusterOf("abort", endpointOf(t, "e", server)))
	resp, err := set.Do(t.Context(), post("abort"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	if closeErr := set.Close(ctx); !errors.Is(closeErr, context.DeadlineExceeded) {
		t.Fatalf("Close = %v, want the drain deadline", closeErr)
	}
	_, err = readAll(t, resp)
	if KindOf(err) != KindClosed {
		t.Fatalf("read after a forced close = %v, want %s", err, KindClosed)
	}
}
