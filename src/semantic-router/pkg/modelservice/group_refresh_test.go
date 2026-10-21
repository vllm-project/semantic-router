package modelservice

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// A refresh that began before the process exited must not publish the exited
// process's readiness or its card: the process it describes is gone.
func TestARefreshThatOutlivesItsProcessCommitsNothing(t *testing.T) {
	released := make(chan struct{})
	inFlight := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			writeJSON(w, http.StatusOK, map[string]any{"api_version": "3.2.0", "status": "ready"})
		case "/v1/models":
			select {
			case inFlight <- struct{}{}:
			default:
			}
			<-released
			writeJSON(w, http.StatusOK, map[string]any{
				"api_version": "3.2.0", "object": "list",
				"data": []any{map[string]any{"id": "primary", "object": "model", "family": "fixture", "ready": true, "surfaces": []string{"decisions"}, "question_types": []string{"choice"}}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	plan := &processPlan{logical: "primary", name: "repro", members: map[string]string{"primary": "primary"}}
	g := newGroup(plan, client, true)

	done := make(chan struct{})
	go func() {
		defer close(done)
		g.refresh(context.Background())
	}()

	// Wait until the discovery of the card is in flight, then let the
	// process exit while it waits.
	select {
	case <-inFlight:
	case <-time.After(10 * time.Second):
		t.Fatal("the refresh never reached the model list")
	}
	g.processExited(errors.New("killed"), time.Minute)
	close(released)
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the refresh never returned")
	}

	served := g.models["primary"]
	if served.ready.Load() {
		t.Error("the exited process published its readiness")
	}
	if served.state != "restarting" {
		t.Errorf("state = %q, want restarting", served.state)
	}
	if served.card != nil {
		t.Error("the exited process published its card")
	}
}

// A refresh that begins after the exit describes the new process, so it
// commits as usual.
func TestARefreshAfterTheExitCommitsAsUsual(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			writeJSON(w, http.StatusOK, map[string]any{"api_version": "3.2.0", "status": "ready"})
		case "/v1/models":
			writeJSON(w, http.StatusOK, map[string]any{
				"api_version": "3.2.0", "object": "list",
				"data": []any{map[string]any{"id": "primary", "object": "model", "family": "fixture", "ready": true, "surfaces": []string{"decisions"}, "question_types": []string{"choice"}}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	plan := &processPlan{logical: "primary", name: "repro", members: map[string]string{"primary": "primary"}}
	g := newGroup(plan, client, true)

	g.processExited(errors.New("killed"), time.Minute)
	g.refresh(context.Background())

	served := g.models["primary"]
	if !served.ready.Load() {
		t.Fatal("the restarted process did not become ready")
	}
	if served.card == nil {
		t.Fatal("the restarted process published no card")
	}
}
