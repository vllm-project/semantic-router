package modelservice

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRecoveryRequiresFreshModelCard(t *testing.T) {
	var phase, modelCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := phase.Load()
		switch r.URL.Path {
		case "/health":
			state, code := "ready", http.StatusOK
			if p == 1 {
				state, code = "starting", http.StatusServiceUnavailable
			}
			writeJSON(w, code, map[string]any{"api_version": "3.2.0", "status": state})
		case "/v1/models":
			modelCalls.Add(1)
			if p == 2 {
				writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": map[string]any{"code": "unavailable", "message": "metadata temporarily unavailable"}})
				return
			}
			question := "choice"
			if p == 3 {
				question = "span"
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"api_version": "3.2.0", "object": "list",
				"data": []any{map[string]any{"id": "primary", "object": "model", "family": "fixture", "ready": true, "surfaces": []string{"decisions"}, "question_types": []string{question}}},
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
	g := newGroup(plan, client, false)
	ctx := context.Background()
	g.refresh(ctx)
	served := g.models["primary"]
	if !served.ready.Load() || !served.card.Answers("choice") {
		t.Fatal("initial fixture was not ready")
	}
	phase.Store(1)
	g.refresh(ctx)
	if served.ready.Load() {
		t.Fatal("offline fixture remained ready")
	}
	phase.Store(2)
	g.refresh(ctx)
	var qTypes []string
	if served.card != nil {
		qTypes = served.card.QuestionTypes
	}
	t.Logf("health recovered, metadata 503: ready=%v, model-list calls=%d, question types=%v", served.ready.Load(), modelCalls.Load(), qTypes)
	if served.ready.Load() {
		t.Error("model became ready using stale card while /v1/models failed")
	}
	phase.Store(3)
	g.refresh(ctx)
	if !served.ready.Load() || !served.card.Answers("span") {
		t.Errorf("model failed to recover with fresh metadata: ready=%v, types=%v", served.ready.Load(), served.card.QuestionTypes)
	}
}

func TestRefreshDiscardsProbeFromExitedProcess(t *testing.T) {
	modelsEntered := make(chan struct{})
	releaseModels := make(chan struct{})
	var modelsOnce sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			writeJSON(w, http.StatusOK, map[string]any{"api_version": RuntimeAPIVersion, "status": "ready"})
		case "/v1/models":
			modelsOnce.Do(func() { close(modelsEntered) })
			<-releaseModels
			writeJSON(w, http.StatusOK, map[string]any{
				"api_version": RuntimeAPIVersion, "object": "list",
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
	plan := &processPlan{logical: "primary", name: "exited-process", members: map[string]string{"primary": "primary"}}
	g := newGroup(plan, client, false)

	refreshDone := make(chan struct{})
	go func() {
		defer close(refreshDone)
		g.refresh(context.Background())
	}()
	select {
	case <-modelsEntered:
	case <-refreshDone:
		t.Fatal("refresh returned before discovery began")
	}
	g.processExited(errors.New("killed"), time.Minute)
	close(releaseModels)
	<-refreshDone

	served := g.models["primary"]
	if served.ready.Load() {
		t.Error("probe from the exited process marked the model ready")
	}
	if served.state != "restarting" {
		t.Errorf("probe from the exited process overwrote state = %q, want restarting", served.state)
	}
	if served.card != nil {
		t.Errorf("probe from the exited process restored a card: %+v", served.card)
	}
}
