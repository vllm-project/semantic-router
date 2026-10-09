package modelservice

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestReproRecoveryRequiresFreshModelCard(t *testing.T) {
	var phase, modelCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := phase.Load()
		switch r.URL.Path {
		case "/health":
			state, code := "ready", http.StatusOK
			if p == 1 {
				state, code = "starting", http.StatusServiceUnavailable
			}
			writeJSON(w, code, map[string]any{"api_version": "2.0.0", "status": state})
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
				"api_version": "2.0.0", "object": "list",
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
	t.Logf("health recovered, metadata 503: ready=%v, model-list calls=%d, question types=%v", served.ready.Load(), modelCalls.Load(), served.card.QuestionTypes)
	if served.ready.Load() {
		t.Error("model became ready using stale metadata after /v1/models failed")
	}
	phase.Store(3)
	for range 3 {
		g.refresh(ctx)
	}
	t.Logf("metadata endpoint repaired, after 3 more probes: ready=%v, model-list calls=%d, question types=%v", served.ready.Load(), modelCalls.Load(), served.card.QuestionTypes)
	if !served.card.Answers("span") {
		t.Error("recovered runtime's new capabilities were never fetched")
	}
}

// A recovering model whose entry is missing from a successful discovery
// response stays not-ready and keeps retrying, while a healthy peer in the
// same process remains ready throughout.
func TestRecoveryMissingModelEntryKeepsPeerReady(t *testing.T) {
	var phase atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := phase.Load()
		switch r.URL.Path {
		case "/health":
			primary := "ready"
			if p == 1 {
				primary = "starting"
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"api_version": "2.0.0", "status": "ready",
				"models": map[string]any{
					"primary": map[string]any{"status": primary},
					"peer":    map[string]any{"status": "ready"},
				},
			})
		case "/v1/models":
			question := "choice"
			if p == 3 {
				question = "span"
			}
			data := []any{cardFixture("peer", "choice")}
			if p != 2 {
				data = append([]any{cardFixture("primary", question)}, data...)
			}
			writeJSON(w, http.StatusOK, map[string]any{"api_version": "2.0.0", "object": "list", "data": data})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	plan := &processPlan{logical: "both", name: "repro-peers", members: map[string]string{"primary": "primary", "peer": "peer"}}
	g := newGroup(plan, client, false)
	ctx := context.Background()
	primary, peer := g.models["primary"], g.models["peer"]

	g.refresh(ctx)
	if !primary.ready.Load() || !peer.ready.Load() {
		t.Fatal("initial fixture was not ready")
	}
	phase.Store(1)
	g.refresh(ctx)
	if primary.ready.Load() {
		t.Error("offline model remained ready")
	}
	if !peer.ready.Load() {
		t.Error("healthy peer lost readiness when its sibling went offline")
	}
	phase.Store(2)
	g.refresh(ctx)
	if primary.ready.Load() {
		t.Error("model became ready while its entry was missing from discovery")
	}
	if !peer.ready.Load() || !peer.card.Answers("choice") {
		t.Error("healthy peer was disturbed by the missing-entry recovery")
	}
	phase.Store(3)
	g.refresh(ctx)
	if !primary.ready.Load() || !primary.card.Answers("span") {
		t.Error("recovered model did not pick up its fresh card")
	}
	if !peer.ready.Load() {
		t.Error("healthy peer lost readiness during recovery")
	}
}

func cardFixture(id, question string) map[string]any {
	return map[string]any{"id": id, "object": "model", "family": "fixture", "ready": true, "surfaces": []string{"decisions"}, "question_types": []string{question}}
}
