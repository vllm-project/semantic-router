package modelservice

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
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
