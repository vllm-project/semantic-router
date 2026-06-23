//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type routerOutcomeLearningRuntime struct {
	last *routerruntime.RouterOutcome
}

func (r *routerOutcomeLearningRuntime) UpdateOutcome(
	_ context.Context,
	outcome *routerruntime.RouterOutcome,
) routerruntime.RouterOutcomeResult {
	r.last = outcome
	return routerruntime.RouterOutcomeResult{Updated: 1, Recorded: true}
}

func TestNormalizeRouterOutcomeRequestIncludesTargetRef(t *testing.T) {
	outcome, validationErr := normalizeRouterOutcomeRequest(RouterOutcomeRequest{
		ReplayID:  " replay-1 ",
		Source:    "agent",
		Target:    "model",
		TargetRef: " model-a ",
		Verdict:   "good_fit",
		Score:     routerOutcomeFloatPtr(0.75),
		Metadata: map[string]string{
			"run_id": " run-1 ",
		},
	})
	if validationErr != nil {
		t.Fatalf("expected valid outcome, got %v", validationErr)
	}
	if outcome.ReplayID != "replay-1" ||
		outcome.TargetRef != "model-a" ||
		outcome.Score != 0.75 ||
		outcome.Metadata["run_id"] != "run-1" {
		t.Fatalf("unexpected normalized outcome: %#v", outcome)
	}
}

func TestNormalizeRouterOutcomeRequestAcceptsProviderAndRouterTargets(t *testing.T) {
	for _, target := range []string{"provider", "router"} {
		outcome, validationErr := normalizeRouterOutcomeRequest(RouterOutcomeRequest{
			ReplayID: "replay-1",
			Source:   target,
			Target:   target,
			Verdict:  "failed",
		})
		if validationErr != nil {
			t.Fatalf("expected target %q to be valid, got %v", target, validationErr)
		}
		if string(outcome.Target) != target {
			t.Fatalf("expected target %q, got %#v", target, outcome)
		}
	}
}

func TestHandleRouterOutcomeUsesLearningRuntime(t *testing.T) {
	learningRuntime := &routerOutcomeLearningRuntime{}
	runtimeRegistry := routerruntime.NewRegistry(nil)
	runtimeRegistry.SetLearningRuntime(learningRuntime)
	server := &ClassificationAPIServer{runtimeRegistry: runtimeRegistry}

	body, _ := json.Marshal(RouterOutcomeRequest{
		ReplayID:  "replay-1",
		Source:    "agent",
		Target:    "model",
		TargetRef: "model-a",
		Verdict:   "good_fit",
		Score:     routerOutcomeFloatPtr(1),
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/router/outcomes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.handleRouterOutcome(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
	if learningRuntime.last == nil || learningRuntime.last.TargetRef != "model-a" {
		t.Fatalf("expected outcome forwarded to runtime, got %#v", learningRuntime.last)
	}
	var response RouterOutcomeResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("expected JSON response: %v", err)
	}
	if response.Updated != 1 || !response.Recorded {
		t.Fatalf("expected outcome response to expose update and replay recording, got %#v", response)
	}
}

func TestHandleRouterOutcomeRejectsInvalidScore(t *testing.T) {
	body, _ := json.Marshal(RouterOutcomeRequest{
		ReplayID: "replay-1",
		Source:   "agent",
		Target:   "model",
		Verdict:  "good_fit",
		Score:    routerOutcomeFloatPtr(2),
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/router/outcomes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleRouterOutcome(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
}

func routerOutcomeFloatPtr(value float64) *float64 {
	return &value
}
