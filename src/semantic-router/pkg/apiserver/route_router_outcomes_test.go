//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
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
		RecordOnly: true,
	})
	if validationErr != nil {
		t.Fatalf("expected valid outcome, got %v", validationErr)
	}
	if outcome.ReplayID != "replay-1" ||
		outcome.TargetRef != "model-a" ||
		outcome.Score != 0.75 ||
		outcome.Metadata["run_id"] != "run-1" ||
		!outcome.RecordOnly {
		t.Fatalf("unexpected normalized outcome: %#v", outcome)
	}
}

func TestNormalizeRouterOutcomePreservesOptionalScore(t *testing.T) {
	for _, provided := range []bool{false, true} {
		req := RouterOutcomeRequest{ReplayID: "replay-1", Target: "model", Verdict: "good_fit"}
		if provided {
			req.Score = routerOutcomeFloatPtr(0)
		}
		outcome, err := normalizeRouterOutcomeRequest(req)
		if err != nil || outcome.Score != 0 || outcome.ScoreProvided != provided {
			t.Fatalf("score presence lost: provided=%t outcome=%+v err=%v", provided, outcome, err)
		}
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
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(learningOutcomeIdempotencyHeader, "outcome-key-1")
	req = req.WithContext(withManagementPrincipal(req.Context(), managementPrincipal{
		Role:        "operator",
		AuthEnabled: true,
	}))
	w := httptest.NewRecorder()

	server.handleRouterOutcome(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
	if learningRuntime.last == nil || learningRuntime.last.TargetRef != "model-a" {
		t.Fatalf("expected outcome forwarded to runtime, got %#v", learningRuntime.last)
	}
	if learningRuntime.last.Source != routerruntime.RouterOutcomeSourceOperator {
		t.Fatalf("expected principal-derived source operator, got %#v", learningRuntime.last.Source)
	}
	if learningRuntime.last.IdempotencyKey != "outcome-key-1" {
		t.Fatalf("expected idempotency key forwarded, got %#v", learningRuntime.last)
	}
	var response RouterOutcomeResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("expected JSON response: %v", err)
	}
	if response.Updated != 1 || !response.Recorded {
		t.Fatalf("expected outcome response to expose update and replay recording, got %#v", response)
	}
}

func TestHandleRouterOutcomeRequiresIdempotencyKey(t *testing.T) {
	body, _ := json.Marshal(RouterOutcomeRequest{
		ReplayID: "replay-1",
		Target:   "model",
		Verdict:  "good_fit",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleRouterOutcome(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleRouterOutcomeUsesAuthenticatedUserProvenance(t *testing.T) {
	learningRuntime := &routerOutcomeLearningRuntime{}
	runtimeRegistry := routerruntime.NewRegistry(nil)
	runtimeRegistry.SetLearningRuntime(learningRuntime)
	server := &ClassificationAPIServer{runtimeRegistry: runtimeRegistry}
	body, _ := json.Marshal(RouterOutcomeRequest{
		ReplayID: "replay-user",
		Source:   "operator",
		Target:   "model",
		Verdict:  "good_fit",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", bytes.NewReader(body))
	req.Header.Set(learningOutcomeIdempotencyHeader, "outcome-user-1")
	req.Header.Set(headers.VSROutcomeSource, "user")
	req.Header.Set(headers.VSROutcomePrincipal, "dashboard-session:session-1")
	req = req.WithContext(withManagementPrincipal(req.Context(), managementPrincipal{
		Role:        "dashboard_control_plane",
		AuthEnabled: true,
	}))
	w := httptest.NewRecorder()

	server.handleRouterOutcome(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
	if learningRuntime.last == nil || learningRuntime.last.Source != routerruntime.RouterOutcomeSourceUser {
		t.Fatalf("persisted outcome source = %#v, want user", learningRuntime.last)
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
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(learningOutcomeIdempotencyHeader, "outcome-key-score")
	w := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleRouterOutcome(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestLearningOutcomeIngestPolicyRateLimit(t *testing.T) {
	policy := &learningOutcomeIngestPolicy{
		limit:  2,
		window: learningOutcomeRateWindow,
		hits:   map[string][]time.Time{},
	}
	principal := managementPrincipal{Role: "operator", AuthEnabled: true}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", nil)
	req.Header.Set(learningOutcomeIdempotencyHeader, "k1")
	if _, _, err := policy.enforce(req, principal); err != nil {
		t.Fatalf("first request should pass: %#v", err)
	}
	req.Header.Set(learningOutcomeIdempotencyHeader, "k2")
	if _, _, err := policy.enforce(req, principal); err != nil {
		t.Fatalf("second request should pass: %#v", err)
	}
	req.Header.Set(learningOutcomeIdempotencyHeader, "k3")
	if _, _, err := policy.enforce(req, principal); err == nil || err.Code != "RATE_LIMITED" {
		t.Fatalf("expected rate limit, got %#v", err)
	}
}

func TestLearningOutcomeIngestPolicyIsolatesAuthenticatedSessions(t *testing.T) {
	policy := &learningOutcomeIngestPolicy{
		limit:  1,
		window: learningOutcomeRateWindow,
		hits:   map[string][]time.Time{},
	}
	principal := managementPrincipal{Role: "dashboard_control_plane", AuthEnabled: true}
	requestForSession := func(key, sessionID string) *http.Request {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", nil)
		req.Header.Set(learningOutcomeIdempotencyHeader, key)
		req.Header.Set(headers.VSROutcomeSource, "user")
		req.Header.Set(headers.VSROutcomePrincipal, "dashboard-session:"+sessionID)
		return req
	}

	if _, source, err := policy.enforce(requestForSession("a-1", "session-a"), principal); err != nil ||
		source != routerruntime.RouterOutcomeSourceUser {
		t.Fatalf("first session request = source %q error %#v", source, err)
	}
	if _, source, err := policy.enforce(requestForSession("b-1", "session-b"), principal); err != nil ||
		source != routerruntime.RouterOutcomeSourceUser {
		t.Fatalf("independent session request = source %q error %#v", source, err)
	}
	if _, _, err := policy.enforce(requestForSession("a-2", "session-a"), principal); err == nil ||
		err.Code != "RATE_LIMITED" {
		t.Fatalf("expected only session-a to be saturated, got %#v", err)
	}
}

func TestLearningOutcomeIngestPolicyRejectsDelegationFromOtherRoles(t *testing.T) {
	policy := newLearningOutcomeIngestPolicy()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/observability/outcomes", nil)
	req.Header.Set(learningOutcomeIdempotencyHeader, "operator-key")
	req.Header.Set(headers.VSROutcomeSource, "user")
	req.Header.Set(headers.VSROutcomePrincipal, "dashboard-session:session-a")

	_, _, err := policy.enforce(req, managementPrincipal{Role: "operator", AuthEnabled: true})
	if err == nil || err.Code != "INVALID_OUTCOME_PROVENANCE" {
		t.Fatalf("non-Dashboard delegated provenance error = %#v", err)
	}
}

func routerOutcomeFloatPtr(value float64) *float64 {
	return &value
}
