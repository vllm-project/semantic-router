package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// A balance-style rule must retain its nested logic and the router's result,
// including when an unknown signal is resolved by on_unknown policy.
const balancePreviewTrace = `[{"decision_name":"formal_math_proof","state":"unknown","matched":true,"on_unknown":"match","confidence":0,"root_trace":{"node_type":"AND","state":"unknown","matched":false,"confidence":0,"children":[{"node_type":"leaf","signal_type":"domain","signal_name":"math","state":"true","matched":true,"confidence":0.9},{"node_type":"NOT","state":"unknown","matched":false,"confidence":0,"children":[{"node_type":"OR","state":"unknown","matched":false,"confidence":0,"children":[{"node_type":"leaf","signal_type":"fact_check","signal_name":"verification_required","state":"unknown","matched":false,"confidence":0,"signal_error":"classifier unavailable"},{"node_type":"leaf","signal_type":"keyword","signal_name":"correction_feedback_markers","state":"false","matched":false,"confidence":0}]}]}]}}]`

func TestTopologyPreviewPreservesNestedRuntimeTrace(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "true", r.URL.Query().Get("trace"))
		var request RouterIntentRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		require.Equal(t, "vllm-sr/balance", request.Model)
		require.NoError(t, json.NewEncoder(w).Encode(RouterEvalResponse{
			RoutingDecision:        "formal_math_proof",
			EvalTrace:              json.RawMessage(balancePreviewTrace),
			RecommendedModels:      []string{"candidate-a", "candidate-b"},
			SelectedModel:          "candidate-b",
			SelectionStatus:        "selected",
			SelectionMethod:        "static",
			SignalErrors:           map[string]string{"fact_check:verification_required": "classifier unavailable"},
			AppliedUnknownPolicies: map[string]string{"formal_math_proof": "match"},
		}))
	}))
	defer server.Close()

	handler := TopologyTestQueryHandler("missing-config.yaml", server.URL)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(`{"query":"Prove this theorem","model":"vllm-sr/balance"}`)))
	require.Equal(t, http.StatusOK, recorder.Code)
	var result TestQueryResult
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
	require.JSONEq(t, balancePreviewTrace, string(result.EvalTrace))
	require.Len(t, result.EvaluatedRules, 1)
	rule := result.EvaluatedRules[0]
	require.Equal(t, "AND(domain:math, NOT(OR(fact_check:verification_required, keyword:correction_feedback_markers)))", rule.Expression)
	require.Equal(t, "unknown", rule.State)
	require.True(t, rule.IsMatch, "the router applied on_unknown; the dashboard must not reevaluate it")
	require.Equal(t, []string{"candidate-b"}, result.MatchedModels)
	require.Equal(t, []string{"candidate-a", "candidate-b"}, result.RecommendedModels)
	require.Contains(t, result.HighlightedPath, "model-candidate-b")
	require.NotContains(t, result.HighlightedPath, "model-candidate-a")
	require.Equal(t, "selected", result.SelectionStatus)
	require.Equal(t, "match", result.AppliedUnknownPolicies["formal_math_proof"])
	require.NotEmpty(t, result.Warning)
}

func TestTopologyPreviewPreservesUnresolvedDecisionDiagnostics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		require.NoError(t, json.NewEncoder(w).Encode(RouterEvalResponse{
			EvalTrace:     json.RawMessage(balancePreviewTrace),
			DecisionError: "decision unresolved",
			SignalErrors:  map[string]string{"fact_check:verification_required": "classifier unavailable"},
		}))
	}))
	defer server.Close()

	recorder := httptest.NewRecorder()
	TopologyTestQueryHandler("missing-config.yaml", server.URL).ServeHTTP(recorder,
		httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(`{"query":"hello"}`)))
	require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	var result TestQueryResult
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
	require.False(t, result.IsAccurate)
	require.Equal(t, "decision unresolved", result.Warning)
	require.Equal(t, "decision unresolved", result.DecisionError)
	require.JSONEq(t, balancePreviewTrace, string(result.EvalTrace))
	require.NotNil(t, result.MatchedSignals)
	require.NotNil(t, result.MatchedModels)
	require.Equal(t, "classifier unavailable", result.SignalErrors["fact_check:verification_required"])
}

func TestTopologyPreviewDoesNotInventSelectionOrEvaluation(t *testing.T) {
	result := convertRouterResponse(TestQueryRequest{}, &RouterEvalResponse{
		RoutingDecision:   "route",
		RecommendedModels: []string{"candidate-a", "candidate-b"},
		SelectionStatus:   "execution_required",
		SelectionReason:   "selection requires model execution",
	}, "missing-config.yaml")
	require.Empty(t, result.MatchedModels)
	require.Empty(t, result.EvaluatedRules)
	require.Equal(t, "selection requires model execution", result.SelectionReason)
}

func TestTopologyPreviewFailureResponsesHaveUsableArrays(t *testing.T) {
	for _, test := range []struct {
		name         string
		body         string
		routerStatus int
		wantStatus   int
		wantWarning  string
	}{
		{"unavailable", "maintenance", http.StatusServiceUnavailable, http.StatusServiceUnavailable, "Router API error (status 503)"},
		{"invalid JSON", "not JSON", http.StatusOK, http.StatusBadGateway, "Failed to parse Router API response"},
		{"invalid routing model", `{"error":{"code":"INVALID_ROUTING_MODEL","message":"unknown routing model"}}`, http.StatusBadRequest, http.StatusBadRequest, "INVALID_ROUTING_MODEL: unknown routing model"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(test.routerStatus)
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			recorder := httptest.NewRecorder()
			TopologyTestQueryHandler("", server.URL).ServeHTTP(recorder,
				httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(`{"query":"hello"}`)))
			require.Equal(t, test.wantStatus, recorder.Code)
			var result TestQueryResult
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
			require.False(t, result.IsAccurate)
			require.Equal(t, test.wantWarning, result.Warning)
			require.NotNil(t, result.MatchedSignals)
			require.NotNil(t, result.MatchedModels)
			require.Empty(t, result.MatchedDecision)
		})
	}
}

func TestTopologyPreviewHonorsCallerCancellation(t *testing.T) {
	started := make(chan struct{})
	canceled := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
		close(canceled)
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	finished := make(chan *TestQueryResult, 1)
	go func() {
		finished <- callRouterAPI(ctx, TestQueryRequest{Query: "hello", Mode: TestQueryModeDryRun}, server.URL, "")
	}()
	<-started
	cancel()
	select {
	case result := <-finished:
		require.False(t, result.IsAccurate)
		require.Equal(t, http.StatusBadGateway, result.HTTPStatus)
	case <-time.After(time.Second):
		t.Fatal("preview did not stop after caller canceled")
	}
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("router request was not canceled")
	}
}

func TestTopologyPreviewAllowsColdClassifierWarmup(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		warmup := time.NewTimer(11 * time.Second)
		defer warmup.Stop()
		select {
		case <-warmup.C:
			_ = json.NewEncoder(w).Encode(RouterEvalResponse{RoutingDecision: "balance"})
		case <-r.Context().Done():
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	result := callRouterAPI(ctx, TestQueryRequest{Query: "hello", Mode: TestQueryModeDryRun}, server.URL, "")
	require.True(t, result.IsAccurate, "cold preview must not fail at the former ten-second timeout: %s", result.Warning)
	require.Equal(t, "balance", result.MatchedDecision)
}
