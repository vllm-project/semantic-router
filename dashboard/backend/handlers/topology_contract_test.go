package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func topologyScopedConfig(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(`version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - name: local
          provider: vllm
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  decisions: []
entrypoints:
  - model_names: [public/balanced]
    recipe: balanced
recipes:
  - name: balanced
    routing:
      decisions: []
  - name: unbound
    routing:
      decisions: []
`), 0o600))
	_, err := routerconfig.Parse(path)
	require.NoError(t, err)
	return path
}

func TestTopologyPreviewForwardsEnvelopeAndScope(t *testing.T) {
	const messages = `[{"role":"system","content":"Be brief"},{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]`
	const tools = `[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/routing/preview", r.URL.Path)
		require.Equal(t, "true", r.URL.Query().Get("trace"))
		var payload map[string]json.RawMessage
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		require.JSONEq(t, messages, string(payload["messages"]))
		require.JSONEq(t, tools, string(payload["tools"]))
		require.JSONEq(t, `{"tenant":"example"}`, string(payload["metadata"]))
		require.JSONEq(t, `"public/balanced"`, string(payload["model"]))
		require.NotContains(t, payload, "recipe", "the Router resolves the model, not a client-supplied recipe")
		require.NoError(t, json.NewEncoder(w).Encode(RouterEvalResponse{
			RequestedModel: "public/balanced", Recipe: "balanced",
			DecisionResult: &RouterEvalDecisionResult{DecisionName: "chosen", Algorithm: "static"},
		}))
	}))
	defer server.Close()
	recorder := httptest.NewRecorder()
	TopologyTestQueryHandler(topologyScopedConfig(t), server.URL).ServeHTTP(recorder,
		httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(
			`{"model":"public/balanced","recipe":"balanced","messages":`+messages+`,"tools":`+tools+`,"metadata":{"tenant":"example"}}`)))
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	var result TestQueryResult
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
	require.True(t, result.IsAccurate)
	require.Equal(t, "balanced", result.Recipe)
	require.Equal(t, "public/balanced", result.RequestedModel)
	require.Equal(t, "static", result.Algorithm)
}

func TestTopologyPreviewRejectsInvalidScopeBeforeDispatch(t *testing.T) {
	for _, tc := range []struct {
		name, model, recipe string
		status              int
	}{
		{"unbound", "", "unbound", http.StatusBadRequest},
		{"wrong recipe", "public/balanced", "unbound", http.StatusConflict},
		{"unknown model", "public/missing", "balanced", http.StatusConflict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Error("invalid scope must not reach the Router")
			}))
			defer server.Close()
			body, err := json.Marshal(TestQueryRequest{Query: "hello", Model: tc.model, Recipe: tc.recipe})
			require.NoError(t, err)
			recorder := httptest.NewRecorder()
			TopologyTestQueryHandler(topologyScopedConfig(t), server.URL).ServeHTTP(recorder,
				httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(string(body))))
			require.Equal(t, tc.status, recorder.Code, recorder.Body.String())
		})
	}
}

func TestTopologyPreviewRejectsMismatchedResponseScope(t *testing.T) {
	for _, tc := range []struct {
		name, model, recipe string
		status              int
	}{
		{"wrong recipe", "public/balanced", "default", http.StatusOK},
		{"wrong model", "public/other", "balanced", http.StatusOK},
		{"missing scope", "", "", http.StatusOK},
		{"error from another recipe", "public/balanced", "default", http.StatusServiceUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				require.NoError(t, json.NewEncoder(w).Encode(RouterEvalResponse{
					RequestedModel: tc.model, Recipe: tc.recipe, RoutingDecision: "wrong-route",
					EvalTrace: json.RawMessage(balancePreviewTrace),
				}))
			}))
			defer server.Close()
			recorder := httptest.NewRecorder()
			TopologyTestQueryHandler(topologyScopedConfig(t), server.URL).ServeHTTP(recorder,
				httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(`{"query":"hello","model":"public/balanced","recipe":"balanced"}`)))
			require.Equal(t, http.StatusConflict, recorder.Code, recorder.Body.String())
			var result TestQueryResult
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
			require.False(t, result.IsAccurate)
			require.Empty(t, result.EvalTrace)
			require.Empty(t, result.MatchedDecision)
			require.Equal(t, []string{"client"}, result.HighlightedPath)
		})
	}
}

func TestTopologyPreviewHighlightsOnlySelectedDecision(t *testing.T) {
	trace := json.RawMessage(`[{"decision_name":"rejected","state":"false","matched":false},{"decision_name":"selected","state":"true","matched":true},{"decision_name":"lower_priority_match","state":"true","matched":true}]`)
	result := convertRouterResponse(TestQueryRequest{}, &RouterEvalResponse{
		EvalTrace: trace, RoutingDecision: "selected", DecisionResult: &RouterEvalDecisionResult{DecisionName: "selected"},
	}, "")
	require.Len(t, result.EvaluatedRules, 3, "retain every evaluation for the trace viewer")
	require.Equal(t, []string{"client", "decision-selected"}, result.HighlightedPath)
}

func TestTopologyPreviewPreservesTimeoutAndFailRequestPolicy(t *testing.T) {
	const trace = `[{"decision_name":"unresolved","state":"unknown","matched":false,"on_unknown":"fail_request","confidence":0,"root_trace":{"node_type":"leaf","signal_type":"classifier","signal_name":"intent","state":"unknown","matched":false,"confidence":0,"signal_error":"classifier timed out"}}]`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		require.NoError(t, json.NewEncoder(w).Encode(RouterEvalResponse{
			EvalTrace: json.RawMessage(trace), DecisionError: "decision unresolved",
			SignalErrors:           map[string]string{"classifier:intent": "classifier timed out"},
			AppliedUnknownPolicies: map[string]string{"unresolved": "fail_request"},
		}))
	}))
	defer server.Close()
	recorder := httptest.NewRecorder()
	TopologyTestQueryHandler("", server.URL).ServeHTTP(recorder,
		httptest.NewRequest(http.MethodPost, "/api/topology/test-query", strings.NewReader(`{"query":"hello"}`)))
	require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	var result TestQueryResult
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
	require.JSONEq(t, trace, string(result.EvalTrace))
	require.Equal(t, "unknown", result.EvaluatedRules[0].State)
	require.Equal(t, "fail_request", result.AppliedUnknownPolicies["unresolved"])
	require.Equal(t, []string{"client"}, result.HighlightedPath)
	require.False(t, result.IsAccurate)
}
