package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	routerdecision "github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func nestedDecisionTrace() []routerdecision.DecisionTrace {
	return []routerdecision.DecisionTrace{
		{
			DecisionName: "coding_decision",
			Matched:      true,
			Confidence:   0.92,
			RootTrace: &routerdecision.TraceNode{
				NodeType: "OR",
				Matched:  true,
				Children: []*routerdecision.TraceNode{
					{NodeType: "leaf", SignalType: "keyword", SignalName: "coding", Matched: true, Confidence: 1.0},
					{NodeType: "leaf", SignalType: "domain", SignalName: "code", Matched: false, Confidence: 0.0},
				},
			},
		},
		{
			// A decision the router also evaluated to true while evaluating
			// all signals, but did not select. Its leaves must not be
			// highlighted alongside the selected decision's.
			DecisionName: "reasoning_decision",
			Matched:      true,
			Confidence:   0.4,
			RootTrace: &routerdecision.TraceNode{
				NodeType:   "leaf",
				SignalType: "keyword",
				SignalName: "thinking",
				Matched:    true,
				Confidence: 0.4,
			},
		},
	}
}

func TestTopologyTestQueryHandler_UsesEvalTrace(t *testing.T) {
	configPath := setupTestConfig(t)
	defer func() { _ = os.RemoveAll(filepath.Dir(configPath)) }()

	routerAPIURL := setupMockRouterAPI(t, RouterEvalResponse{
		RequestedModel:    "gpt-4",
		RecommendedModels: []string{"gpt-4"},
		DecisionResult: &RouterEvalDecisionResult{
			DecisionName: "coding_decision",
			Algorithm:    "priority",
			MatchedSignals: &RouterMatchedSignals{
				Keywords: []string{"coding"},
			},
		},
		EvalTrace: nestedDecisionTrace(),
	})

	handler := TopologyTestQueryHandler(configPath, routerAPIURL)
	reqBody := TestQueryRequest{Query: "Help me debug this function", Mode: TestQueryModeDryRun}
	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/api/topology/test-query", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)

	var result TestQueryResult
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &result))

	assert.True(t, result.IsAccurate, "trace-driven result must remain accurate")
	assert.Equal(t, "priority", result.Algorithm)
	assert.Empty(t, result.Warning)
	require.Len(t, result.EvalTrace, 2, "eval_trace must pass through unchanged, including unselected decisions")
	assert.Equal(t, "coding_decision", result.EvalTrace[0].DecisionName)
	assert.Equal(t, "OR", result.EvalTrace[0].RootTrace.NodeType)
	require.Len(t, result.EvalTrace[0].RootTrace.Children, 2, "nested children must survive the round trip")

	// Only the selected decision's matched leaf is highlighted.
	assert.Contains(t, result.HighlightedPath, "signal-keyword-coding")
	assert.NotContains(t, result.HighlightedPath, "signal-keyword-thinking",
		"a decision that matched but was not selected must not highlight its leaves")

	// The flat fallback reconstruction must not run when a trace is present.
	assert.Empty(t, result.EvaluatedRules)
}

func TestTopologyTestQueryHandler_FallsBackWithoutTrace(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(fallbackTestConfig), 0o644))

	routerAPIURL := setupMockRouterAPI(t, RouterEvalResponse{
		RecommendedModels: []string{"gpt-4"},
		RoutingDecision:   "coding_decision",
		DecisionResult: &RouterEvalDecisionResult{
			DecisionName: "coding_decision",
			MatchedSignals: &RouterMatchedSignals{
				Keywords: []string{"coding"},
			},
		},
		// EvalTrace intentionally omitted.
	})

	handler := TopologyTestQueryHandler(configPath, routerAPIURL)
	reqBody := TestQueryRequest{Query: "Help me debug this function", Mode: TestQueryModeDryRun}
	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/api/topology/test-query", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	var result TestQueryResult
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &result))

	assert.False(t, result.IsAccurate, "a derived fallback must never report accurate")
	assert.Contains(t, result.Warning, "did not return an eval trace")
	assert.NotEmpty(t, result.EvaluatedRules, "fallback reconstruction should run when no trace is present")
	assert.Empty(t, result.EvalTrace)
}

func TestVerifyRecipeScope_FlagsMismatchAgainstSelectedScope(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(recipeScopedTestConfig), 0o644))

	t.Run("matching recipe leaves the result accurate", func(t *testing.T) {
		result := &TestQueryResult{IsAccurate: true}
		verifyRecipeScope(result, &RouterEvalResponse{
			RequestedModel: "vllm-sr/mom-balanced-v1",
			Recipe:         "balanced",
		}, configPath, "vllm-sr/mom-balanced-v1")

		assert.True(t, result.IsAccurate)
		assert.Empty(t, result.Warning)
		assert.Equal(t, "balanced", result.Recipe)
	})

	t.Run("mismatched recipe is flagged, not silently accepted", func(t *testing.T) {
		result := &TestQueryResult{IsAccurate: true}
		verifyRecipeScope(result, &RouterEvalResponse{
			RequestedModel: "vllm-sr/mom-balanced-v1",
			Recipe:         "default",
		}, configPath, "vllm-sr/mom-balanced-v1")

		assert.False(t, result.IsAccurate)
		assert.Contains(t, result.Warning, `evaluated recipe "default"`)
		assert.Contains(t, result.Warning, `expected "balanced"`)
	})
}

// fallbackTestConfig is a minimal valid canonical config with multiple
// decisions, for exercising the flat fallback reconstruction that runs when
// the router returns no eval trace.
const fallbackTestConfig = `
version: v0.3
providers:
  models:
    - name: gpt-4
      backend_refs:
        - name: gpt-4-backend
          endpoint: 127.0.0.1:9000
          protocol: http
          type: openai
routing:
  modelCards:
    - name: gpt-4
  signals:
    keywords:
      - name: thinking
        operator: OR
        keywords: ["think", "reason"]
      - name: coding
        operator: OR
        keywords: ["code", "debug"]
  decisions:
    - name: reasoning_decision
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: thinking
      modelRefs:
        - model: gpt-4
    - name: coding_decision
      priority: 90
      rules:
        operator: OR
        conditions:
          - type: keyword
            name: coding
      modelRefs:
        - model: gpt-4
    - name: default_decision
      priority: 10
      modelRefs:
        - model: gpt-4
`

// recipeScopedTestConfig is a minimal canonical config with a named recipe
// bound to an entrypoint, for exercising recipe-scope verification.
const recipeScopedTestConfig = `
version: v0.3
providers:
  models:
    - name: gpt-4
      backend_refs:
        - name: gpt-4-backend
          endpoint: 127.0.0.1:9000
          protocol: http
          type: openai
routing:
  modelCards:
    - name: gpt-4
  decisions:
    - name: default_decision
      modelRefs:
        - model: gpt-4
recipes:
  - name: balanced
    routing:
      decisions:
        - name: balanced_decision
          modelRefs:
            - model: gpt-4
entrypoints:
  - model_names: ["vllm-sr/mom-balanced-v1"]
    recipe: balanced
`

func TestApplyEvalTrace_ReturnsFalseWithoutTrace(t *testing.T) {
	result := &TestQueryResult{}
	ok := applyEvalTrace(result, &RouterEvalResponse{})
	assert.False(t, ok)
	assert.Nil(t, result.EvalTrace)
}
