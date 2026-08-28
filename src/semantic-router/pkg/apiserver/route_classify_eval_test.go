//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type evalDeadlineRecorder struct {
	*httptest.ResponseRecorder
	deadline time.Time
}

func (r *evalDeadlineRecorder) SetWriteDeadline(deadline time.Time) error {
	r.deadline = deadline
	return nil
}

type evalCaptureClassificationService struct {
	lastEvalReq services.IntentRequest
	evalResp    *services.EvalResponse
}

func (s *evalCaptureClassificationService) ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error) {
	return &services.IntentResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error) {
	s.lastEvalReq = req
	if s.evalResp != nil {
		return s.evalResp, nil
	}
	return &services.EvalResponse{OriginalText: "captured"}, nil
}

func (s *evalCaptureClassificationService) DetectPII(req services.PIIRequest) (*services.PIIResponse, error) {
	return &services.PIIResponse{}, nil
}

func (s *evalCaptureClassificationService) CheckSecurity(_ context.Context, req services.SecurityRequest) (*services.SecurityResponse, error) {
	return &services.SecurityResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	return &services.UnifiedBatchResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	return &services.FactCheckResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyUserFeedback(req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error) {
	return &services.UserFeedbackResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyNLI(_ services.NLIRequest) (*services.NLIResponse, error) {
	return nil, fmt.Errorf("NLI not available in eval stub")
}

func (s *evalCaptureClassificationService) IsNLIReady() bool { return false }

func (s *evalCaptureClassificationService) HasUnifiedClassifier() bool      { return true }
func (s *evalCaptureClassificationService) HasClassifier() bool             { return true }
func (s *evalCaptureClassificationService) HasFactCheckClassifier() bool    { return true }
func (s *evalCaptureClassificationService) HasHallucinationDetector() bool  { return true }
func (s *evalCaptureClassificationService) HasHallucinationExplainer() bool { return true }
func (s *evalCaptureClassificationService) HasFeedbackDetector() bool       { return true }
func (s *evalCaptureClassificationService) UpdateConfig(_ *config.RouterConfig) {
}

func (s *evalCaptureClassificationService) RefreshRuntimeConfig(_ *config.RouterConfig) {
}

func TestHandleEvalClassification_AcceptsMessagesArray(t *testing.T) {
	fakeSvc := &evalCaptureClassificationService{
		evalResp: &services.EvalResponse{
			OriginalText: "Still wrong. Explain inflation vs recession in plain English.",
		},
	}
	apiServer := &ClassificationAPIServer{classificationSvc: fakeSvc}

	reqBody := map[string]interface{}{
		"model": "vllm-sr/mom-v1-vault",
		"tools": []map[string]interface{}{
			{"type": "function", "function": map[string]interface{}{"name": "search"}},
		},
		"functions": []map[string]interface{}{
			{"name": "legacy_search", "parameters": map[string]interface{}{"type": "object"}},
		},
		"tool_choice":           map[string]interface{}{"type": "function"},
		"function_call":         map[string]interface{}{"name": "legacy_search"},
		"response_format":       map[string]interface{}{"type": "json_object"},
		"max_tokens":            8192,
		"max_completion_tokens": 4096,
		"metadata":              map[string]string{"cohort": "canary"},
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are a careful tutor."},
			{"role": "user", "content": "Explain inflation vs recession in plain English."},
			{"role": "assistant", "content": "Inflation means prices rise over time."},
			{"role": "user", "content": []map[string]string{
				{"type": "text", "text": "Still wrong."},
				{"type": "text", "text": "Explain inflation vs recession in plain English."},
			}},
		},
	}
	body := mustMarshalEvalRequest(t, reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/eval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleEvalClassification(rr, req)

	requireEvalHTTPStatus(t, rr)
	assertForwardedEvalRequest(t, fakeSvc.lastEvalReq)
	resp := decodeEvalResponse(t, rr)
	if resp.OriginalText != "Still wrong. Explain inflation vs recession in plain English." {
		t.Fatalf("unexpected response payload: %+v", resp)
	}
}

func TestHandleEvalClassification_ExtendsWriteDeadline(t *testing.T) {
	fakeSvc := &evalCaptureClassificationService{}
	apiServer := &ClassificationAPIServer{classificationSvc: fakeSvc}
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/eval",
		strings.NewReader(`{"text":"evaluate every configured signal"}`),
	)
	recorder := &evalDeadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
	started := time.Now()

	apiServer.handleEvalClassification(recorder, req)

	requireEvalHTTPStatus(t, recorder.ResponseRecorder)
	minimumDeadline := started.Add(apiEvalWriteTimeout - time.Second)
	if recorder.deadline.Before(minimumDeadline) {
		t.Fatalf(
			"eval write deadline = %s, want at least %s",
			recorder.deadline,
			minimumDeadline,
		)
	}
}

func mustMarshalEvalRequest(t *testing.T, request any) []byte {
	t.Helper()
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	return body
}

func requireEvalHTTPStatus(t *testing.T, response *httptest.ResponseRecorder) {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, response.Code, response.Body.String())
	}
}

func assertForwardedEvalRequest(t *testing.T, request services.IntentRequest) {
	t.Helper()
	assertForwardedEvalCollections(t, request)
	assertForwardedEvalControls(t, request)
	if request.Model != "vllm-sr/mom-v1-vault" || request.Metadata["cohort"] != "canary" {
		t.Fatalf("expected model and metadata to be forwarded: %#v", request)
	}
}

func assertForwardedEvalCollections(t *testing.T, request services.IntentRequest) {
	t.Helper()
	if len(request.Messages) != 4 || len(request.Tools) != 1 || len(request.Functions) != 1 {
		t.Fatalf("unexpected forwarded request collections: %#v", request)
	}
}

func assertForwardedEvalControls(t *testing.T, request services.IntentRequest) {
	t.Helper()
	if len(request.ToolChoice) == 0 || len(request.FunctionCall) == 0 || len(request.ResponseFormat) == 0 {
		t.Fatalf("expected prompt control fields to be forwarded: %#v", request)
	}
	if string(request.MaxTokens) != "8192" || string(request.MaxCompletionTokens) != "4096" {
		t.Fatalf("expected exact output token controls, got max_tokens=%s max_completion_tokens=%s", request.MaxTokens, request.MaxCompletionTokens)
	}
	if request.Options == nil || !request.Options.EvaluateAllSignals {
		t.Fatalf("expected evaluate_all_signals=true, got %#v", request.Options)
	}
}

func decodeEvalResponse(t *testing.T, response *httptest.ResponseRecorder) services.EvalResponse {
	t.Helper()
	var decoded services.EvalResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return decoded
}

func TestHandleEvalClassification_UsesFullRequestContextWithoutEchoingPrivatePayloads(t *testing.T) {
	const (
		benignCurrentUser = "Summarize the completed lookup in one sentence."
		priorMarker       = "PRIVATE_PRIOR_HISTORY_SENTINEL"
		toolMarker        = "PRIVATE_TOOL_RESULT_SENTINEL"
		argumentMarker    = "PRIVATE_TOOL_ARGUMENT_SENTINEL"
		schemaMarker      = "PRIVATE_TOOL_SCHEMA_SENTINEL"
		formatMarker      = "PRIVATE_RESPONSE_SCHEMA_SENTINEL"
	)

	cfg := contextEvalRouterConfig()
	apiServer := newContextEvalServer(t, cfg)

	core, observedLogs := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(core))
	defer restoreLogger()

	reqBody := privateContextEvalRequest(
		benignCurrentUser,
		priorMarker,
		toolMarker,
		argumentMarker,
		schemaMarker,
		formatMarker,
	)
	body := mustMarshalEvalRequest(t, reqBody)
	if estimate := classification.EstimateOpenAIRequestContext(body); estimate.TokenFloor <= 10_000 {
		t.Fatalf("test request did not cross the 10K context boundary: %+v", estimate)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/eval?trace=true", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	apiServer.handleEvalClassification(rr, req)

	requireEvalHTTPStatus(t, rr)
	resp := decodeEvalResponse(t, rr)
	assertLargeContextEvalResponse(t, resp, benignCurrentUser)

	responseText := rr.Body.String()
	var logText strings.Builder
	for _, entry := range observedLogs.All() {
		fmt.Fprint(&logText, entry.Message, entry.ContextMap())
	}
	assertEvalPrivacyMarkersAbsent(t, responseText, logText.String(), []string{
		priorMarker,
		toolMarker,
		argumentMarker,
		schemaMarker,
		formatMarker,
	})
}

func contextEvalRouterConfig() *config.RouterConfig {
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ContextRules: []config.ContextRule{
				{
					Name:      "small-request-context",
					MinTokens: config.TokenCount("0"),
					MaxTokens: config.TokenCount("10K"),
				},
				{
					Name:      "large-request-context",
					MinTokens: config.TokenCount("10K"),
					MaxTokens: config.TokenCount("128K"),
				},
			}},
			Decisions: []config.Decision{{
				Name:     "large-request-context-route",
				Priority: 10,
				Rules: config.RuleNode{
					Type: config.SignalTypeContext,
					Name: "large-request-context",
				},
			}},
		},
	}
}

func privateContextEvalRequest(
	benignCurrentUser string,
	priorMarker string,
	toolMarker string,
	argumentMarker string,
	schemaMarker string,
	formatMarker string,
) map[string]interface{} {
	return map[string]interface{}{
		"messages": []map[string]interface{}{
			{"role": "user", "content": priorMarker + strings.Repeat("p", 2_048)},
			{
				"role": "assistant",
				"tool_calls": []map[string]interface{}{{
					"id":   "call-private",
					"type": "function",
					"function": map[string]interface{}{
						"name":      "private_lookup",
						"arguments": argumentMarker + strings.Repeat("a", 4_096),
					},
				}},
			},
			{
				"role":         "tool",
				"tool_call_id": "call-private",
				"content":      toolMarker + strings.Repeat("t", 2_048),
			},
			{"role": "user", "content": benignCurrentUser},
		},
		"tools": []map[string]interface{}{{
			"type": "function",
			"function": map[string]interface{}{
				"name":        "private_lookup",
				"description": schemaMarker + strings.Repeat("s", 4_096),
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"key": map[string]string{"type": "string"}},
				},
			},
		}},
		"functions": []map[string]interface{}{{
			"name":        "legacy_private_lookup",
			"description": schemaMarker + strings.Repeat("l", 4_096),
			"parameters":  map[string]interface{}{"type": "object"},
		}},
		"tool_choice":   map[string]interface{}{"type": "function", "function": map[string]string{"name": "private_lookup"}},
		"function_call": map[string]interface{}{"name": "legacy_private_lookup"},
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":        "private_response",
				"description": formatMarker + strings.Repeat("r", 4_096),
				"schema":      map[string]interface{}{"type": "object"},
			},
		},
		"max_tokens":            1_024,
		"max_completion_tokens": 512,
	}
}

func newContextEvalServer(t *testing.T, cfg *config.RouterConfig) *ClassificationAPIServer {
	t.Helper()
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("create classifier: %v", err)
	}
	return &ClassificationAPIServer{classificationSvc: services.NewClassificationService(classifier, cfg)}
}

func assertLargeContextEvalResponse(t *testing.T, response services.EvalResponse, currentUser string) {
	t.Helper()
	if response.DecisionResult == nil || response.DecisionResult.DecisionName != "large-request-context-route" {
		t.Fatalf("expected large context decision, got %#v", response.DecisionResult)
	}
	if response.DecisionResult.MatchedSignals == nil ||
		!containsString(response.DecisionResult.MatchedSignals.Context, "large-request-context") {
		t.Fatalf("expected matched large context signal, got %#v", response.DecisionResult.MatchedSignals)
	}
	if len(response.EvalTrace) == 0 || response.OriginalText != currentUser {
		t.Fatalf("expected trace and only current user text, got %+v", response)
	}
}

func assertEvalPrivacyMarkersAbsent(t *testing.T, responseText, logText string, markers []string) {
	t.Helper()
	for _, marker := range markers {
		if strings.Contains(responseText, marker) {
			t.Fatalf("response exposed private request payload %q", marker)
		}
		if strings.Contains(logText, marker) {
			t.Fatalf("logs exposed private request payload %q", marker)
		}
	}
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
