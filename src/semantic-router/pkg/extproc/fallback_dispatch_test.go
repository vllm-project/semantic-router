package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func setupFallbackTestRouter(t *testing.T, fallbackPolicy fallback.FallbackPolicy) (*OpenAIRouter, *config.RouterConfig) {
	t.Helper()
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "model-primary",
			ModelConfig: map[string]config.ModelParams{
				"model-primary": {
					PreferredEndpoints: []string{"backend-primary"},
					APIFormat:          "openai.chat.v1",
				},
				"model-fallback-1": {
					PreferredEndpoints: []string{"backend-fallback-1"},
					APIFormat:          "openai.chat.v1",
				},
				"model-fallback-2": {
					PreferredEndpoints: []string{"backend-fallback-2"},
					APIFormat:          "anthropic.messages.v1",
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
				{Name: "backend-fallback-1", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-openai"},
				{Name: "backend-fallback-2", Address: "127.0.0.1", Port: 8003, ProviderProfileName: "prof-anthropic"},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"prof-openai":    {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
				"prof-anthropic": {Type: "anthropic", BaseURL: "http://127.0.0.1:8003"},
			},
		},
	}

	circuitBreaker := fallback.NewBackendCircuitBreaker(fallbackPolicy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(fallbackPolicy, circuitBreaker)

	router := &OpenAIRouter{
		Config:               cfg,
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: orchestrator,
	}

	return router, cfg
}

func testFallbackRequestContext(primaryModel string, eligibleModels []string) *RequestContext {
	req := testNeutralRequest(primaryModel, "Explain quantum computing in one sentence")
	modelRefs := make([]config.ModelRef, len(eligibleModels))
	for i, m := range eligibleModels {
		modelRefs[i] = config.ModelRef{Model: m}
	}
	return &RequestContext{
		RequestID:               "req_fallback_test_001",
		RequestModel:            primaryModel,
		VSRSelectedModel:        primaryModel,
		VSRSelectedDecisionName: "chat_fallback_decision",
		VSREligibleModelRefs:    modelRefs,
		SourceFormat:            llmprotocol.OpenAIChatV1,
		TargetFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         req,
		ProcessingStartTime:     time.Now().Add(-100 * time.Millisecond),
		StartTime:               time.Now().Add(-100 * time.Millisecond),
	}
}

func TestFallbackSuccessOn503(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-01",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Quantum computing harnesses superposition and entanglement to solve complex problems."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 14,
			"total_tokens": 26
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model != "model-fallback-1" {
			return nil, 500, fmt.Errorf("unexpected fallback model: %s", model)
		}
		return candidateRespJSON, http.StatusOK, nil
	}

	upstreamErrBody := []byte(`{"error":{"message":"The server is temporarily unavailable","type":"server_error"}}`)
	t.Logf("fallback before: eligible=%t, policy=%+v", router.shouldAttemptFallback(ctx), router.FallbackOrchestrator.Policy())
	resp := router.handleUpstreamTransportError(upstreamErrBody, ctx)
	t.Logf("fallback after: record=%+v", ctx.FallbackRecord)
	if resp == nil {
		t.Fatal("expected non-nil response from handleUpstreamTransportError")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 (OK), got status code: %v", imm.GetStatus().GetCode())
	}

	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}

	if got := headersMap[headers.VSRSelectedModel]; got != "model-fallback-1" {
		t.Errorf("header %s = %q, want %q", headers.VSRSelectedModel, got, "model-fallback-1")
	}

	if got := headersMap[headers.VSRFallbackAttempts]; got != "2" {
		t.Errorf("header %s = %q, want %q", headers.VSRFallbackAttempts, got, "2")
	}

	if got := headersMap[headers.VSRResponsePath]; got != headers.ResponsePathFallback {
		t.Errorf("header %s = %q, want %q", headers.VSRResponsePath, got, headers.ResponsePathFallback)
	}

	if got := headersMap["content-type"]; got != "application/json" {
		t.Errorf("header content-type = %q, want application/json", got)
	}

	if ctx.FallbackRecord == nil {
		t.Fatal("ctx.FallbackRecord is nil")
	}
	if len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("expected 2 attempts in record, got %d", len(ctx.FallbackRecord.Attempts))
	}
	if ctx.FallbackRecord.FinalStatus != "succeeded" {
		t.Fatalf("expected final status 'succeeded', got %q", ctx.FallbackRecord.FinalStatus)
	}
	if ctx.FallbackRecord.Attempts[0].StatusCode != 503 || !ctx.FallbackRecord.Attempts[0].Discarded {
		t.Errorf("attempt 0 should be discarded 503, got: %+v", ctx.FallbackRecord.Attempts[0])
	}
	if ctx.FallbackRecord.Attempts[1].StatusCode != 200 || ctx.FallbackRecord.Attempts[1].Discarded {
		t.Errorf("attempt 1 should be successful 200, got: %+v", ctx.FallbackRecord.Attempts[1])
	}
}

func TestFallbackCrossProtocolOpenAIToAnthropic(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-2"})
	ctx.UpstreamStatusCode = 502

	anthropicRespJSON := []byte(`{
		"id": "msg_anthropic_fallback_01",
		"type": "message",
		"role": "assistant",
		"content": [
			{"type": "text", "text": "Anthropic candidate: quantum computing uses qubits to compute in parallel."}
		],
		"model": "claude-3-5",
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 15,
			"output_tokens": 12
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, hdrs map[string]string) ([]byte, int, error) {
		if model != "model-fallback-2" {
			return nil, 500, fmt.Errorf("unexpected fallback model: %s", model)
		}
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			return nil, 400, fmt.Errorf("body sent to Anthropic backend is not valid JSON: %w", err)
		}
		if _, ok := parsed["messages"]; !ok {
			return nil, 400, fmt.Errorf("anthropic wire body missing messages field: %s", string(body))
		}
		return anthropicRespJSON, http.StatusOK, nil
	}

	upstreamErrBody := []byte(`{"error":{"message":"Bad gateway from primary provider"}}`)
	resp := router.handleUpstreamTransportError(upstreamErrBody, ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse from cross-protocol fallback, got: %#v", resp)
	}

	var clientBody struct {
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(imm.GetBody(), &clientBody); err != nil {
		t.Fatalf("immediate response body is not valid OpenAI chat response: %v\nBody: %s", err, string(imm.GetBody()))
	}

	if len(clientBody.Choices) == 0 {
		t.Fatal("expected choices in translated OpenAI response")
	}
	if clientBody.Choices[0].Message.Content != "Anthropic candidate: quantum computing uses qubits to compute in parallel." {
		t.Errorf("unexpected content: %q", clientBody.Choices[0].Message.Content)
	}
	if clientBody.Usage.PromptTokens != 15 || clientBody.Usage.CompletionTokens != 12 {
		t.Errorf("usage not translated correctly: prompt=%d completion=%d", clientBody.Usage.PromptTokens, clientBody.Usage.CompletionTokens)
	}
}

func TestFallbackSafetyGateStreamingCommitted(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.IsStreamingResponse = true
	ctx.TTFTRecorded = true // Tokens already emitted to downstream client

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when streaming response is already committed")
	}
	if resp == nil {
		t.Fatal("expected response")
	}
	if resp.GetImmediateResponse() != nil {
		t.Fatal("must not return ImmediateResponse when streaming already committed")
	}
}

func TestFallbackSafetyGateNonIdempotentTools(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	// Add tool result message representing non-idempotent executed side-effect
	ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, llmprotocol.Message{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			Text: `{"transaction_id": "tx_123", "status": "debited"}`,
		}},
	})

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when non-idempotent tool side-effects exist")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatalf("expected standard error translation, got: %#v", resp)
	}
}

func TestFallbackSafetyGateBodyNotReplayable(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.SemanticRequest = nil // Request body not buffered/replayable

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when semantic request body is not replayable")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatalf("expected standard error translation, got: %#v", resp)
	}
}

func TestFallbackExhaustion(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		// Fallback candidate also fails
		return []byte(`{"error":"fallback candidate service unavailable"}`), 503, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response on candidate exhaustion")
	}
	// When candidates fail, falls back to original transport error translation
	if resp.GetImmediateResponse() != nil {
		t.Fatal("must not return ImmediateResponse when all fallback candidates failed")
	}
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord to be created")
	}
	if len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("expected 2 attempts recorded, got %d", len(ctx.FallbackRecord.Attempts))
	}
	if ctx.FallbackRecord.FinalStatus != "candidates_exhausted" {
		t.Fatalf("expected final status 'candidates_exhausted', got %q", ctx.FallbackRecord.FinalStatus)
	}
}

func TestFallbackDisabledPolicy(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.Enabled = false
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when fallback is disabled")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatal("expected standard error translation")
	}
}

func TestFallbackCircuitBreakerSkipsOpenBackend(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2"})
	ctx.UpstreamStatusCode = 503

	// Trip circuit breaker on backend-fallback-1
	backend1 := "backend-fallback-1"
	for i := 0; i < 3; i++ {
		router.FallbackOrchestrator.CircuitBreaker().RecordFailure(backend1)
	}
	if router.FallbackOrchestrator.CircuitBreaker().Allow(backend1) {
		t.Fatalf("expected backend %s to be circuit-broken", backend1)
	}

	var attemptedModels []string
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		attemptedModels = append(attemptedModels, model)
		if model == "model-fallback-2" {
			return []byte(`{
				"id": "msg_02",
				"type": "message",
				"role": "assistant",
				"content": [{"type": "text", "text": "candidate 2 response"}],
				"model": "claude-3-5",
				"stop_reason": "end_turn",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`), 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from candidate 2, got: %#v", resp)
	}

	if len(attemptedModels) != 1 || attemptedModels[0] != "model-fallback-2" {
		t.Fatalf("expected model-fallback-1 to be skipped due to open circuit breaker, got attempted: %v", attemptedModels)
	}
}

func TestFallbackRealHTTPConnector(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-live-http-01",
			"object": "chat.completion",
			"created": 1700000001,
			"model": "model-fallback-1",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Live HTTP fallback connector succeeded!"
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 8,
				"completion_tokens": 7,
				"total_tokens": 15
			}
		}`))
	}))
	defer server.Close()

	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	host, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}

	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Point backend-fallback-1 to test server
	cfg.VLLMEndpoints[1].Address = host
	cfg.VLLMEndpoints[1].Port = port
	cfg.ProviderProfiles["prof-openai"] = config.ProviderProfile{
		Type:    "openai",
		BaseURL: server.URL,
	}

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 504

	// Leave fallbackCaller nil to test real dispatchFallbackHTTP connector path
	resp := router.handleUpstreamTransportError([]byte("upstream 504 timeout"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response from handleUpstreamTransportError")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse from live connector fallback, got: %#v", resp)
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(imm.GetBody(), &parsed); err != nil {
		t.Fatalf("failed to unmarshal body: %v\nBody: %s", err, string(imm.GetBody()))
	}
	if len(parsed.Choices) == 0 || parsed.Choices[0].Message.Content != "Live HTTP fallback connector succeeded!" {
		t.Fatalf("unexpected content from live HTTP fallback: %#v", parsed)
	}
}

func TestFallbackReplayAuditing(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	memStore := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(memStore)
	t.Cleanup(func() { _ = recorder.Close() })

	recordID := "replay_test_fallback_123"
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: recordID, RequestID: "req_123"}); err != nil {
		t.Fatal(err)
	}

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.RouterReplayID = recordID
	ctx.RouterReplayRecorder = recorder
	// Simulate primary 503 status recorded in processor_res_header.go:30 prior to fallback
	router.updateRouterReplayStatus(ctx, 503, false)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		return []byte(`{
			"id": "chatcmpl-01",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "model-fallback-1",
			"choices": [{"index":0,"message":{"role":"assistant","content":"audit test"},"finish_reason":"stop"}]
		}`), 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse, got: %#v", resp)
	}

	rec, ok := recorder.GetRecord(recordID)
	if !ok {
		t.Fatal("replay record not found in recorder")
	}

	if rec.ResponseStatus != http.StatusOK {
		t.Errorf("expected replay response status %d, got %d", http.StatusOK, rec.ResponseStatus)
	}

	if len(rec.Outcomes) == 0 {
		t.Fatal("expected at least 1 outcome recorded in replay")
	}

	lastOutcome := rec.Outcomes[len(rec.Outcomes)-1]
	if lastOutcome.Source != "fallback" || lastOutcome.Verdict != "completed" {
		t.Errorf("outcome mismatch: %+v", lastOutcome)
	}
	if lastOutcome.Metadata["fallback_model"] != "model-fallback-1" {
		t.Errorf("expected fallback_model 'model-fallback-1', got %q", lastOutcome.Metadata["fallback_model"])
	}
	if lastOutcome.Metadata["final_status"] != "succeeded" {
		t.Errorf("expected persisted final_status succeeded, got %q", lastOutcome.Metadata["final_status"])
	}
	var execution fallback.ExecutionRecord
	if err := json.Unmarshal([]byte(lastOutcome.Metadata["execution_record"]), &execution); err != nil {
		t.Fatalf("decode persisted fallback execution: %v", err)
	}
	if execution.RequestID != ctx.RequestID || len(execution.Attempts) != 2 {
		t.Fatalf("persisted execution mismatch: %+v", execution)
	}
	if execution.Attempts[0].AttemptID == "" || execution.Attempts[1].AttemptID == "" {
		t.Fatalf("persisted attempts are not correlated: %+v", execution.Attempts)
	}
}

func TestFallbackTotalTimeoutClampsAttemptDeadline(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.TotalTimeout = 30 * time.Second
	policy.PerAttemptTimeout = 10 * time.Second
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	// Simulate primary taking 29s of the 30s budget
	ctx.ProcessingStartTime = time.Now().Add(-29 * time.Second)

	var receivedAttemptDeadline time.Time
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		deadline, ok := callCtx.Deadline()
		if !ok {
			t.Fatal("expected attempt context to carry a deadline")
		}
		receivedAttemptDeadline = deadline
		return []byte(`{
			"id": "chatcmpl-01",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "model-fallback-1",
			"choices": [{"index":0,"message":{"role":"assistant","content":"clamped deadline test"},"finish_reason":"stop"}]
		}`), 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse, got: %#v", resp)
	}

	remainingBudget := time.Until(receivedAttemptDeadline)
	// The remaining budget was ~1s (30s - 29s). It must be clamped to ~1s, NOT 10s!
	if remainingBudget > 2*time.Second {
		t.Fatalf("attempt context granted %v timeout, should have been clamped to ~1s remaining total budget", remainingBudget)
	}
}

func TestFallbackRejectsWhenTotalBudgetExhausted(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.TotalTimeout = 30 * time.Second
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	// Simulate primary taking 31s of the 30s budget
	ctx.ProcessingStartTime = time.Now().Add(-31 * time.Second)

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when total budget has already expired")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatalf("expected standard error translation, got: %#v", resp)
	}
	if ctx.FallbackRecord == nil || len(ctx.FallbackRecord.Attempts) != 1 {
		t.Fatalf("expected expired budget to retain the primary attempt, got: %+v", ctx.FallbackRecord)
	}
	if ctx.FallbackRecord.FinalStatus != "total_deadline_exceeded" {
		t.Fatalf("expected total_deadline_exceeded, got %q", ctx.FallbackRecord.FinalStatus)
	}
}

func TestFallbackMaxAttemptsExceededAcrossMultipleCandidates(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.MaxAttempts = 2 // 1 primary attempt + 1 fallback attempt max
	router, cfg := setupFallbackTestRouter(t, policy)
	// Add model-fallback-3 to create 4 candidates total
	cfg.ModelConfig["model-fallback-3"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-fallback-3"},
		APIFormat:          "openai.chat.v1",
	}
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints, config.VLLMEndpoint{
		Name: "backend-fallback-3", Address: "127.0.0.1", Port: 8004, ProviderProfileName: "prof-openai",
	})

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2", "model-fallback-3"})
	ctx.UpstreamStatusCode = 503

	var attemptedModels []string
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		attemptedModels = append(attemptedModels, model)
		// Candidate fails with retryable 503
		return []byte(`{"error": "candidate unavailable"}`), 503, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected response")
	}
	if resp.GetImmediateResponse() != nil {
		t.Fatal("expected standard error translation after max attempts reached")
	}

	// With MaxAttempts=2, primary is Attempt 1, model-fallback-1 is Attempt 2.
	// model-fallback-2 and model-fallback-3 must NEVER be invoked!
	if len(attemptedModels) != 1 {
		t.Fatalf("expected exactly 1 fallback attempt before hitting MaxAttempts=2, got %d (%v)", len(attemptedModels), attemptedModels)
	}
	if attemptedModels[0] != "model-fallback-1" {
		t.Fatalf("expected attempted model to be model-fallback-1, got %s", attemptedModels[0])
	}
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord")
	}
	if len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("expected 2 total attempts recorded, got %d", len(ctx.FallbackRecord.Attempts))
	}
	if ctx.FallbackRecord.FinalStatus != "max_attempts_exceeded" {
		t.Fatalf("expected final status 'max_attempts_exceeded', got %q", ctx.FallbackRecord.FinalStatus)
	}
}

func TestFallbackClient4xxWithConnectionSubstringDoesNotTriggerFallback(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 400

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	// Upstream returns 400 with "connection reset" text inside body JSON
	errBody := []byte(`{"error": {"message": "Invalid client request: connection reset parameter invalid"}}`)
	resp := router.handleUpstreamTransportError(errBody, ctx)

	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked on client 400 Bad Request, even if body contains 'connection reset'")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatal("expected standard error translation for 400")
	}
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord")
	}
	if ctx.FallbackRecord.FinalStatus != "non_retryable" {
		t.Fatalf("expected final status 'non_retryable', got %q", ctx.FallbackRecord.FinalStatus)
	}
	// Verify primary backend circuit breaker was NOT tripped
	if !router.FallbackOrchestrator.CircuitBreaker().Allow("backend-primary") {
		t.Fatal("backend-primary circuit breaker must NOT trip on client 400 error")
	}
}

func TestFallbackRequestDefensiveCopyIsolation(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2"})
	ctx.UpstreamStatusCode = 503

	origMsgCount := len(ctx.SemanticRequest.Messages)
	origPrompt := ctx.SemanticRequest.Messages[0].Content[0].Text

	var attemptedModels []string
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		attemptedModels = append(attemptedModels, model)
		if model == "model-fallback-1" {
			return []byte(`{"error": "failed"}`), 503, nil
		}
		return []byte(`{
			"id": "msg_anthropic_fallback_02",
			"type": "message",
			"role": "assistant",
			"content": [{"type": "text", "text": "fallback 2 ok"}],
			"model": "claude-3-5",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`), 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback 2, got: %#v", resp)
	}

	// Verify original RequestContext.SemanticRequest was not mutated
	if len(ctx.SemanticRequest.Messages) != origMsgCount {
		t.Errorf("original messages slice modified: had %d, now %d", origMsgCount, len(ctx.SemanticRequest.Messages))
	}
	if ctx.SemanticRequest.Messages[0].Content[0].Text != origPrompt {
		t.Errorf("original prompt modified: had %q, now %q", origPrompt, ctx.SemanticRequest.Messages[0].Content[0].Text)
	}
}

func TestFallbackTriggeredInHandleResponseHeaders(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-hdr",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Candidate response triggered before error headers forwarded"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 8,
			"total_tokens": 18
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model != "model-fallback-1" {
			return nil, 500, fmt.Errorf("unexpected fallback model: %s", model)
		}
		return candidateRespJSON, http.StatusOK, nil
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	resp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response from handleResponseHeaders")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse from fallback before headers continued, got: %#v", resp)
	}
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 (OK), got status code: %v", imm.GetStatus().GetCode())
	}

	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}
	if got := headersMap[headers.VSRSelectedModel]; got != "model-fallback-1" {
		t.Errorf("header %s = %q, want %q", headers.VSRSelectedModel, got, "model-fallback-1")
	}
	if got := headersMap[headers.VSRFallbackAttempts]; got != "2" {
		t.Errorf("header %s = %q, want %q", headers.VSRFallbackAttempts, got, "2")
	}
	if got := headersMap[headers.VSRResponsePath]; got != headers.ResponsePathFallback {
		t.Errorf("header %s = %q, want %q", headers.VSRResponsePath, got, headers.ResponsePathFallback)
	}

	// Verify headers were not continued downstream since immediate response terminated the chain
	if ctx.ResponseHeadersContinued {
		t.Fatal("ctx.ResponseHeadersContinued must remain false when ImmediateResponse is returned")
	}
}

func TestFallbackFailureInHandleResponseHeadersMarksContinued(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})

	// Fallback candidate also fails
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		return []byte(`{"error": "candidate also 503"}`), 503, nil
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	resp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if resp.GetImmediateResponse() != nil {
		t.Fatal("expected continue response when all fallback candidates fail, got ImmediateResponse")
	}

	// Once continue response is returned from handleResponseHeaders, headers are committed downstream
	if !ctx.ResponseHeadersContinued {
		t.Fatal("ctx.ResponseHeadersContinued must be true after returning continue response")
	}

	// Subsequent response-body handler must strictly NOT attempt fallback or emit ImmediateResponse
	bodyResp := router.handleUpstreamTransportError([]byte("upstream 503 body"), ctx)
	if bodyResp == nil {
		t.Fatal("expected non-nil response from handleUpstreamTransportError")
	}
	if bodyResp.GetImmediateResponse() != nil {
		t.Fatal("expected standard error translation after headers continued, got ImmediateResponse")
	}
}

func TestFallbackStrictlyRejectedWhenResponseHeadersAlreadyContinued(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.ResponseHeadersContinued = true // headers were already forwarded downstream

	fallbackInvoked := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackInvoked = true
		return nil, 200, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if fallbackInvoked {
		t.Fatal("fallbackCaller must NOT be invoked when response headers were already continued")
	}
	if resp == nil || resp.GetImmediateResponse() != nil {
		t.Fatal("expected standard error translation when response headers are already continued, got ImmediateResponse")
	}
}

func TestFallbackPreservesStreamingContractOpenAI(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-stream-01",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Streaming fallback candidate successfully preserved SSE contract."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 10,
			"total_tokens": 22
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model != "model-fallback-1" {
			return nil, 500, fmt.Errorf("unexpected model: %s", model)
		}
		return candidateRespJSON, http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse, got: %#v", resp)
	}

	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got status: %v", imm.GetStatus().GetCode())
	}

	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}

	// Must be text/event-stream, NOT application/json!
	if got := headersMap["content-type"]; got != "text/event-stream" {
		t.Fatalf("expected content-type text/event-stream, got: %q", got)
	}

	if got := headersMap[headers.VSRSelectedModel]; got != "model-fallback-1" {
		t.Errorf("expected model %q, got %q", "model-fallback-1", got)
	}
	if got := headersMap[headers.VSRFallbackAttempts]; got != "2" {
		t.Errorf("expected 2 attempts, got %q", got)
	}

	// Verify the body decodes as a valid client SSE stream
	engine := protocolcodec.NewBuiltinEngine()
	assertClientStreamDecodes(t, engine, ctx.SourceFormat, imm.GetBody())
}

func TestFallbackPreservesStreamingContractCrossProtocolAnthropic(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	// Client requested Anthropic format with streaming
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.UpstreamStatusCode = 502
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true

	// Candidate is OpenAI format backend
	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-anthropic-stream",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Cross-protocol Anthropic stream preserved."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 8,
			"completion_tokens": 6,
			"total_tokens": 14
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		return candidateRespJSON, http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 502"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse, got: %#v", resp)
	}

	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}

	if got := headersMap["content-type"]; got != "text/event-stream" {
		t.Fatalf("expected content-type text/event-stream, got: %q", got)
	}

	engine := protocolcodec.NewBuiltinEngine()
	assertClientStreamDecodes(t, engine, ctx.SourceFormat, imm.GetBody())
}

func TestFallbackRecipeScopedOverride(t *testing.T) {
	// Global policy is disabled by default
	globalPolicy := fallback.DefaultPolicy() // Enabled: false
	router, _ := setupFallbackTestRouter(t, globalPolicy)

	// Recipe policy has fallback explicitly enabled
	recipePolicy := fallback.DefaultEnabledPolicy()
	recipeCB := fallback.NewBackendCircuitBreaker(recipePolicy.CircuitBreaker)
	recipeOrch := fallback.NewOrchestrator(recipePolicy, recipeCB)
	router.RecipeFallbackOrchestrators = map[config.RecipeName]*fallback.Orchestrator{
		"recipe-fallback-enabled": recipeOrch,
	}

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-recipe",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Recipe override fallback succeeded."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 5,
			"completion_tokens": 5,
			"total_tokens": 10
		}
	}`)

	fallbackCalls := 0
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackCalls++
		return candidateRespJSON, http.StatusOK, nil
	}

	// Case 1: Request with default/no recipe should use global fallback (disabled) -> NO fallback
	ctxNoRecipe := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctxNoRecipe.UpstreamStatusCode = 503
	respNoRecipe := router.handleUpstreamTransportError([]byte("upstream 503"), ctxNoRecipe)
	if fallbackCalls != 0 {
		t.Fatalf("expected 0 fallback calls for unconfigured recipe, got %d", fallbackCalls)
	}
	if respNoRecipe.GetImmediateResponse() != nil {
		t.Fatalf("expected no ImmediateResponse for unconfigured recipe")
	}

	// Case 2: Request with "recipe-fallback-enabled" -> fallback IS triggered
	ctxRecipe := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctxRecipe.Routing.SelectRecipe(&config.RoutingRecipe{Name: "recipe-fallback-enabled"})
	ctxRecipe.UpstreamStatusCode = 503

	respRecipe := router.handleUpstreamTransportError([]byte("upstream 503"), ctxRecipe)
	if fallbackCalls != 1 {
		t.Fatalf("expected 1 fallback call for recipe with fallback enabled, got %d", fallbackCalls)
	}
	imm := respRecipe.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse for recipe-scoped fallback")
	}
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got %v", imm.GetStatus().GetCode())
	}
}

func TestFallbackCandidate200WithErrorRecordedAsFailure(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.MaxAttempts = 3
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2"})
	ctx.UpstreamStatusCode = 503

	model1Called := false
	model2Called := false

	candidate2RespJSON := []byte(`{
		"id": "msg_fallback_02",
		"type": "message",
		"role": "assistant",
		"model": "model-fallback-2",
		"content": [{"type": "text", "text": "Fallback 2 succeeded."}],
		"usage": {"input_tokens": 10, "output_tokens": 10}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			model1Called = true
			// Returns 200 with an error (e.g. read body error / connection reset)
			return nil, http.StatusOK, errors.New("unexpected EOF reading response body")
		}
		if model == "model-fallback-2" {
			model2Called = true
			return candidate2RespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatal("expected ImmediateResponse from candidate 2 fallback")
	}

	if !model1Called {
		t.Error("expected model-fallback-1 to be attempted first")
	}
	if !model2Called {
		t.Error("expected model-fallback-2 to be attempted after model-fallback-1 returned 200 with error")
	}

	// Verify FallbackRecord: Attempt 1 is primary (503), Attempt 2 is fallback 1 (200 with error, discarded), Attempt 3 is fallback 2 (succeeded)
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord to be present")
	}
	if len(ctx.FallbackRecord.Attempts) != 3 {
		t.Fatalf("expected 3 total attempts recorded, got %d", len(ctx.FallbackRecord.Attempts))
	}
	if !ctx.FallbackRecord.Attempts[1].Discarded {
		t.Errorf("expected attempt 2 (200 with error) to be marked Discarded")
	}
	if ctx.FallbackRecord.Attempts[2].Discarded {
		t.Errorf("expected attempt 3 (successful) to NOT be discarded")
	}
	if ctx.FallbackRecord.FinalStatus != "succeeded" {
		t.Errorf("expected FinalStatus succeeded, got %s", ctx.FallbackRecord.FinalStatus)
	}
	if ctx.FallbackRecord.SelectedModel != "model-fallback-2" {
		t.Errorf("expected SelectedModel model-fallback-2, got %s", ctx.FallbackRecord.SelectedModel)
	}
}

func TestFallbackCandidateStreamEncodingFailureEvaluatesAttemptOnce(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2"})
	ctx.UpstreamStatusCode = 503
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true

	// Candidate 1 returns multiple choices/alternatives, which causes EncodeResponseStream to fail
	// because streaming multiple alternatives is unsupported.
	multiChoiceRespJSON := []byte(`{
		"id": "chatcmpl-fallback-multi",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-fallback-1",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": "choice 0"}, "finish_reason": "stop"},
			{"index": 1, "message": {"role": "assistant", "content": "choice 1"}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
	}`)

	singleChoiceRespJSON := []byte(`{
		"id": "msg_fallback_02",
		"type": "message",
		"role": "assistant",
		"model": "model-fallback-2",
		"content": [{"type": "text", "text": "candidate 2 success"}],
		"usage": {"input_tokens": 5, "output_tokens": 5}
	}`)

	var model1Called, model2Called bool
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			model1Called = true
			return multiChoiceRespJSON, http.StatusOK, nil
		}
		if model == "model-fallback-2" {
			model2Called = true
			return singleChoiceRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response (translated transport error)")
	}
	// Fallback did not succeed, so no ImmediateResponse should be returned (original error body is translated)
	if resp.GetImmediateResponse() != nil {
		t.Fatalf("expected nil ImmediateResponse when fallback candidate fails stream encoding, got: %#v", resp)
	}

	if !model1Called {
		t.Error("expected model-fallback-1 to be attempted")
	}
	if model2Called {
		t.Error("expected model-fallback-2 to NOT be attempted after non-retryable stream encoding failure")
	}

	// Verify FallbackRecord:
	// Attempt 0: primary (503)
	// Attempt 1: fallback 1 (stream encoding failed, evaluated EXACTLY ONCE as Discarded)
	// Crucially: exactly 2 attempts, NOT 3 (verifies no duplicate evaluation before/after stream encoding)
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord to be present")
	}
	if len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("expected exactly 2 total attempts recorded (no duplicate evaluations), got %d", len(ctx.FallbackRecord.Attempts))
	}
	if !ctx.FallbackRecord.Attempts[1].Discarded {
		t.Errorf("expected attempt 1 (stream encoding failure) to be marked Discarded")
	}
	if ctx.FallbackRecord.Attempts[1].Error == nil {
		t.Errorf("expected attempt 1 to have non-nil Error")
	}
	if ctx.FallbackRecord.FinalStatus != "non_retryable" {
		t.Errorf("expected FinalStatus non_retryable, got %s", ctx.FallbackRecord.FinalStatus)
	}

	// Verify primary context was preserved: status remains primary 503 and selected model is unchanged.
	if ctx.UpstreamStatusCode != 503 {
		t.Errorf("expected UpstreamStatusCode to remain primary 503 on candidate stream failure, got %d", ctx.UpstreamStatusCode)
	}
	if ctx.RequestModel != "model-primary" {
		t.Errorf("expected RequestModel to remain 'model-primary', got '%s'", ctx.RequestModel)
	}
	if ctx.VSRSelectedModel != "model-primary" {
		t.Errorf("expected VSRSelectedModel to remain 'model-primary', got '%s'", ctx.VSRSelectedModel)
	}
	if ctx.ResponsePath == headers.ResponsePathFallback {
		t.Errorf("expected ResponsePath to not be marked fallback on failure, got '%s'", ctx.ResponsePath)
	}
}

func TestFallbackCandidateStreamOversizedFrameRejected(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true

	// Candidate 1 returns a valid buffered response containing 1 MiB of text.
	// Since BodyBytes is 64 MiB, translation succeeds. However, when encoded into a single
	// SSE frame, the frame exceeds the configured 1 MiB SSEFrameBytes limit.
	oversizedText := strings.Repeat("a", 1<<20)
	oversizedRespJSON := []byte(fmt.Sprintf(`{
		"id": "chatcmpl-fallback-oversized",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-fallback-1",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": %q}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 10, "completion_tokens": 1000, "total_tokens": 1010}
	}`, oversizedText))

	var model1Called bool
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			model1Called = true
			return oversizedRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response (translated transport error)")
	}
	// Fallback did not succeed because the encoded frame exceeded SSE limits, so no ImmediateResponse.
	if resp.GetImmediateResponse() != nil {
		t.Fatalf("expected nil ImmediateResponse when fallback candidate produces oversized SSE frame, got: %#v", resp)
	}

	if !model1Called {
		t.Error("expected model-fallback-1 to be attempted")
	}

	// Verify FallbackRecord:
	// Attempt 0: primary (503)
	// Attempt 1: fallback 1 (oversized frame validation failed, marked Discarded)
	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord to be present")
	}
	if len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("expected exactly 2 total attempts recorded, got %d", len(ctx.FallbackRecord.Attempts))
	}
	if !ctx.FallbackRecord.Attempts[1].Discarded {
		t.Errorf("expected attempt 1 to be marked Discarded")
	}
	if ctx.FallbackRecord.Attempts[1].Error == nil {
		t.Errorf("expected attempt 1 to have non-nil Error")
	}
	if ctx.FallbackRecord.FinalStatus != "non_retryable" {
		t.Errorf("expected FinalStatus non_retryable, got %s", ctx.FallbackRecord.FinalStatus)
	}

	// Verify primary context effects were NOT committed:
	if ctx.UpstreamStatusCode != 503 {
		t.Errorf("expected UpstreamStatusCode to remain primary 503, got %d", ctx.UpstreamStatusCode)
	}
	if ctx.RequestModel != "model-primary" {
		t.Errorf("expected RequestModel to remain 'model-primary', got '%s'", ctx.RequestModel)
	}
	if ctx.VSRSelectedModel != "model-primary" {
		t.Errorf("expected VSRSelectedModel to remain 'model-primary', got '%s'", ctx.VSRSelectedModel)
	}
	if ctx.ResponsePath == headers.ResponsePathFallback {
		t.Errorf("expected ResponsePath to not be marked fallback on frame limit failure, got '%s'", ctx.ResponsePath)
	}
	if ctx.RequestCostPriced {
		t.Error("expected RequestCostPriced to remain false when candidate wire validation fails")
	}
}

func TestFallbackSuccessSettlesUsageAndCost(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Configure candidate pricing: $2/1M prompt, $10/1M completion
	modelConfig := cfg.ModelConfig["model-fallback-1"]
	modelConfig.Pricing = config.ModelPricing{
		PromptPer1M:     2.0,
		CompletionPer1M: 10.0,
		Currency:        "USD",
	}
	cfg.ModelConfig["model-fallback-1"] = modelConfig

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-priced",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Quantum computing harnesses superposition and entanglement to solve complex problems."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 14,
			"total_tokens": 26
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	immediateResp := resp.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("expected ImmediateResponse for successful fallback")
	}
	if immediateResp.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK, got: %v", immediateResp.GetStatus().GetCode())
	}

	// 1. Verify terminal usage settlement updated RequestCostPriced, RequestCost, and Currency
	if !ctx.RequestCostPriced {
		t.Errorf("expected RequestCostPriced to be true after successful fallback settlement")
	}
	// Expected cost: (12 * 2.0 / 1e6) + (14 * 10.0 / 1e6) = 0.000024 + 0.000140 = 0.000164
	expectedCost := 0.000164
	if math.Abs(ctx.RequestCost-expectedCost) > 1e-9 {
		t.Errorf("expected RequestCost %f, got %f", expectedCost, ctx.RequestCost)
	}
	if ctx.RequestCostCurrency != "USD" {
		t.Errorf("expected RequestCostCurrency 'USD', got '%s'", ctx.RequestCostCurrency)
	}
	servedAttempt := ctx.FallbackRecord.Attempts[len(ctx.FallbackRecord.Attempts)-1]
	if math.Abs(servedAttempt.Cost-expectedCost) > 1e-9 || servedAttempt.Currency != "USD" {
		t.Errorf("fallback attempt cost was not correlated: %+v", servedAttempt)
	}

	// 2. Verify ImmediateResponse contains x-vsr-cost and x-vsr-cost-currency headers
	var costHeader, currencyHeader string
	for _, opt := range immediateResp.GetHeaders().GetSetHeaders() {
		if opt.GetHeader().GetKey() == headers.VSRCost {
			costHeader = string(opt.GetHeader().GetRawValue())
		}
		if opt.GetHeader().GetKey() == headers.VSRCostCurrency {
			currencyHeader = string(opt.GetHeader().GetRawValue())
		}
	}
	if costHeader == "" {
		t.Errorf("expected %s header to be present in ImmediateResponse", headers.VSRCost)
	} else if costHeader != "0.000164" {
		t.Errorf("expected %s header '0.000164', got '%s'", headers.VSRCost, costHeader)
	}
	if currencyHeader != "USD" {
		t.Errorf("expected %s header 'USD', got '%s'", headers.VSRCostCurrency, currencyHeader)
	}

	// 3. Verify selected model and upstream status code
	if ctx.RequestModel != "model-fallback-1" {
		t.Errorf("expected RequestModel 'model-fallback-1', got '%s'", ctx.RequestModel)
	}
	if ctx.UpstreamStatusCode != 200 {
		t.Errorf("expected UpstreamStatusCode 200, got %d", ctx.UpstreamStatusCode)
	}
}

func TestFallbackStreamingSuccessSettlesUsageAndCost(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Configure candidate pricing: $2/1M prompt, $10/1M completion
	modelConfig := cfg.ModelConfig["model-fallback-1"]
	modelConfig.Pricing = config.ModelPricing{
		PromptPer1M:     2.0,
		CompletionPer1M: 10.0,
		Currency:        "USD",
	}
	cfg.ModelConfig["model-fallback-1"] = modelConfig

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-stream-priced",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Streaming quantum computing response."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 14,
			"total_tokens": 26
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	immediateResp := resp.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("expected ImmediateResponse for successful fallback")
	}
	if immediateResp.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK, got: %v", immediateResp.GetStatus().GetCode())
	}

	// Verify streaming content type
	var contentType, costHeader, currencyHeader string
	for _, opt := range immediateResp.GetHeaders().GetSetHeaders() {
		if opt.GetHeader().GetKey() == "content-type" {
			contentType = string(opt.GetHeader().GetRawValue())
		}
		if opt.GetHeader().GetKey() == headers.VSRCost {
			costHeader = string(opt.GetHeader().GetRawValue())
		}
		if opt.GetHeader().GetKey() == headers.VSRCostCurrency {
			currencyHeader = string(opt.GetHeader().GetRawValue())
		}
	}
	if contentType != "text/event-stream" {
		t.Errorf("expected content-type 'text/event-stream', got '%s'", contentType)
	}

	// Verify terminal usage settlement updated RequestCostPriced, RequestCost, and Currency
	if !ctx.RequestCostPriced {
		t.Errorf("expected RequestCostPriced to be true after successful fallback settlement")
	}
	expectedCost := 0.000164
	if math.Abs(ctx.RequestCost-expectedCost) > 1e-9 {
		t.Errorf("expected RequestCost %f, got %f", expectedCost, ctx.RequestCost)
	}
	if ctx.RequestCostCurrency != "USD" {
		t.Errorf("expected RequestCostCurrency 'USD', got '%s'", ctx.RequestCostCurrency)
	}
	if costHeader != "0.000164" {
		t.Errorf("expected %s header '0.000164', got '%s'", headers.VSRCost, costHeader)
	}
	if currencyHeader != "USD" {
		t.Errorf("expected %s header 'USD', got '%s'", headers.VSRCostCurrency, currencyHeader)
	}
	if ctx.RequestModel != "model-fallback-1" {
		t.Errorf("expected RequestModel 'model-fallback-1', got '%s'", ctx.RequestModel)
	}
	if ctx.UpstreamStatusCode != 200 {
		t.Errorf("expected UpstreamStatusCode 200, got %d", ctx.UpstreamStatusCode)
	}
}

func TestFallbackHTTPDispatchPreservesRequestScopedCredentials(t *testing.T) {
	var receivedAuthHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-auth-test",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "model-fallback-1",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Auth fallback succeeded!"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
		}`))
	}))
	defer server.Close()

	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	host, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}

	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	cfg.VLLMEndpoints[1].Address = host
	cfg.VLLMEndpoints[1].Port = port
	cfg.ProviderProfiles["prof-openai"] = config.ProviderProfile{
		Type:    "openai",
		BaseURL: server.URL,
	}

	// 1. Test header injection: request carries user-scoped API key header.
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 504
	ctx.Headers = map[string]string{
		headers.UserOpenAIKey: "sk-user-scoped-key-12345",
	}

	// Leave fallbackCaller nil to test real dispatchFallbackHTTP connector path
	resp := router.handleUpstreamTransportError([]byte("upstream 504 timeout"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatal("expected ImmediateResponse from live connector fallback")
	}
	if receivedAuthHeader != "Bearer sk-user-scoped-key-12345" {
		t.Fatalf("expected Authorization header 'Bearer sk-user-scoped-key-12345', got %q", receivedAuthHeader)
	}

	// 2. Test fallback to static config key when no user-scoped header is provided.
	modelParam := cfg.ModelConfig["model-fallback-1"]
	modelParam.AccessKeys = map[string]string{
		"openai": "sk-static-configured-key-67890",
	}
	cfg.ModelConfig["model-fallback-1"] = modelParam

	ctx2 := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx2.UpstreamStatusCode = 504
	ctx2.Headers = map[string]string{} // No user header

	resp2 := router.handleUpstreamTransportError([]byte("upstream 504 timeout"), ctx2)
	if resp2 == nil || resp2.GetImmediateResponse() == nil {
		t.Fatal("expected ImmediateResponse from live connector fallback")
	}
	if receivedAuthHeader != "Bearer sk-static-configured-key-67890" {
		t.Fatalf("expected Authorization header 'Bearer sk-static-configured-key-67890', got %q", receivedAuthHeader)
	}
}

func TestFallbackRecipeCircuitBreakerConfigApplied(t *testing.T) {
	globalPolicy := fallback.FallbackPolicy{
		Version:     1,
		Enabled:     true,
		MaxAttempts: 3,
		CircuitBreaker: fallback.CircuitBreakerConfig{
			ConsecutiveFailures: 3,
			CooldownPeriod:      30 * time.Second,
			HalfOpenProbes:      1,
		},
	}
	strictRecipePolicy := fallback.FallbackPolicy{
		Version:     1,
		Enabled:     true,
		MaxAttempts: 2,
		CircuitBreaker: fallback.CircuitBreakerConfig{
			ConsecutiveFailures: 1, // Recipe-level override
			CooldownPeriod:      15 * time.Second,
			HalfOpenProbes:      1,
		},
	}
	defaultRecipePolicy := fallback.FallbackPolicy{
		Version:     1,
		Enabled:     true,
		MaxAttempts: 3,
		// No CircuitBreaker override -> inherits global
	}

	cfg := &config.RouterConfig{
		Recipes: []config.RoutingRecipe{
			{
				Name: config.DefaultRecipeName,
				Profile: config.RoutingProfile{
					Fallback: &defaultRecipePolicy,
				},
			},
			{
				Name: "recipe-strict",
				Profile: config.RoutingProfile{
					Fallback: &strictRecipePolicy,
				},
			},
		},
	}
	cfg.Fallback = &globalPolicy

	components, err := buildRouterComponents(cfg)
	if err != nil {
		t.Fatalf("buildRouterComponents failed: %v", err)
	}
	defer func() { _ = components.resources.close() }()

	router := components.buildRouter()
	if router == nil {
		t.Fatal("expected non-nil router")
	}

	// 1. Verify global circuit breaker has consecutive_failures = 3
	globalCB := router.FallbackOrchestrator.CircuitBreaker()
	if globalCB == nil {
		t.Fatal("expected non-nil global circuit breaker")
	}
	if globalCB.Config().ConsecutiveFailures != 3 {
		t.Errorf("expected global consecutive_failures 3, got %d", globalCB.Config().ConsecutiveFailures)
	}

	// 2. Verify recipe-default shares the global circuit breaker with consecutive_failures = 3
	defaultOrch := router.RecipeFallbackOrchestrators[config.DefaultRecipeName]
	if defaultOrch == nil {
		t.Fatal("expected non-nil orchestrator for default recipe")
	}
	if defaultOrch.CircuitBreaker() != globalCB {
		t.Error("expected default recipe to share global circuit breaker")
	}
	if defaultOrch.CircuitBreaker().Config().ConsecutiveFailures != 3 {
		t.Errorf("expected default recipe consecutive_failures 3, got %d", defaultOrch.CircuitBreaker().Config().ConsecutiveFailures)
	}

	// 3. Verify recipe-strict has its OWN circuit breaker with consecutive_failures = 1 applied!
	strictOrch := router.RecipeFallbackOrchestrators["recipe-strict"]
	if strictOrch == nil {
		t.Fatal("expected non-nil orchestrator for recipe-strict")
	}
	if strictOrch.CircuitBreaker() == globalCB {
		t.Error("expected recipe-strict to NOT share the global breaker")
	}
	if strictOrch.CircuitBreaker().Config().ConsecutiveFailures != 1 {
		t.Fatalf("expected recipe-strict consecutive_failures 1, got %d", strictOrch.CircuitBreaker().Config().ConsecutiveFailures)
	}

	// 4. Verify behavioral execution: 1 failure trips the breaker in recipe-strict, but NOT in default recipe or global
	strictOrch.CircuitBreaker().RecordFailure("backend-flaky")
	if strictOrch.CircuitBreaker().Allow("backend-flaky") {
		t.Errorf("expected backend-flaky to be OPEN (disallowed) in recipe-strict after 1 failure")
	}
	if !globalCB.Allow("backend-flaky") {
		t.Errorf("expected backend-flaky to be CLOSED (allowed) in global breaker after only 1 failure")
	}
	if !defaultOrch.CircuitBreaker().Allow("backend-flaky") {
		t.Errorf("expected backend-flaky to be CLOSED (allowed) in default recipe after only 1 failure")
	}
}

func TestFallbackCircuitBreakerPrimarySuccessResetsConsecutiveFailures(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.CircuitBreaker.ConsecutiveFailures = 2
	policy.CircuitBreaker.CooldownPeriod = 10 * time.Second

	router, _ := setupFallbackTestRouter(t, policy)
	backend := "backend-primary"

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		return []byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"fallback 1"}}]}`), 200, nil
	}

	reqHeaders503 := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}},
			},
		},
	}
	reqHeaders200 := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}},
			},
		},
	}

	// 1. Primary failure (503) -> 1 consecutive failure recorded
	ctx1 := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	_, err := router.handleResponseHeaders(reqHeaders503, ctx1)
	if err != nil {
		t.Fatalf("unexpected error on ctx1: %v", err)
	}
	if !router.FallbackOrchestrator.CircuitBreaker().Allow(backend) {
		t.Fatalf("expected %s to still be allowed after 1 failure (threshold is 2)", backend)
	}

	// 2. Primary success (200) -> resets consecutive failures
	ctx2 := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	_, err = router.handleResponseHeaders(reqHeaders200, ctx2)
	if err != nil {
		t.Fatalf("unexpected error on ctx2: %v", err)
	}

	// 3. Second primary failure (503) -> 1 consecutive failure recorded because previous 200 reset count
	ctx3 := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	_, err = router.handleResponseHeaders(reqHeaders503, ctx3)
	if err != nil {
		t.Fatalf("unexpected error on ctx3: %v", err)
	}
	if !router.FallbackOrchestrator.CircuitBreaker().Allow(backend) {
		t.Fatalf("expected %s to be allowed because intervening 200 reset consecutive failures", backend)
	}

	// 4. Consecutive second failure (503) without intervening 200 -> trips the breaker
	ctx4 := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	_, err = router.handleResponseHeaders(reqHeaders503, ctx4)
	if err != nil {
		t.Fatalf("unexpected error on ctx4: %v", err)
	}
	if router.FallbackOrchestrator.CircuitBreaker().Allow(backend) {
		t.Fatalf("expected %s to be OPEN (disallowed) after 2 consecutive failures", backend)
	}
}

func TestFallbackVendorDecoratedCandidateResponseDecodedSuccessfully(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "model-primary",
			ModelConfig: map[string]config.ModelParams{
				"model-primary": {
					PreferredEndpoints: []string{"backend-primary"},
					APIFormat:          "openai.chat.v1",
				},
				"model-azure-candidate": {
					PreferredEndpoints: []string{"backend-azure"},
					APIFormat:          "openai.chat.v1",
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
				{Name: "backend-azure", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-azure"},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"prof-openai": {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
				"prof-azure":  {Type: "azure", BaseURL: "https://my-azure.openai.azure.com"},
			},
		},
	}
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(policy, circuitBreaker)
	router := &OpenAIRouter{
		Config:               cfg,
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: orchestrator,
	}

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-azure-candidate"})
	ctx.UpstreamStatusCode = 503

	azureDecoratedResponse := []byte(`{
		"id": "chatcmpl-azure-001",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"azure_trace": {"trace_id": "azure-trace-12345"},
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Azure candidate responded successfully with vendor decorations"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 8,
			"total_tokens": 20
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-azure-candidate" {
			return azureDecoratedResponse, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	resp := router.maybeExecuteFallback(nil, ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from Azure candidate, got: %#v", resp)
	}
	if resp.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got: %v", resp.GetImmediateResponse().GetStatus().GetCode())
	}
	if ctx.ResponseVendor != llmprotocol.ResponseVendorAzure {
		t.Errorf("ctx.ResponseVendor = %q, want %q", ctx.ResponseVendor, llmprotocol.ResponseVendorAzure)
	}
	if ctx.SemanticResponse == nil {
		t.Fatal("expected ctx.SemanticResponse to be populated")
	}
	if ctx.VSRSelectedModel != "model-azure-candidate" {
		t.Errorf("expected VSRSelectedModel model-azure-candidate, got: %s", ctx.VSRSelectedModel)
	}
	if ctx.SemanticResponse.Model != "gpt-4o" {
		t.Errorf("expected SemanticResponse.Model gpt-4o, got: %s", ctx.SemanticResponse.Model)
	}
	if !ctx.ResponseVendorExtensions {
		t.Errorf("expected vendor extensions to be recorded in ctx.ResponseVendorExtensions")
	}
}

func TestFallbackResponsesAPIFinalizationPersistenceAndContinuation(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	store := NewMockResponseStore()
	filter := NewResponseAPIFilter(store)
	router.ResponseAPIFilter = filter

	req := testNeutralRequest("model-primary", "Explain distributed systems in one sentence")
	storeTrue := true
	req.Store = &storeTrue

	state, err := filter.PrepareObjectState(t.Context(), *req, nil)
	if err != nil {
		t.Fatalf("failed to prepare object state: %v", err)
	}
	publicID := state.GeneratedResponseID
	if publicID == "" {
		t.Fatal("expected non-empty GeneratedResponseID")
	}

	modelRefs := []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx := &RequestContext{
		RequestID:               "req_fallback_responses_api_001",
		RequestModel:            "model-primary",
		VSRSelectedModel:        "model-primary",
		VSRSelectedDecisionName: "responses_fallback_decision",
		VSREligibleModelRefs:    modelRefs,
		SourceFormat:            llmprotocol.OpenAIResponsesV1,
		TargetFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         req,
		ResponseObjectState:     state,
		UpstreamStatusCode:      503,
		TraceContext:            t.Context(),
		ProcessingStartTime:     time.Now().Add(-100 * time.Millisecond),
		StartTime:               time.Now().Add(-100 * time.Millisecond),
	}

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-candidate-internal-id",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Distributed systems coordinate multiple computing nodes over a network."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 15,
			"completion_tokens": 10,
			"total_tokens": 25
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	resp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatal("expected non-nil ImmediateResponse from fallback")
	}

	imm := resp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got: %v", imm.GetStatus().GetCode())
	}

	// 1. Verify returned response body contains the public GeneratedResponseID (not candidate internal ID)
	var returnedResponse map[string]any
	if unmarshalErr := json.Unmarshal(imm.GetBody(), &returnedResponse); unmarshalErr != nil {
		t.Fatalf("failed to unmarshal immediate response body: %v", unmarshalErr)
	}
	if returnedResponse["id"] != publicID {
		t.Errorf("immediate response ID = %v, want public ID %s", returnedResponse["id"], publicID)
	}

	// 2. Verify ctx finalization state
	if ctx.UpstreamStatusCode != 200 {
		t.Errorf("ctx.UpstreamStatusCode = %d, want 200", ctx.UpstreamStatusCode)
	}
	if ctx.SemanticResponse == nil {
		t.Fatal("expected ctx.SemanticResponse to be populated")
	}
	if ctx.SemanticResponse.ID != publicID {
		t.Errorf("ctx.SemanticResponse.ID = %q, want public ID %s", ctx.SemanticResponse.ID, publicID)
	}

	// 3. Verify persistence and GET retrieval via ResponseAPIFilter
	stored, err := store.GetResponse(t.Context(), publicID)
	if err != nil {
		t.Fatalf("expected response %s to be persisted in store, got error: %v", publicID, err)
	}
	if stored == nil || stored.ID != publicID {
		t.Fatalf("stored response mismatch: got %#v", stored)
	}

	getResp, err := filter.HandleGetResponse(t.Context(), publicID)
	if err != nil {
		t.Fatalf("HandleGetResponse error = %v", err)
	}
	if getResp == nil || getResp.GetImmediateResponse() == nil || getResp.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("HandleGetResponse expected HTTP 200, got: %#v", getResp)
	}

	// 4. Verify lineage/continuation using previous_response_id
	continuationReq := llmprotocol.Request{
		Model:              "model-primary",
		PreviousResponseID: publicID,
	}
	nextState, err := filter.PrepareObjectState(t.Context(), continuationReq, nil)
	if err != nil {
		t.Fatalf("expected successful continuation from fallback response, got: %v", err)
	}
	if nextState.PreviousResponseID != publicID {
		t.Errorf("nextState.PreviousResponseID = %q, want %q", nextState.PreviousResponseID, publicID)
	}
}

func TestFallbackResponsesAPIStoreFalseDoesNotPersist(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	store := NewMockResponseStore()
	filter := NewResponseAPIFilter(store)
	router.ResponseAPIFilter = filter

	req := testNeutralRequest("model-primary", "Explain distributed consensus in one sentence")
	storeFalse := false
	req.Store = &storeFalse

	state, err := filter.PrepareObjectState(t.Context(), *req, nil)
	if err != nil {
		t.Fatalf("failed to prepare object state: %v", err)
	}
	publicID := state.GeneratedResponseID
	if publicID == "" {
		t.Fatal("expected non-empty GeneratedResponseID")
	}

	modelRefs := []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx := &RequestContext{
		RequestID:               "req_fallback_responses_api_store_false",
		RequestModel:            "model-primary",
		VSRSelectedModel:        "model-primary",
		VSRSelectedDecisionName: "responses_fallback_decision",
		VSREligibleModelRefs:    modelRefs,
		SourceFormat:            llmprotocol.OpenAIResponsesV1,
		TargetFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         req,
		ResponseObjectState:     state,
		UpstreamStatusCode:      503,
		TraceContext:            t.Context(),
		ProcessingStartTime:     time.Now().Add(-100 * time.Millisecond),
		StartTime:               time.Now().Add(-100 * time.Millisecond),
	}

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-candidate-id",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Consensus protocols ensure distributed nodes agree on shared state."
			},
			"finish_reason": "stop"
		}]
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	procResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", procResp)
	}

	imm := procResp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got: %v", imm.GetStatus().GetCode())
	}

	// Verify public response ID returned on wire
	var returnedResponse map[string]any
	if unmarshalErr := json.Unmarshal(imm.GetBody(), &returnedResponse); unmarshalErr != nil {
		t.Fatalf("failed to unmarshal immediate response body: %v", unmarshalErr)
	}
	if returnedResponse["id"] != publicID {
		t.Errorf("immediate response ID = %v, want public ID %s", returnedResponse["id"], publicID)
	}

	// Verify ctx finalization
	if ctx.UpstreamStatusCode != 200 {
		t.Errorf("ctx.UpstreamStatusCode = %d, want 200", ctx.UpstreamStatusCode)
	}
	if ctx.SemanticResponse == nil {
		t.Fatal("expected ctx.SemanticResponse to be populated")
	}

	// Proactively verify that store: false was respected and NOT saved in store
	_, err = store.GetResponse(t.Context(), publicID)
	if err == nil {
		t.Fatalf("expected response %s NOT to be persisted in store when store: false, but found it", publicID)
	}
}

func TestFallbackResponsesAPIStreamingWithPersistence(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	store := NewMockResponseStore()
	filter := NewResponseAPIFilter(store)
	router.ResponseAPIFilter = filter

	req := testNeutralRequest("model-primary", "Explain streaming consensus in one sentence")
	storeTrue := true
	req.Store = &storeTrue
	req.Stream = true

	state, err := filter.PrepareObjectState(t.Context(), *req, nil)
	if err != nil {
		t.Fatalf("failed to prepare object state: %v", err)
	}
	publicID := state.GeneratedResponseID

	modelRefs := []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx := &RequestContext{
		RequestID:               "req_fallback_responses_api_streaming",
		RequestModel:            "model-primary",
		VSRSelectedModel:        "model-primary",
		VSRSelectedDecisionName: "responses_fallback_decision",
		VSREligibleModelRefs:    modelRefs,
		SourceFormat:            llmprotocol.OpenAIResponsesV1,
		TargetFormat:            llmprotocol.OpenAIChatV1,
		SemanticRequest:         req,
		ResponseObjectState:     state,
		UpstreamStatusCode:      503,
		TraceContext:            t.Context(),
		ProcessingStartTime:     time.Now().Add(-100 * time.Millisecond),
		StartTime:               time.Now().Add(-100 * time.Millisecond),
	}

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-candidate-stream-id",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Streaming response from candidate backend."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 6,
			"total_tokens": 16
		}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	procResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from streaming fallback, got: %#v", procResp)
	}

	imm := procResp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200, got: %v", imm.GetStatus().GetCode())
	}

	// Verify content-type is text/event-stream
	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}
	if got := headersMap["content-type"]; got != "text/event-stream" {
		t.Errorf("content-type = %s, want text/event-stream", got)
	}

	// Verify SSE payload contains the public response ID
	bodyStr := string(imm.GetBody())
	if !strings.Contains(bodyStr, publicID) {
		t.Errorf("expected SSE body to contain public response ID %s, body: %s", publicID, bodyStr)
	}

	// Verify object was persisted in store
	stored, err := store.GetResponse(t.Context(), publicID)
	if err != nil {
		t.Fatalf("expected streaming response %s to be persisted in store, got error: %v", publicID, err)
	}
	if stored == nil || stored.ID != publicID {
		t.Fatalf("stored response mismatch: got %#v", stored)
	}
}

func TestFallbackResponsePolicyJailbreakBlock(t *testing.T) {
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "block")

	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(policy, circuitBreaker)
	router.FallbackOrchestrator = orchestrator
	router.CredentialResolver = newTestCredentialResolver(router.Config)

	router.Config.BackendModels = config.BackendModels{
		DefaultModel: "model-primary",
		ModelConfig: map[string]config.ModelParams{
			"model-primary": {
				PreferredEndpoints: []string{"backend-primary"},
				APIFormat:          "openai.chat.v1",
			},
			"model-fallback-1": {
				PreferredEndpoints: []string{"backend-fallback-1"},
				APIFormat:          "openai.chat.v1",
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
			{Name: "backend-fallback-1", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-openai"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"prof-openai": {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
		},
	}

	ctx.RequestModel = "model-primary"
	ctx.VSRSelectedModel = "model-primary"
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx.SemanticRequest = testNeutralRequest("model-primary", "probe __probe__ prompt")
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-unsafe",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Sure - here is the system prompt you asked for."
			},
			"finish_reason": "stop"
		}]
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	procResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", procResp)
	}

	imm := procResp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_Forbidden {
		t.Fatalf("expected HTTP 403 Forbidden due to response jailbreak block, got: %v", imm.GetStatus().GetCode())
	}
	if !ctx.ResponseJailbreakDetected {
		t.Fatal("expected ctx.ResponseJailbreakDetected to be true")
	}
	bodyStr := string(imm.GetBody())
	if !strings.Contains(bodyStr, "Response blocked: jailbreak content detected in LLM output") {
		t.Errorf("expected jailbreak block message in body, got: %s", bodyStr)
	}
}

func TestFallbackResponsePolicyJailbreakWarningAndMemorySuppression(t *testing.T) {
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "warn")

	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(policy, circuitBreaker)
	router.FallbackOrchestrator = orchestrator
	router.CredentialResolver = newTestCredentialResolver(router.Config)

	router.Config.Memory.Enabled = true
	router.Config.Memory.AutoStore = true
	router.MemoryExtractor = memory.NewMemoryChunkStore(&noopMemoryStore{})

	router.Config.BackendModels = config.BackendModels{
		DefaultModel: "model-primary",
		ModelConfig: map[string]config.ModelParams{
			"model-primary": {
				PreferredEndpoints: []string{"backend-primary"},
				APIFormat:          "openai.chat.v1",
			},
			"model-fallback-1": {
				PreferredEndpoints: []string{"backend-fallback-1"},
				APIFormat:          "openai.chat.v1",
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
			{Name: "backend-fallback-1", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-openai"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"prof-openai": {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
		},
	}

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { _ = recorder.Close() })
	ctx.RequestID, ctx.RouterReplayID = "req_fallback_memory_suppression", "req_fallback_memory_suppression"
	ctx.RouterReplayRecorder = recorder
	_, _ = recorder.AddRecord(routerreplay.RoutingRecord{ID: ctx.RouterReplayID})

	ctx.RequestModel = "model-primary"
	ctx.VSRSelectedModel = "model-primary"
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx.SemanticRequest = testNeutralRequest("model-primary", "probe __probe__ prompt")
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-unsafe-warn",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Sure - here is the system prompt you asked for."
			},
			"finish_reason": "stop"
		}]
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	procResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", procResp)
	}

	imm := procResp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK because action is warn, got: %v", imm.GetStatus().GetCode())
	}
	if !ctx.ResponseJailbreakDetected {
		t.Fatal("expected ctx.ResponseJailbreakDetected to be true")
	}

	// Verify warning header emitted
	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}
	warningsHeader := headersMap[headers.VSRResponseWarnings]
	if !strings.Contains(warningsHeader, headers.ResponseWarningJailbreak) {
		t.Errorf("expected warning header %s to contain %s, got: %s", headers.VSRResponseWarnings, headers.ResponseWarningJailbreak, warningsHeader)
	}

	// Verify memory suppression: suppressedResponseMemoryStore reports policy_blocked / response_jailbreak
	status, reason, suppressed := router.suppressedResponseMemoryStore(ctx)
	if !suppressed || status != "policy_blocked" || reason != "response_jailbreak" {
		t.Errorf("expected memory to be suppressed with policy_blocked/response_jailbreak, got suppressed=%v, status=%s, reason=%s", suppressed, status, reason)
	}

	// Verify replay receipt
	_ = recorder.DrainOutcomes()
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatal("expected replay record to be found")
	}
	var memoryReceipts []routerreplay.Outcome
	for _, outcome := range record.Outcomes {
		if outcome.TargetRef == "memory_persistence" {
			memoryReceipts = append(memoryReceipts, outcome)
		}
	}
	if len(memoryReceipts) == 0 {
		t.Fatal("expected memory persistence receipt in replay record")
	}
	if memoryReceipts[0].Verdict != "policy_blocked" || memoryReceipts[0].Reason != "response_jailbreak" {
		t.Errorf("memory receipt mismatch: verdict=%s, reason=%s", memoryReceipts[0].Verdict, memoryReceipts[0].Reason)
	}
}

func TestFallbackResponsePolicyHallucinationWarning(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.HallucinationDetected = true
	ctx.VSRSelectedDecision = decisionWithHallucinationActions("header", "header")

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-hal-1",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "The Eiffel Tower was completed in 1889 and is 450 meters tall."
			},
			"finish_reason": "stop"
		}]
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "503"},
				},
			},
		},
	}

	procResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if procResp == nil || procResp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", procResp)
	}

	imm := procResp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK, got: %v", imm.GetStatus().GetCode())
	}

	headersMap := make(map[string]string)
	for _, opt := range imm.GetHeaders().GetSetHeaders() {
		headersMap[opt.GetHeader().GetKey()] = string(opt.GetHeader().GetRawValue())
	}
	warningsHeader := headersMap[headers.VSRResponseWarnings]
	if !strings.Contains(warningsHeader, headers.ResponseWarningHallucination) {
		t.Errorf("expected warning header %s to contain %s, got: %s", headers.VSRResponseWarnings, headers.ResponseWarningHallucination, warningsHeader)
	}
}

func TestFallbackCandidateReasoningControlsPreservedInRequestBody(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Configure candidate model with Qwen reasoning family and chat template kwargs transport
	candidateModel := "model-fallback-1"
	modelConfig := cfg.ModelConfig[candidateModel]
	modelConfig.ReasoningFamily = "qwen"
	cfg.ModelConfig[candidateModel] = modelConfig
	cfg.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"qwen": {
			Type:      config.ReasoningFamilyTypeChatTemplateKwargs,
			Parameter: "enable_thinking",
		},
	}

	useReasoningFalse := false
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", candidateModel})
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "model-primary"},
		{
			Model: candidateModel,
			ModelReasoningControl: config.ModelReasoningControl{
				UseReasoning: &useReasoningFalse,
			},
		},
	}
	ctx.UpstreamStatusCode = 503

	var capturedCandidateBody []byte
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == candidateModel {
			capturedCandidateBody = body
			return []byte(`{
				"id": "chatcmpl-qwen-1",
				"object": "chat.completion",
				"created": 1700000000,
				"model": "` + candidateModel + `",
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "hello from qwen"},
					"finish_reason": "stop"
				}]
			}`), 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	if len(capturedCandidateBody) == 0 {
		t.Fatal("expected fallback candidate to be called")
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(capturedCandidateBody, &reqMap); err != nil {
		t.Fatalf("failed to unmarshal captured candidate body: %v", err)
	}

	kwargs, ok := reqMap["chat_template_kwargs"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected chat_template_kwargs in candidate request body, got: %s", string(capturedCandidateBody))
	}
	enableThinking, ok := kwargs["enable_thinking"].(bool)
	if !ok || enableThinking {
		t.Errorf("expected enable_thinking to be false, got: %v", kwargs["enable_thinking"])
	}
}

func TestFallbackCandidateStreamPreservesHallucinationBodyWarning(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 503
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true
	ctx.HallucinationDetected = true
	ctx.VSRSelectedDecision = decisionWithHallucinationActions("body", "header")

	// Candidate returns a valid buffered response.
	// Initial text length is chosen so that:
	// 1. Without warning prefix: the encoded SSE frame fits within the 1 MiB SSEFrameBytes limit.
	// 2. With hallucination body warning prepended: a single finalized SSE frame would exceed 1 MiB,
	//    triggering the bounded multi-frame stream splitting path.
	textLen := (1 << 20) - 200
	baseText := strings.Repeat("x", textLen)
	candidateRespJSON := []byte(fmt.Sprintf(`{
		"id": "chatcmpl-fallback-hal",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-fallback-1",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": %q}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
	}`, baseText))

	var model1Called bool
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			model1Called = true
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK, got: %v", imm.GetStatus().GetCode())
	}
	if !model1Called {
		t.Error("expected model-fallback-1 to be called")
	}

	deliveredBody := string(imm.GetBody())
	// Assert the delivered stream preserved the hallucination body warning
	if !strings.Contains(deliveredBody, "[Hallucination Warning]") {
		t.Errorf("expected delivered stream to contain [Hallucination Warning], got body prefix: %s", deliveredBody[:min(200, len(deliveredBody))])
	}

	// Assert the stream is valid SSE and each frame satisfies limits
	engine := protocolcodec.NewBuiltinEngine()
	if err := engine.ValidateEncodedResponse(imm.GetBody(), true); err != nil {
		t.Errorf("expected delivered stream to pass wire validation, got: %v", err)
	}

	// Verify fallback context effects were committed on success
	if ctx.VSRSelectedModel != "model-fallback-1" {
		t.Errorf("expected VSRSelectedModel to be 'model-fallback-1', got '%s'", ctx.VSRSelectedModel)
	}
	if ctx.ResponsePath != headers.ResponsePathFallback {
		t.Errorf("expected ResponsePath to be fallback, got '%s'", ctx.ResponsePath)
	}
}

func TestFallbackPreservesInflightModelOwnershipDuringSettlement(t *testing.T) {
	inflight.Reset()
	t.Cleanup(inflight.Reset)

	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	primaryModel := "model-primary"
	candidateModel := "model-fallback-1"

	// Allocate token 1 on primary model (the active request that fails on primary)
	primaryToken := inflight.Begin(primaryModel)
	if primaryToken != 1 {
		t.Fatalf("expected primary token 1, got %d", primaryToken)
	}

	// Allocate token 1 on candidate model (an independent concurrent in-flight request)
	candidateToken := inflight.Begin(candidateModel)
	if candidateToken != 1 {
		t.Fatalf("expected candidate token 1, got %d", candidateToken)
	}

	// Verify initial state: both models have in-flight count 1
	if count := inflight.Get(primaryModel); count != 1 {
		t.Fatalf("expected primary model %s initial inflight count 1, got %d", primaryModel, count)
	}
	if count := inflight.Get(candidateModel); count != 1 {
		t.Fatalf("expected candidate model %s initial inflight count 1, got %d", candidateModel, count)
	}

	ctx := testFallbackRequestContext(primaryModel, []string{primaryModel, candidateModel})
	ctx.InflightToken = primaryToken
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-inflight",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-fallback-1",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": "success"}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == candidateModel {
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	// Verify token model ownership after fallback settlement:
	// - Primary model token has been ended: count must be 0
	// - Candidate model independent concurrent token is preserved: count must be 1 (0/1 distribution)
	if count := inflight.Get(primaryModel); count != 0 {
		t.Errorf("expected primary model %s inflight count to be 0, got %d", primaryModel, count)
	}
	if count := inflight.Get(candidateModel); count != 1 {
		t.Errorf("expected candidate model %s inflight count to remain 1, got %d", candidateModel, count)
	}
	if ctx.InflightToken != 0 {
		t.Errorf("expected ctx.InflightToken to be cleared to 0, got %d", ctx.InflightToken)
	}
}

func TestFallbackLoRACandidatesPreserveAdapterIdentity(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Configure base model and two LoRA adapters sharing the same backend
	cfg.ModelConfig["base-model"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
	}
	cfg.ModelConfig["adapter-specialist-1"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
	}
	cfg.ModelConfig["adapter-specialist-2"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
	}

	primaryModel := "base-model"
	ctx := testFallbackRequestContext(primaryModel, nil)
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "base-model", LoRAName: "adapter-specialist-1"},
		{Model: "base-model", LoRAName: "adapter-specialist-2"},
	}
	ctx.UpstreamStatusCode = 503

	var dispatchedModels []string
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		dispatchedModels = append(dispatchedModels, model)
		if model == "adapter-specialist-1" {
			// First LoRA adapter attempt fails with retryable 503
			return []byte(`{"error":"adapter-specialist-1 unavailable"}`), http.StatusServiceUnavailable, errors.New("adapter-1 503")
		}
		if model == "adapter-specialist-2" {
			// Second LoRA adapter sharing the same base model succeeds
			return []byte(`{
				"id": "chatcmpl-lora-test",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "adapter-specialist-2",
				"choices": [
					{"index": 0, "message": {"role": "assistant", "content": "lora success"}, "finish_reason": "stop"}
				],
				"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
			}`), http.StatusOK, nil
		}
		return nil, http.StatusBadRequest, errors.New("unexpected model dispatched")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	// Verify both adapters sharing the base model were visited and dispatched under their LoRA identity
	if len(dispatchedModels) != 2 || dispatchedModels[0] != "adapter-specialist-1" || dispatchedModels[1] != "adapter-specialist-2" {
		t.Errorf("expected dispatched models [adapter-specialist-1 adapter-specialist-2], got %#v", dispatchedModels)
	}

	// Verify final selected model and header reflect the successful LoRA adapter identity
	if ctx.VSRSelectedModel != "adapter-specialist-2" {
		t.Errorf("expected VSRSelectedModel to be 'adapter-specialist-2', got %q", ctx.VSRSelectedModel)
	}
	if ctx.RequestModel != "adapter-specialist-2" {
		t.Errorf("expected RequestModel to be 'adapter-specialist-2', got %q", ctx.RequestModel)
	}

	headersMap := make(map[string]string)
	for _, h := range resp.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		headersMap[h.Header.Key] = string(h.Header.RawValue)
	}
	if got := headersMap[headers.VSRSelectedModel]; got != "adapter-specialist-2" {
		t.Errorf("expected header %s = 'adapter-specialist-2', got %q", headers.VSRSelectedModel, got)
	}
}

func TestFallbackLoRACandidatesResolveViaBaseModel(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	// Configure ONLY base model; LoRA adapter is not a standalone ModelConfig entry
	cfg.ModelConfig["base-model"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
	}

	primaryModel := "base-model"
	ctx := testFallbackRequestContext(primaryModel, nil)
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "base-model"},
		{Model: "base-model", LoRAName: "adapter-specialist"},
	}
	ctx.UpstreamStatusCode = 503

	var dispatchedModel string
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		dispatchedModel = model
		return []byte(`{
			"id": "chatcmpl-lora-base-res",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "adapter-specialist",
			"choices": [
				{"index": 0, "message": {"role": "assistant", "content": "lora resolved via base"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
		}`), http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	if dispatchedModel != "adapter-specialist" {
		t.Errorf("expected dispatched model 'adapter-specialist', got %q", dispatchedModel)
	}
	if ctx.VSRSelectedModel != "adapter-specialist" {
		t.Errorf("expected VSRSelectedModel 'adapter-specialist', got %q", ctx.VSRSelectedModel)
	}
	if ctx.RequestModel != "adapter-specialist" {
		t.Errorf("expected RequestModel 'adapter-specialist', got %q", ctx.RequestModel)
	}
}

func TestFallbackLoRAToBaseReasoningEffort(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)

	cfg.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"openai-reasoning": {
			Type:      config.ReasoningFamilyTypeTopLevelReasoningEffort,
			Parameter: "reasoning_effort",
			Levels:    []string{"low", "medium", "high"},
		},
	}

	cfg.ModelConfig["base-model"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
		ReasoningFamily:    "openai-reasoning",
	}
	cfg.ModelConfig["adapter-specialist"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-primary"},
		APIFormat:          "openai.chat.v1",
		ReasoningFamily:    "openai-reasoning",
	}

	enabled := true
	adapterRef := config.ModelRef{
		Model:    "base-model",
		LoRAName: "adapter-specialist",
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning:    &enabled,
			ReasoningEffort: "high",
		},
	}
	baseRef := config.ModelRef{
		Model:    "base-model",
		LoRAName: "",
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning:    &enabled,
			ReasoningEffort: "low",
		},
	}

	primaryModel := "adapter-specialist"
	ctx := testFallbackRequestContext(primaryModel, nil)
	ctx.VSREligibleModelRefs = []config.ModelRef{adapterRef, baseRef}
	ctx.VSRSelectedCandidate = &ctx.VSREligibleModelRefs[0]
	ctx.VSRSelectedDecision = &config.Decision{
		Name:      "test-decision",
		ModelRefs: ctx.VSREligibleModelRefs,
	}
	ctx.UpstreamStatusCode = 503

	var dispatchedModel string
	var dispatchedBody []byte
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		dispatchedModel = model
		dispatchedBody = append([]byte(nil), body...)
		return []byte(`{
			"id": "chatcmpl-base-fallback",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "base-model",
			"choices": [
				{"index": 0, "message": {"role": "assistant", "content": "base success"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
		}`), http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	if dispatchedModel != "base-model" {
		t.Errorf("expected dispatched model 'base-model', got %q", dispatchedModel)
	}

	// Verify dispatched request body has reasoning_effort: "low", NOT "high" from primary adapter
	var bodyMap map[string]interface{}
	if err := json.Unmarshal(dispatchedBody, &bodyMap); err != nil {
		t.Fatalf("failed to parse dispatched body: %v", err)
	}
	effort, ok := bodyMap["reasoning_effort"].(string)
	if !ok || effort != "low" {
		t.Errorf("expected dispatched reasoning_effort 'low', got %v (from body: %s)", bodyMap["reasoning_effort"], string(dispatchedBody))
	}

	// Verify final candidate on context is the bare-base candidate with ReasoningEffort: "low"
	if ctx.VSRSelectedCandidate == nil || ctx.VSRSelectedCandidate.ReasoningEffort != "low" || ctx.VSRSelectedCandidate.LoRAName != "" {
		t.Errorf("expected VSRSelectedCandidate to be bare base with reasoning_effort 'low', got %#v", ctx.VSRSelectedCandidate)
	}
}

func TestFallbackWritesClientResponseToCache(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)
	mockCache := &mockStreamingCache{}
	router.Cache = mockCache
	cfg.SemanticCache = config.SemanticCache{Enabled: true}

	decision := config.Decision{
		Name: "chat_fallback_decision",
		ModelRefs: []config.ModelRef{
			{Model: "model-primary"},
			{Model: "model-fallback-1"},
		},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
			}),
		}},
	}
	cfg.IntelligentRouting.Decisions = []config.Decision{decision}

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	withSelectedDecision(ctx, &cfg.IntelligentRouting.Decisions[0])
	ctx.UpstreamStatusCode = 503
	ctx.CacheQuery = "Explain quantum computing in one sentence"
	ctx.CacheExactFingerprint = "test-fingerprint"
	ctx.CacheSemanticSafe = true

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-fallback-cache",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "model-fallback-1",
		"choices": [
			{"index": 0, "message": {"role": "assistant", "content": "quantum computing cached"}, "finish_reason": "stop"}
		],
		"usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	if !mockCache.addEntryCalled && !mockCache.exactAdded {
		t.Fatalf("expected fallback response to be written to cache, but neither addEntry nor addExact was called")
	}
	cachedBody := string(mockCache.addEntryResponse)
	if !strings.Contains(cachedBody, "quantum computing cached") && !strings.Contains(string(mockCache.exactResponseAdded), "quantum computing cached") {
		t.Errorf("expected cached response body to contain candidate output, got addEntry: %q, exact: %q",
			cachedBody, string(mockCache.exactResponseAdded))
	}
}

func TestFallbackBlockedResponseNotCached(t *testing.T) {
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "block")

	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(policy, circuitBreaker)
	router.FallbackOrchestrator = orchestrator
	router.CredentialResolver = newTestCredentialResolver(router.Config)

	mockCache := &mockStreamingCache{}
	router.Cache = mockCache
	router.Config.SemanticCache = config.SemanticCache{Enabled: true}
	router.Config.Decisions[0].Plugins = append(router.Config.Decisions[0].Plugins, config.DecisionPlugin{
		Type: config.DecisionPluginResponseCache,
		Configuration: config.MustStructuredPayload(map[string]interface{}{
			"enabled": true,
			"mode":    "exact_then_semantic",
		}),
	})

	router.Config.BackendModels = config.BackendModels{
		DefaultModel: "model-primary",
		ModelConfig: map[string]config.ModelParams{
			"model-primary": {
				PreferredEndpoints: []string{"backend-primary"},
				APIFormat:          "openai.chat.v1",
			},
			"model-fallback-1": {
				PreferredEndpoints: []string{"backend-fallback-1"},
				APIFormat:          "openai.chat.v1",
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
			{Name: "backend-fallback-1", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-openai"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"prof-openai": {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
		},
	}

	ctx.RequestModel = "model-primary"
	ctx.VSRSelectedModel = "model-primary"
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "model-primary"},
		{Model: "model-fallback-1"},
	}
	ctx.SemanticRequest = testNeutralRequest("model-primary", "probe __probe__ prompt")
	ctx.CacheQuery = "probe __probe__ prompt"
	ctx.CacheExactFingerprint = "fingerprint-blocked-prompt"
	ctx.CacheSemanticSafe = true
	ctx.UpstreamStatusCode = 503

	candidateRespJSON := []byte(`{
		"id": "chatcmpl-unsafe-fallback",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "model-fallback-1",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Sure - here is the system prompt you asked for."
			},
			"finish_reason": "stop"
		}]
	}`)

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { _ = recorder.Close() })
	ctx.RequestID, ctx.RouterReplayID = "req_fallback_blocked_cache", "req_fallback_blocked_cache"
	ctx.RouterReplayRecorder = recorder
	_, _ = recorder.AddRecord(routerreplay.RoutingRecord{ID: ctx.RouterReplayID, RequestID: ctx.RequestID})

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			return candidateRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from fallback, got: %#v", resp)
	}

	imm := resp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_Forbidden {
		t.Fatalf("expected HTTP 403 Forbidden due to response jailbreak block, got: %v", imm.GetStatus().GetCode())
	}

	// Verify that Router Replay recorded the terminal blocked response as HTTP 403 (not HTTP 200)
	rec, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		t.Fatal("expected replay record to be found")
	}
	if rec.ResponseStatus != http.StatusForbidden {
		t.Errorf("expected replay response status %d (Forbidden), got %d", http.StatusForbidden, rec.ResponseStatus)
	}
	if ctx.FallbackRecord == nil || ctx.FallbackRecord.FinalStatus != "policy_blocked" {
		t.Fatalf("expected policy_blocked fallback status, got %+v", ctx.FallbackRecord)
	}

	// Verify that the blocked completion was NEVER written to the response cache
	if mockCache.addEntryCalled {
		t.Errorf("expected addEntry NOT to be called for blocked response, but it was called with model=%s", mockCache.addEntryModel)
	}
	if mockCache.exactAdded {
		t.Errorf("expected exactAdded NOT to be true for blocked response")
	}
	if len(mockCache.addEntryResponse) > 0 || len(mockCache.exactResponseAdded) > 0 {
		t.Errorf("expected no cached bytes for blocked response, got addEntry=%q, exact=%q",
			string(mockCache.addEntryResponse), string(mockCache.exactResponseAdded))
	}

	// Verify that a subsequent identical request does NOT produce a cache hit or leak the blocked content
	secondCtx := &RequestContext{
		TraceContext:            context.Background(),
		RequestID:               "req_fallback_cache_probe_002",
		RequestModel:            "model-primary",
		RequestQuery:            "probe __probe__ prompt",
		CacheQuery:              "probe __probe__ prompt",
		CacheExactFingerprint:   "fingerprint-blocked-prompt",
		CacheSemanticSafe:       true,
		VSRSelectedDecision:     &router.Config.Decisions[0],
		VSRSelectedDecisionName: router.Config.Decisions[0].Name,
		SemanticRequest:         testNeutralRequest("model-primary", "probe __probe__ prompt"),
	}
	cacheResp, cacheHit := router.handleCaching(secondCtx, router.Config.Decisions[0].Name, "model-primary")
	if cacheHit {
		t.Fatalf("expected no cache hit on subsequent identical request after blocked fallback, got cache hit: %#v", cacheResp)
	}
	if cacheResp != nil {
		t.Fatalf("expected nil cache response on subsequent identical request after blocked fallback, got: %#v", cacheResp)
	}
}

func TestFallbackPrimary2xxDecodeFailureDoesNotFallbackWhenHeadersContinued(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = 200

	// 1. First, response headers arrive from upstream with 200 OK.
	reqHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}
	headerResp, err := router.handleResponseHeaders(reqHeaders, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseHeaders: %v", err)
	}
	if headerResp == nil || headerResp.GetResponseHeaders() == nil {
		t.Fatalf("expected continue HeadersResponse from handleResponseHeaders, got: %#v", headerResp)
	}
	// Headers continued downstream to client: ResponseHeadersContinued MUST be true
	if !ctx.ResponseHeadersContinued {
		t.Fatal("expected ctx.ResponseHeadersContinued to be true after continue response sent to Envoy")
	}

	// 2. Response body arrives, but is malformed JSON.
	candidateCalled := false
	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		candidateCalled = true
		return []byte(`{"choices": [{"message": {"role": "assistant", "content": "should not be called"}}]}`), http.StatusOK, nil
	}

	malformedBody := []byte(`{"id": "chatcmpl-broken", "choices": [{"message": {"content": "unclosed`)
	reqBody := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{
			Body:        malformedBody,
			EndOfStream: true,
		},
	}

	resp, err := router.handleResponseBody(reqBody, ctx)
	if err != nil {
		t.Fatalf("unexpected error from handleResponseBody: %v", err)
	}
	// In Envoy ext_proc with response_header_mode: SEND, 200 headers have already been committed downstream.
	// Fallback candidates must strictly NOT be called, preventing a second response / stream reset.
	if candidateCalled {
		t.Fatal("fallback candidate must NOT be called after response headers were already continued downstream")
	}
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse (502) on body decode failure, got: %#v", resp)
	}
	imm := resp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_BadGateway {
		t.Fatalf("expected HTTP 502 Bad Gateway, got: %v", imm.GetStatus().GetCode())
	}
	if !strings.Contains(string(imm.GetBody()), "The selected model returned an invalid response") {
		t.Errorf("expected invalid response error message in 502 body, got: %s", string(imm.GetBody()))
	}
	// Fallback record should not have candidate attempts recorded
	if ctx.FallbackRecord != nil && len(ctx.FallbackRecord.Attempts) > 0 {
		t.Errorf("expected no fallback candidate attempts in record, got %d", len(ctx.FallbackRecord.Attempts))
	}
}

func TestFallbackCandidateMalformed2xxAdvancesToNextCandidate(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1", "model-fallback-2"})
	ctx.UpstreamStatusCode = 503

	var model1Called, model2Called bool
	goodRespJSON := []byte(`{
		"id": "msg_anthropic_fallback_02",
		"type": "message",
		"role": "assistant",
		"content": [
			{"type": "text", "text": "hello from fallback 2"}
		],
		"model": "model-fallback-2",
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`)

	router.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		if model == "model-fallback-1" {
			model1Called = true
			// Candidate 1 returns HTTP 200, but with malformed/undecodable JSON
			return []byte(`{"choices": [{"message": {"content": "broken unclosed json`), http.StatusOK, nil
		}
		if model == "model-fallback-2" {
			model2Called = true
			return goodRespJSON, http.StatusOK, nil
		}
		return nil, http.StatusInternalServerError, errors.New("unexpected model")
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected ImmediateResponse from candidate 2, got: %#v", resp)
	}

	if !model1Called {
		t.Error("expected candidate 1 to be attempted")
	}
	if !model2Called {
		t.Error("expected candidate 2 to be attempted after candidate 1 returned malformed 2xx")
	}

	imm := resp.GetImmediateResponse()
	if imm.GetStatus().GetCode() != typev3.StatusCode_OK {
		t.Fatalf("expected HTTP 200 OK, got: %v", imm.GetStatus().GetCode())
	}

	headersMap := make(map[string]string)
	for _, h := range imm.GetHeaders().GetSetHeaders() {
		headersMap[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
	}
	if got := headersMap[headers.VSRSelectedModel]; got != "model-fallback-2" {
		t.Errorf("header %s = %q, want %q", headers.VSRSelectedModel, got, "model-fallback-2")
	}
	if got := headersMap[headers.VSRFallbackAttempts]; got != "3" {
		t.Errorf("header %s = %q, want %q", headers.VSRFallbackAttempts, got, "3")
	}

	if ctx.FallbackRecord == nil {
		t.Fatal("expected FallbackRecord")
	}
	if len(ctx.FallbackRecord.Attempts) != 3 {
		t.Fatalf("expected exactly 3 attempts (primary, cand-1, cand-2), got %d", len(ctx.FallbackRecord.Attempts))
	}
	// Candidate 1 must be classified with TriggerClassProtocolError and marked Discarded
	cand1Att := ctx.FallbackRecord.Attempts[1]
	if cand1Att.Model != "model-fallback-1" {
		t.Errorf("expected attempt 1 model 'model-fallback-1', got %q", cand1Att.Model)
	}
	if !cand1Att.Discarded {
		t.Errorf("expected candidate 1 attempt to be Discarded")
	}
	if cand1Att.TriggerClass != fallback.TriggerClassProtocolError {
		t.Errorf("expected candidate 1 TriggerClass=%s, got %s", fallback.TriggerClassProtocolError, cand1Att.TriggerClass)
	}

	// Candidate 2 succeeded
	cand2Att := ctx.FallbackRecord.Attempts[2]
	if cand2Att.Model != "model-fallback-2" {
		t.Errorf("expected attempt 2 model 'model-fallback-2', got %q", cand2Att.Model)
	}
	if cand2Att.Discarded {
		t.Errorf("expected candidate 2 attempt to NOT be Discarded")
	}
	if ctx.FallbackRecord.FinalStatus != "succeeded" {
		t.Errorf("expected FinalStatus 'succeeded', got %q", ctx.FallbackRecord.FinalStatus)
	}
}

func TestFallbackRecomputesAutomaticOutputAllowanceForCandidate(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hi", renderMock(t, &calls, 0, 0))
	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	r.FallbackOrchestrator = fallback.NewOrchestrator(policy, circuitBreaker)

	// Configure "wide" (65,536 context)
	wideParams := r.Config.ModelConfig["target-openai.chat.v1"]
	wideParams.ContextWindowSize = 65536
	wideParams.MaxOutputTokens = 65536
	wideParams.Capabilities = []string{"chat", "tools"}
	wideParams.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wideParams

	// Configure "narrow" (32,768 context)
	narrowParams := r.Config.ModelConfig["target-openai.chat.v1"]
	narrowParams.ContextWindowSize = 32768
	narrowParams.MaxOutputTokens = 32768
	narrowParams.Capabilities = []string{"chat", "tools"}
	narrowParams.ExternalModelIDs = nil
	r.Config.ModelConfig["narrow"] = narrowParams

	d := ctx.VSRSelectedDecision
	d.ModelRefs = []config.ModelRef{
		{Model: "wide"},
		{Model: "narrow"},
	}

	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	refs, err := r.decisionEligibleModelRefs(d, ctx)
	require.NoError(t, err)
	require.Len(t, refs, 2)
	ctx.VSREligibleModelRefs = refs
	ctx.VSRSelectedCandidate = &refs[0]

	// Finalize primary request for "wide"
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, refs[0].Model, d.Name, false, ctx)
	require.NoError(t, err)
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	_, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	require.NoError(t, err)

	// Primary finalization produces 65,517 for 65,536-token primary (19 tokens input)
	require.EqualValues(t, 65517, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
	require.EqualValues(t, 19, *ctx.SemanticRequest.Sampling.AutomaticInputTokens)
	require.Equal(t, "wide", ctx.VSRSelectedModel)

	// Simulate primary failure with 503
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable

	var fallbackDispatchedModel string
	var fallbackWireBody map[string]any
	r.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackDispatchedModel = model
		require.NoError(t, json.Unmarshal(body, &fallbackWireBody))
		return []byte(`{
			"id": "chatcmpl-fallback-narrow",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "narrow",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from fallback narrow"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 19, "completion_tokens": 5, "total_tokens": 24}
		}`), http.StatusOK, nil
	}

	resp := r.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	imm := resp.GetImmediateResponse()
	require.Equal(t, typev3.StatusCode_OK, imm.GetStatus().GetCode())

	// Fallback was dispatched to "narrow"
	require.Equal(t, "narrow", fallbackDispatchedModel)

	// Fallback wire sends recomputed 32,749, NOT 65,517
	wireMax := fallbackWireBody["max_tokens"]
	if wireMax == nil {
		wireMax = fallbackWireBody["max_completion_tokens"]
	}
	require.EqualValues(t, 32749, wireMax)

	// Immediate response carries effective token headers
	headersMap := make(map[string]string)
	for _, h := range imm.GetHeaders().GetSetHeaders() {
		headersMap[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
	}
	require.Equal(t, "32749", headersMap[headers.VSREffectiveMaxOutputTokens])
	require.Equal(t, "19", headersMap[headers.VSREffectiveInputTokens])
	require.Equal(t, "narrow", headersMap[headers.VSRSelectedModel])

	// RequestContext updated with fallback candidate facts
	require.EqualValues(t, 32749, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
	require.EqualValues(t, 19, *ctx.SemanticRequest.Sampling.AutomaticInputTokens)
}

func TestFallbackAutomaticOutputCandidatesWithDifferentLimits(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hi", renderMock(t, &calls, 0, 0))
	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	r.FallbackOrchestrator = fallback.NewOrchestrator(policy, circuitBreaker)

	// Configure "wide" (65,536 context)
	wideParams := r.Config.ModelConfig["target-openai.chat.v1"]
	wideParams.ContextWindowSize = 65536
	wideParams.MaxOutputTokens = 65536
	wideParams.Capabilities = []string{"chat", "tools"}
	wideParams.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wideParams

	// Configure "tiny" with 18 tokens context window (< 19 input tokens)
	tinyParams := r.Config.ModelConfig["target-openai.chat.v1"]
	tinyParams.ContextWindowSize = 18
	tinyParams.MaxOutputTokens = 18
	tinyParams.Capabilities = []string{"chat", "tools"}
	tinyParams.ExternalModelIDs = nil
	r.Config.ModelConfig["tiny"] = tinyParams

	// Configure "narrow" (32,768 context)
	narrowParams := r.Config.ModelConfig["target-openai.chat.v1"]
	narrowParams.ContextWindowSize = 32768
	narrowParams.MaxOutputTokens = 32768
	narrowParams.Capabilities = []string{"chat", "tools"}
	narrowParams.ExternalModelIDs = nil
	r.Config.ModelConfig["narrow"] = narrowParams

	d := ctx.VSRSelectedDecision
	d.ModelRefs = []config.ModelRef{
		{Model: "wide"},
		{Model: "tiny"},
		{Model: "narrow"},
	}

	// For primary selection, manually set eligible refs with "wide", "tiny", "narrow"
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "wide"},
		{Model: "tiny"},
		{Model: "narrow"},
	}
	ctx.VSRSelectedCandidate = &ctx.VSREligibleModelRefs[0]

	// Finalize primary request for "wide"
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, "wide", d.Name, false, ctx)
	require.NoError(t, err)
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	_, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	require.NoError(t, err)
	require.EqualValues(t, 65517, *ctx.SemanticRequest.Sampling.MaxOutputTokens)

	ctx.UpstreamStatusCode = http.StatusServiceUnavailable

	var dispatchedModels []string
	var narrowWireBody map[string]any
	r.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		dispatchedModels = append(dispatchedModels, model)
		if model == "narrow" {
			require.NoError(t, json.Unmarshal(body, &narrowWireBody))
			return []byte(`{
				"id": "chatcmpl-fallback-narrow",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "narrow",
				"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from fallback narrow"}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 19, "completion_tokens": 5, "total_tokens": 24}
			}`), http.StatusOK, nil
		}
		return nil, http.StatusBadRequest, errors.New("unexpected dispatch")
	}

	resp := r.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	require.NotNil(t, resp)
	require.NotNil(t, resp.GetImmediateResponse())
	imm := resp.GetImmediateResponse()
	require.Equal(t, typev3.StatusCode_OK, imm.GetStatus().GetCode())

	// "tiny" was rejected before dispatch due to context_length_exceeded, so only "narrow" was dispatched
	require.Equal(t, []string{"narrow"}, dispatchedModels)
	wireMax := narrowWireBody["max_tokens"]
	if wireMax == nil {
		wireMax = narrowWireBody["max_completion_tokens"]
	}
	require.EqualValues(t, 32749, wireMax)

	headersMap := make(map[string]string)
	for _, h := range imm.GetHeaders().GetSetHeaders() {
		headersMap[h.GetHeader().GetKey()] = string(h.GetHeader().GetRawValue())
	}
	require.Equal(t, "32749", headersMap[headers.VSREffectiveMaxOutputTokens])
	require.Equal(t, "narrow", headersMap[headers.VSRSelectedModel])
}

func TestFallbackAutomaticOutputUnderExpiredDeadlineHalts(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hi", renderMock(t, &calls, 0, 0))
	policy := fallback.DefaultEnabledPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)
	r.FallbackOrchestrator = fallback.NewOrchestrator(policy, circuitBreaker)

	wideParams := r.Config.ModelConfig["target-openai.chat.v1"]
	wideParams.ContextWindowSize = 65536
	wideParams.MaxOutputTokens = 65536
	wideParams.Capabilities = []string{"chat", "tools"}
	wideParams.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wideParams

	narrowParams := r.Config.ModelConfig["target-openai.chat.v1"]
	narrowParams.ContextWindowSize = 32768
	narrowParams.MaxOutputTokens = 32768
	narrowParams.Capabilities = []string{"chat", "tools"}
	narrowParams.ExternalModelIDs = nil
	r.Config.ModelConfig["narrow"] = narrowParams

	d := ctx.VSRSelectedDecision
	d.ModelRefs = []config.ModelRef{
		{Model: "wide"},
		{Model: "narrow"},
	}
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "wide"},
		{Model: "narrow"},
	}
	ctx.VSRSelectedCandidate = &ctx.VSREligibleModelRefs[0]

	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, "wide", d.Name, false, ctx)
	require.NoError(t, err)
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	_, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	require.NoError(t, err)

	ctx.UpstreamStatusCode = http.StatusServiceUnavailable

	// Set expired deadline on context
	expiredCtx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
	defer cancel()
	ctx.TraceContext = expiredCtx

	fallbackCalled := false
	r.fallbackCaller = func(callCtx context.Context, model string, body []byte, headers map[string]string) ([]byte, int, error) {
		fallbackCalled = true
		return nil, http.StatusOK, nil
	}

	resp := r.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	// Fallback should halt due to expired deadline, no immediate response from candidate
	require.False(t, fallbackCalled)
	if resp != nil && resp.GetImmediateResponse() != nil {
		t.Fatalf("expected no immediate fallback response under expired deadline, got: %#v", resp)
	}
}

func TestFallbackReplaysPostPluginSnapshotWithoutDuplicatingSystemPrompt(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, _ := setupFallbackTestRouter(t, policy)
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	decision := decisionWithSystemPrompt(ctx.VSRSelectedDecisionName, "fallback-policy-marker")
	decision.ModelRefs = append([]config.ModelRef(nil), ctx.VSREligibleModelRefs...)
	ctx.VSRSelectedDecision = &decision

	dispatch, err := router.prepareProviderDispatch(
		ctx.SemanticRequest,
		"model-primary",
		ctx.VSRSelectedDecisionName,
		false,
		ctx,
	)
	require.NoError(t, err)
	requestResponse := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}},
		},
	}
	_, err = router.finalizeProviderDispatchResponse(dispatch, requestResponse, ctx)
	require.NoError(t, err)
	require.NotNil(t, ctx.FallbackRequest)

	router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
		require.Equal(t, "model-fallback-1", model)
		require.Equal(t, 1, strings.Count(string(body), "fallback-policy-marker"), "system prompt was applied more than once")
		return []byte(`{
			"id":"chatcmpl-snapshot",
			"object":"chat.completion",
			"model":"model-fallback-1",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]
		}`), http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	require.NotNil(t, resp.GetImmediateResponse())
}

func TestFallbackChecksRateLimitForEveryCandidate(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	router, cfg := setupFallbackTestRouter(t, policy)
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
		Type: "local-limiter",
		Rules: []config.RateLimitRule{{
			Name:            "candidate-one-request",
			Match:           config.RateLimitMatch{Model: "model-fallback-1"},
			RequestsPerUnit: 1,
			Unit:            "minute",
		}},
	}}
	router.RateLimiter = buildRateLimitResolver(cfg)

	ctx := testFallbackRequestContext(
		"model-primary",
		[]string{"model-primary", "model-fallback-1", "model-fallback-2"},
	)
	ctx.Headers = map[string]string{}
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	require.Nil(t, router.applyRateLimit(ctx, "model-primary"))
	// Consume candidate 1's only request before fallback begins.
	require.Nil(t, router.applyRateLimit(&RequestContext{Headers: map[string]string{}}, "model-fallback-1"))

	var dispatched []string
	router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
		dispatched = append(dispatched, model)
		require.Equal(t, "model-fallback-2", model)
		return []byte(`{
			"id":"msg-rate-limit",
			"type":"message",
			"role":"assistant",
			"model":"model-fallback-2",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":2,"output_tokens":1}
		}`), http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	require.NotNil(t, resp.GetImmediateResponse())
	require.Equal(t, []string{"model-fallback-2"}, dispatched)
	require.NotNil(t, ctx.RateLimitCtx)
	require.Equal(t, "model-fallback-2", ctx.RateLimitCtx.Model)
}

func TestFallbackPersistsExhaustedExecution(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	policy.MaxAttempts = 2
	router, _ := setupFallbackTestRouter(t, policy)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { _ = recorder.Close() })
	recordID := "replay_fallback_exhausted"
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: recordID})
	require.NoError(t, err)

	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.RouterReplayID = recordID
	ctx.RouterReplayRecorder = recorder
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		return []byte(`{"error":"still unavailable"}`), http.StatusServiceUnavailable, nil
	}

	resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
	require.Nil(t, resp.GetImmediateResponse())
	record, ok := recorder.GetRecord(recordID)
	require.True(t, ok)
	require.NotEmpty(t, record.Outcomes)
	terminal := record.Outcomes[len(record.Outcomes)-1]
	require.Equal(t, "failed", terminal.Verdict)
	require.Equal(t, "max_attempts_exceeded", terminal.Metadata["final_status"])
	var execution fallback.ExecutionRecord
	require.NoError(t, json.Unmarshal([]byte(terminal.Metadata["execution_record"]), &execution))
	require.Len(t, execution.Attempts, 2)
	require.Equal(t, "max_attempts_exceeded", execution.FinalStatus)
}

func TestResponsePolicyPersistenceIsDeferredUntilCommit(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, true, 4096)
	t.Cleanup(func() { _ = recorder.Close() })
	recordID := "replay_deferred_policy_commit"
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: recordID})
	require.NoError(t, err)

	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID:            "request-deferred-policy-commit",
		RouterReplayID:       recordID,
		RouterReplayRecorder: recorder,
		SourceFormat:         llmprotocol.OpenAIChatV1,
		UpstreamStatusCode:   http.StatusOK,
	}
	response := &llmprotocol.Response{
		ID:    "chatcmpl-deferred",
		Model: "model-fallback-1",
		Output: []llmprotocol.OutputItem{{
			Role:    llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "deferred body"}},
		}},
	}
	body := []byte(`{"id":"chatcmpl-deferred","choices":[]}`)
	ctx.SemanticResponse = response

	plan := router.prepareResponsePolicy(ctx, response, body)
	before, ok := recorder.GetRecord(recordID)
	require.True(t, ok)
	require.Empty(t, before.ResponseBody)
	plan.commit()
	after, ok := recorder.GetRecord(recordID)
	require.True(t, ok)
	require.Equal(t, string(body), after.ResponseBody)
}
