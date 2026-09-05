package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type promptCacheWireRequest struct {
	System []struct {
		Text         string `json:"text"`
		CacheControl *struct {
			Type string `json:"type"`
			TTL  string `json:"ttl"`
		} `json:"cache_control"`
	} `json:"system"`
	Messages []struct {
		Content []struct {
			CacheControl interface{} `json:"cache_control"`
		} `json:"content"`
	} `json:"messages"`
	Tools []struct {
		Name         string `json:"name"`
		CacheControl *struct {
			Type string `json:"type"`
			TTL  string `json:"ttl"`
		} `json:"cache_control"`
	} `json:"tools"`
}

func TestEncodeDispatchRequestInjectsAnthropicPromptCacheMarkers(t *testing.T) {
	request := promptCacheTestRequest()
	before := testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues(
		config.DecisionPluginPromptCache,
		"prompt-cache-route",
		promptCacheActionInserted,
	))
	ctx := &RequestContext{
		SourceFormat:            llmprotocol.OpenAIChatV1,
		TargetFormat:            llmprotocol.AnthropicMessagesV1,
		SemanticRequest:         request,
		VSRSelectedDecisionName: "prompt-cache-route",
		VSRSelectedDecision:     promptCacheTestDecision(map[string]interface{}{"enabled": true, "ttl": "1h"}),
	}

	body, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode dispatch request: %v", err)
	}

	var wire promptCacheWireRequest
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}
	assertPromptCacheWireRequest(t, wire)
	assertPromptCacheReceipt(t, ctx, promptCacheActionInserted, 2, 0)
	if request.Generation != 1 {
		t.Fatalf("retained request generation = %d, want 1", request.Generation)
	}
	if countPromptCacheMarkers(*request) != 0 {
		t.Fatalf("provider marker leaked into retained request: %#v", request)
	}
	after := testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues(
		config.DecisionPluginPromptCache,
		"prompt-cache-route",
		promptCacheActionInserted,
	))
	if after != before+1 {
		t.Fatalf("prompt cache metric = %v, want %v", after, before+1)
	}

	repeatedBody, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("repeat encode dispatch request: %v", err)
	}
	if !bytes.Equal(repeatedBody, body) {
		t.Fatal("repeat encode changed the provider request")
	}
	if request.Generation != 1 {
		t.Fatalf("repeat encode changed retained generation: %d", request.Generation)
	}
	if countPromptCacheMarkers(*request) != 0 {
		t.Fatalf("repeat encode changed retained prompt cache state: %#v", request)
	}
	repeatedMetric := testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues(
		config.DecisionPluginPromptCache,
		"prompt-cache-route",
		promptCacheActionInserted,
	))
	if repeatedMetric != after {
		t.Fatalf("repeat encode metric = %v, want %v", repeatedMetric, after)
	}
}

func assertPromptCacheWireRequest(t *testing.T, wire promptCacheWireRequest) {
	t.Helper()
	if len(wire.System) != 2 {
		t.Fatalf("system blocks = %#v", wire.System)
	}
	if wire.System[0].CacheControl != nil {
		t.Fatalf("first system block was marked: %#v", wire.System)
	}
	if wire.System[1].CacheControl == nil || wire.System[1].CacheControl.TTL != "1h" {
		t.Fatalf("last system block cache marker = %#v", wire.System)
	}
	if len(wire.Tools) != 2 {
		t.Fatalf("tools = %#v", wire.Tools)
	}
	if wire.Tools[0].CacheControl != nil {
		t.Fatalf("first tool was marked: %#v", wire.Tools)
	}
	if wire.Tools[1].CacheControl == nil || wire.Tools[1].CacheControl.TTL != "1h" {
		t.Fatalf("last tool cache marker = %#v", wire.Tools)
	}
	if len(wire.Messages) != 1 || len(wire.Messages[0].Content) != 1 {
		t.Fatalf("message content = %#v", wire.Messages)
	}
	if wire.Messages[0].Content[0].CacheControl != nil {
		t.Fatalf("message content was marked: %#v", wire.Messages)
	}
}

func assertPromptCacheReceipt(
	t *testing.T,
	ctx *RequestContext,
	action string,
	inserted int,
	preserved int,
) {
	t.Helper()
	if ctx.PromptCacheAction != action {
		t.Fatalf("prompt cache action = %q, want %q", ctx.PromptCacheAction, action)
	}
	if ctx.PromptCacheInserted != inserted {
		t.Fatalf("inserted markers = %d, want %d", ctx.PromptCacheInserted, inserted)
	}
	if ctx.PromptCachePreserved != preserved {
		t.Fatalf("preserved markers = %d, want %d", ctx.PromptCachePreserved, preserved)
	}
}

func TestEncodeDispatchRequestPreservesCallerPromptCacheMarkers(t *testing.T) {
	request := promptCacheTestRequest()
	request.Instructions[0].Content[0].Cache = &llmprotocol.CacheDirective{
		Type: "ephemeral",
		TTL:  "1h",
	}
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.AnthropicMessagesV1,
		SemanticRequest:     request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{"enabled": true}),
	}

	if _, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx); err != nil {
		t.Fatalf("encode dispatch request: %v", err)
	}

	if request.Instructions[0].Content[0].Cache.TTL != "1h" ||
		request.Instructions[0].Content[1].Cache != nil ||
		request.Tools[0].Cache != nil ||
		request.Tools[1].Cache != nil {
		t.Fatalf("caller marker ownership changed: %#v", request)
	}
	if ctx.PromptCacheAction != promptCacheActionPreserved ||
		ctx.PromptCacheReason != promptCacheReasonCallerMarkers ||
		ctx.PromptCacheInserted != 0 || ctx.PromptCachePreserved != 1 {
		t.Fatalf("prompt cache receipt = %#v", ctx)
	}
	if request.Generation != 1 {
		t.Fatalf("preserved request generation = %d, want 1", request.Generation)
	}
}

func TestEncodeDispatchRequestDisabledPromptCachePreservesRequest(t *testing.T) {
	request := promptCacheTestRequest()
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.AnthropicMessagesV1,
		SemanticRequest:     request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{}),
	}

	if _, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx); err != nil {
		t.Fatalf("encode dispatch request: %v", err)
	}
	if countPromptCacheMarkers(*request) != 0 {
		t.Fatalf("disabled prompt cache added markers: %#v", request)
	}
	if ctx.PromptCacheAction != "" || request.Generation != 1 {
		t.Fatalf("disabled prompt cache changed request state: %#v", ctx)
	}
}

func TestEncodeDispatchRequestSkipsPromptCacheWithoutEligibleTargets(t *testing.T) {
	request := promptCacheTestRequest()
	request.Instructions = nil
	request.Tools = nil
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.AnthropicMessagesV1,
		SemanticRequest:     request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{"enabled": true}),
	}

	if _, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx); err != nil {
		t.Fatalf("encode dispatch request: %v", err)
	}
	if ctx.PromptCacheAction != promptCacheActionSkipped ||
		ctx.PromptCacheReason != promptCacheReasonNoEligibleTarget ||
		request.Generation != 1 {
		t.Fatalf("prompt cache receipt = %#v", ctx)
	}
}

func TestInjectInstructionPromptCacheMarkerSkipsEmptyText(t *testing.T) {
	request := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{{
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "stable instructions"},
				{Kind: llmprotocol.ContentText},
			},
		}},
	}

	if !injectInstructionPromptCacheMarker(request, "1h") {
		t.Fatal("expected non-empty instruction to receive a cache marker")
	}
	if request.Instructions[0].Content[0].Cache == nil ||
		request.Instructions[0].Content[0].Cache.TTL != "1h" ||
		request.Instructions[0].Content[1].Cache != nil {
		t.Fatalf("instruction cache markers = %#v", request.Instructions[0].Content)
	}

	request.Instructions[0].Content = []llmprotocol.Content{{
		Kind: llmprotocol.ContentText,
	}}
	if injectInstructionPromptCacheMarker(request, "1h") {
		t.Fatalf("empty instruction received a cache marker: %#v", request.Instructions)
	}
}

func TestEncodeDispatchRequestSkipsUnsupportedPromptCacheTarget(t *testing.T) {
	request := promptCacheTestRequest()
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.OpenAIChatV1,
		SemanticRequest:     request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{"enabled": true}),
	}

	if _, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx); err != nil {
		t.Fatalf("fail-open prompt cache policy: %v", err)
	}
	if ctx.PromptCacheAction != promptCacheActionSkipped ||
		ctx.PromptCacheReason != promptCacheReasonUnsupportedTarget {
		t.Fatalf("prompt cache receipt = %#v", ctx)
	}
	if request.Generation != 1 {
		t.Fatalf("unsupported target mutated request generation: %d", request.Generation)
	}
}

func TestEncodeDispatchRequestRejectsUnsupportedPromptCacheTarget(t *testing.T) {
	request := promptCacheTestRequest()
	ctx := &RequestContext{
		SourceFormat:    llmprotocol.OpenAIChatV1,
		TargetFormat:    llmprotocol.OpenAIResponsesV1,
		SemanticRequest: request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{
			"enabled":        true,
			"on_unsupported": "reject",
		}),
	}

	_, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) ||
		protocolError.Category != llmprotocol.ErrorUnsupportedFeature ||
		protocolError.Code != promptCacheErrorTargetUnsupported {
		t.Fatalf("error = %T %v", err, err)
	}
	if ctx.PromptCacheAction != promptCacheActionRejected {
		t.Fatalf("prompt cache receipt = %#v", ctx)
	}
}

func TestFinalizeProviderDispatchResponsePreservesPromptCacheProtocolError(t *testing.T) {
	request := promptCacheTestRequest()
	ctx := &RequestContext{
		Headers:         map[string]string{headers.VSRDebug: "true"},
		SourceFormat:    llmprotocol.OpenAIChatV1,
		TargetFormat:    llmprotocol.OpenAIChatV1,
		SemanticRequest: request,
		VSRSelectedDecision: promptCacheTestDecision(map[string]interface{}{
			"enabled":        true,
			"on_unsupported": "reject",
		}),
	}
	router := &OpenAIRouter{}
	response := buildRequestBodyContinueResponse(&routeHeaderState{}, nil, false)
	_, err := router.finalizeProviderDispatchResponse(
		&providerDispatch{
			logicalModel: "model-a",
			targetFormat: llmprotocol.OpenAIChatV1,
		},
		response,
		ctx,
	)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) ||
		protocolError.Code != promptCacheErrorTargetUnsupported {
		t.Fatalf("finalize error = %T %v", err, err)
	}

	immediate, converted := router.processBodyRoutingError(err, ctx)
	if !converted {
		t.Fatal("prompt cache protocol error was not converted")
	}
	immediate = router.encodeImmediateResponseForClient(immediate, ctx)
	if got := immediate.GetImmediateResponse().GetStatus().GetCode(); got != 400 {
		t.Fatalf("immediate status = %d, want 400", got)
	}
	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.GetImmediateResponse().GetBody(), &envelope); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if envelope.Error.Code != promptCacheErrorTargetUnsupported {
		t.Fatalf("error code = %q", envelope.Error.Code)
	}
	if got := immediateHeaderValue(immediate, headers.VSRPromptCacheAction); got != promptCacheActionRejected {
		t.Fatalf("prompt cache action header = %q", got)
	}
	if got := immediateHeaderValue(immediate, headers.VSRPromptCacheReason); got != promptCacheReasonUnsupportedTarget {
		t.Fatalf("prompt cache reason header = %q", got)
	}

	nonDebug := router.createErrorResponse(400, protocolError.Message)
	addPromptCacheReceiptToImmediateResponse(nonDebug, &RequestContext{
		PromptCacheAction: promptCacheActionRejected,
		PromptCacheReason: promptCacheReasonUnsupportedTarget,
	})
	if got := immediateHeaderValue(nonDebug, headers.VSRPromptCacheAction); got != "" {
		t.Fatalf("non-debug prompt cache action header = %q", got)
	}
}

func promptCacheTestRequest() *llmprotocol.Request {
	maxTokens := int64(16)
	return &llmprotocol.Request{
		Generation: 1,
		Model:      "anthropic-model",
		Instructions: []llmprotocol.InstructionBlock{{
			Role: llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "stable preface"},
				{Kind: llmprotocol.ContentText, Text: "reusable instructions"},
			},
		}},
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
		Tools: []llmprotocol.Tool{
			{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "search", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Sampling: llmprotocol.Sampling{MaxOutputTokens: &maxTokens},
	}
}

func promptCacheTestDecision(configuration map[string]interface{}) *config.Decision {
	return &config.Decision{
		Name: "prompt-cache-route",
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginPromptCache,
			Configuration: config.MustStructuredPayload(configuration),
		}},
	}
}
