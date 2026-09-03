package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestApplySemanticContextCompressionCompressesToolResult(t *testing.T) {
	largeTool := strings.Join([]string{
		"source header",
		strings.Repeat("irrelevant inventory values ", 180),
		"authentication token validator failed",
		strings.Repeat("irrelevant billing values ", 180),
		"source footer",
	}, "\n")
	request := semanticCompressionRequest(largeTool)
	ctx := &RequestContext{
		RequestID:           "req-compress",
		VSRSelectedDecision: contextCompressionTestDecision(false),
	}

	if err := (&OpenAIRouter{}).applySemanticContextCompression(ctx, request); err != nil {
		t.Fatalf("apply semantic context compression: %v", err)
	}
	got := request.Messages[2].Content[0].ToolResult.Content[0].Text
	if got == largeTool || len(got) >= len(largeTool) {
		t.Fatal("large neutral tool result was not compressed")
	}
	if !strings.Contains(got, "authentication token validator") {
		t.Fatalf("query-relevant tool content was removed: %s", got)
	}
	if !ctx.ContextCompressionApplied || ctx.ContextCompressionMessages != 1 {
		t.Fatalf("compression diagnostics missing: %#v", ctx)
	}
	if ctx.ContextCompressionAfter >= ctx.ContextCompressionBefore || request.Generation != 2 {
		t.Fatalf("semantic mutation was not versioned: generation=%d before=%d after=%d", request.Generation, ctx.ContextCompressionBefore, ctx.ContextCompressionAfter)
	}
}

func TestApplySemanticContextCompressionHonorsRequestControl(t *testing.T) {
	request := semanticCompressionRequest(strings.Repeat("large output ", 300))
	ctx := &RequestContext{
		Headers: map[string]string{"x-compression-control": "bypass"},
		VSRSelectedDecision: contextCompressionDecision(map[string]interface{}{
			"request_controls": map[string]interface{}{
				"enabled": true,
				"header":  "x-compression-control",
				"allowed": []string{"bypass"},
			},
		}),
	}
	original := request.Messages[2].Content[0].ToolResult.Content[0].Text

	if err := (&OpenAIRouter{}).applySemanticContextCompression(ctx, request); err != nil {
		t.Fatalf("bypassed compression returned error: %v", err)
	}
	if got := request.Messages[2].Content[0].ToolResult.Content[0].Text; got != original {
		t.Fatalf("bypassed semantic request changed: %q", got)
	}
	if ctx.ContextCompressionApplied || ctx.ContextCompressionSkipReason != "bypassed" {
		t.Fatalf("unexpected bypass diagnostics: %#v", ctx)
	}
}

func TestApplySemanticContextCompressionPreservesOrderedNonTextBlocks(t *testing.T) {
	jsonTool := `{"status":"authentication validator failed","logs":"` + strings.Repeat("x", 5000) + `"}`
	request := semanticCompressionRequest(jsonTool)
	result := request.Messages[2].Content[0].ToolResult
	result.Content = append(result.Content, llmprotocol.Content{
		Kind: llmprotocol.ContentImage, URL: "https://example.com/a.png", MediaType: "image/png",
	})
	ctx := &RequestContext{VSRSelectedDecision: contextCompressionTestDecision(false)}

	if err := (&OpenAIRouter{}).applySemanticContextCompression(ctx, request); err != nil {
		t.Fatalf("compress JSON tool result: %v", err)
	}
	if len(result.Content) != 2 || result.Content[1].Kind != llmprotocol.ContentImage || result.Content[1].URL != "https://example.com/a.png" {
		t.Fatalf("ordered non-text content changed: %#v", result.Content)
	}
	if ctx.ContextCompressionFormat != "json" {
		t.Fatalf("compression format = %q", ctx.ContextCompressionFormat)
	}
}

func TestApplySemanticContextCompressionProtectsRAGUnlessOptedIn(t *testing.T) {
	largeTool := strings.Repeat("retrieved authentication context ", 300)
	protected := semanticCompressionRequest(largeTool)
	protectedCtx := &RequestContext{
		RAGToolCallIDs:      map[string]struct{}{"call_1": {}},
		VSRSelectedDecision: contextCompressionTestDecision(false),
	}
	if err := (&OpenAIRouter{}).applySemanticContextCompression(protectedCtx, protected); err != nil {
		t.Fatalf("protected RAG compression: %v", err)
	}
	if got := protected.Messages[2].Content[0].ToolResult.Content[0].Text; got != largeTool {
		t.Fatal("RAG tool result changed without explicit opt-in")
	}

	optedIn := semanticCompressionRequest(largeTool)
	optedInCtx := &RequestContext{
		RAGToolCallIDs:      map[string]struct{}{"call_1": {}},
		VSRSelectedDecision: contextCompressionTestDecision(true),
	}
	if err := (&OpenAIRouter{}).applySemanticContextCompression(optedInCtx, optedIn); err != nil {
		t.Fatalf("opted-in RAG compression: %v", err)
	}
	if got := optedIn.Messages[2].Content[0].ToolResult.Content[0].Text; got == largeTool {
		t.Fatal("RAG tool result was not compressed after opt-in")
	}
}

func TestPrepareRequestForModelRoutingMutatesNeutralRequestOnly(t *testing.T) {
	largeTool := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " + strings.Repeat("irrelevant billing ", 300)
	request := semanticCompressionRequest(largeTool)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		SemanticRequest:     request,
		VSRSelectedDecision: contextCompressionTestDecision(false),
	}

	parsed, early, err := (&OpenAIRouter{Config: &config.RouterConfig{}}).prepareRequestForModelRouting(
		request, "fix authentication validator", ctx,
	)
	if err != nil || early != nil {
		t.Fatalf("prepare request failed: early=%v err=%v", early, err)
	}
	if parsed != request || ctx.SemanticRequest != request {
		t.Fatal("prepare request replaced the request-scoped neutral IR")
	}
	if strings.Contains(request.Messages[2].Content[0].ToolResult.Content[0].Text, strings.Repeat("irrelevant inventory ", 30)) {
		t.Fatal("provider-neutral request was not compressed")
	}
}

func TestSemanticContextCompressionFailClosedReturnsError(t *testing.T) {
	decision := &config.Decision{
		Name: "fail-closed",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true, "failure_mode": "fail_closed",
			}),
		}},
	}
	var unavailableRouter *OpenAIRouter
	if err := unavailableRouter.applySemanticContextCompression(
		&RequestContext{VSRSelectedDecision: decision}, semanticCompressionRequest("large output"),
	); err == nil {
		t.Fatal("fail_closed compression accepted an unavailable service")
	}
}

func TestInjectSemanticContextRecoveryToolRejectsReservedNameConflict(t *testing.T) {
	request := &llmprotocol.Request{Tools: []llmprotocol.Tool{{Name: contextcompression.RetrieveToolName}}}
	if err := injectSemanticContextRecoveryTool(request, []string{"issued-key"}); err == nil {
		t.Fatal("reserved recovery tool conflict was accepted")
	}
}

func semanticCompressionRequest(toolOutput string) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Model:      "auto",
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "collect diagnostic data"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "call_1", Name: "diagnostics", Arguments: `{"service":"auth"}`,
			}}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
				CallID: "call_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: toolOutput}},
			}}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "I reviewed the diagnostic output."}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "fix authentication validator"}}},
		},
	}
}

func contextCompressionTestDecision(compressRAG bool) *config.Decision {
	ragMode := "preserve"
	if compressRAG {
		ragMode = "extractive"
	}
	return contextCompressionDecision(map[string]interface{}{
		"targets": map[string]interface{}{
			"tool_outputs": map[string]interface{}{
				"mode": "extractive", "min_tokens": 200, "target_tokens": 160,
			},
			"rag": map[string]interface{}{"mode": ragMode},
		},
	})
}

func contextCompressionDecision(extra map[string]interface{}) *config.Decision {
	configuration := map[string]interface{}{"enabled": true}
	for key, value := range extra {
		configuration[key] = value
	}
	return &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(configuration),
		}},
	}
}
