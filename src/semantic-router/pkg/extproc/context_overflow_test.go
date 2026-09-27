package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func overflowFixture(t *testing.T, text string) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	r, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	r.Config.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
	params := r.Config.ModelConfig[model]
	params.Capabilities = []string{"chat", "tools"}
	params.ContextWindowSize = 32768
	params.MaxOutputTokens = 8192
	r.Config.ModelConfig[model] = params
	payload, err := config.NewStructuredPayload(map[string]any{"enabled": true, "targets": map[string]any{"current_user": map[string]any{"mode": "truncate"}, "history": map[string]any{"mode": "extractive", "min_tokens": 2000, "target_tokens": 1000}, "tool_outputs": map[string]any{"mode": "preserve"}}})
	if err != nil {
		t.Fatal(err)
	}
	d := &config.Decision{Name: "bounded", ModelRefs: []config.ModelRef{{Model: model}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic}, Plugins: []config.DecisionPlugin{{Type: "context_compression", Configuration: payload}}}
	request := testNeutralRequest("auto", text)
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8192)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = d
	ctx.TraceContext = context.Background()
	return r, ctx
}

func TestContextOverflowSelectionAndEncodedDispatchUseReducedRequest(t *testing.T) {
	for _, item := range []struct{ heading, filler string }{
		{"HEAD instruction", strings.Repeat(" a", 40000)},
		{"HEAD instruction", strings.Repeat(" padding", 40000)},
		{"HEAD instruction", strings.Repeat("中文🙂", 20000)},
		{"HEAD instruction", strings.Repeat("x", 160000)},
		{"[Request heading]", strings.Repeat("archive text ", 30000)},
		{"{Request heading}", strings.Repeat("archive text ", 30000)},
	} {
		original := item.heading + "\n" + item.filler + "\nTAIL instruction"
		r, ctx := overflowFixture(t, original)
		d := ctx.VSRSelectedDecision
		name, _, _, model, err := r.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: d, Confidence: 1}, "auto", original, ctx)
		if err != nil || model == "" || name != d.Name {
			t.Fatalf("selection: model=%q err=%v", model, err)
		}
		if _, response, prepareErr := r.prepareRequestForModelRouting(ctx.SemanticRequest, original, ctx); prepareErr != nil || response != nil {
			t.Fatalf("normal plugin preparation failed: %v response=%v", prepareErr, response != nil)
		}
		dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, model, d.Name, false, ctx)
		if err != nil || dispatch == nil {
			t.Fatalf("dispatch: %v", err)
		}
		body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(*ctx.SemanticRequest, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
		if err != nil {
			t.Fatal(err)
		}
		outbound, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
		if err != nil {
			t.Fatal(err)
		}
		got := outbound.Messages[0].Content[0].Text
		if !utf8.ValidString(got) || !strings.HasPrefix(got, item.heading+"\n") || !strings.HasSuffix(got, "\nTAIL instruction") || !strings.Contains(got, "context omitted by route compression") || got == original {
			t.Fatalf("invalid outbound truncation: bytes=%d", len(got))
		}
		bound, _ := (overflowTokenCounter{}).CountRequest(model, &contextcompression.RequestIR{Semantic: &outbound})
		if bound+int(*outbound.Sampling.MaxOutputTokens) > 32768 {
			t.Fatalf("outbound bound=%d + output=%d", bound, *outbound.Sampling.MaxOutputTokens)
		}
		if !ctx.ContextCompressionApplied || ctx.ContextCompressionTokenSource != "utf8_byte_upper_bound" || ctx.ContextCompressionAfter >= ctx.ContextCompressionBefore {
			t.Fatal("missing truthful receipt")
		}
		if _, err = r.decisionEligibleModelRefs(d, ctx); err != nil {
			t.Fatalf("reduced body fails unchanged eligibility: %v", err)
		}
	}
}

func TestContextOverflowPreservesSystemToolsAndLatestInstruction(t *testing.T) {
	r, ctx := overflowFixture(t, "Return BLUE43:42")
	request := ctx.SemanticRequest
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Required code BLUE43"}}}}
	request.Tools = []llmprotocol.Tool{{Name: "calculator", InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}}}`)}}
	latest := request.Messages[0]
	request.Messages = []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat("old facts ", 30000)}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "calculator", Arguments: `{"a":19,"b":23}`}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "42"}}}}}},
		latest,
	}
	original, err := selection.EffectiveCandidateRequest(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = r.prepareDecisionContextOverflow(ctx, "auto"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(request.Instructions, original.Instructions) || !reflect.DeepEqual(request.Tools, original.Tools) || !reflect.DeepEqual(request.Messages[1:], original.Messages[1:]) {
		t.Fatal("instruction/tool exchange/current short instruction changed")
	}
	if request.Messages[0].Content[0].Text == original.Messages[0].Content[0].Text {
		t.Fatal("old history unchanged")
	}
}

func TestContextOverflowRecompressesBracketHistoryForToolFollowups(t *testing.T) {
	for _, heading := range []string{"[Request heading]", "{Request heading}"} {
		t.Run(heading, func(t *testing.T) {
			original := heading + "\nHeader MAPLE731\n" + strings.Repeat("Unrelated archival paragraph. ", 15000) + "\nReturn MAPLE731:42."
			firstRouter, firstContext := overflowFixture(t, original)
			first := encodedOverflowDispatch(t, firstRouter, firstContext, original)
			if first.Messages[0].Content[0].Text == original {
				t.Fatal("first turn did not reduce current-user text")
			}
			// Playground retains the original user text, not the first dispatch's
			// lossy view, when it constructs the next turn and tool continuation.
			for _, continuation := range []bool{false, true} {
				latest := "Use calculator to compute 997*991."
				r, ctx := overflowFixture(t, latest)
				request := ctx.SemanticRequest
				request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Use tools accurately."}}}}
				request.Tools = []llmprotocol.Tool{{Name: "calculator", InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}}}`)}}
				request.Messages = append([]llmprotocol.Message{
					{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: original}}},
					{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "MAPLE731:42"}}},
				}, request.Messages...)
				if continuation {
					request.Messages = append(request.Messages,
						llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_calculator", Name: "calculator", Arguments: `{"expression":"997*991"}`}}}},
						llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_calculator", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "988027"}}}}}})
				}
				before, err := selection.EffectiveCandidateRequest(request, nil)
				if err != nil {
					t.Fatal(err)
				}
				outbound := encodedOverflowDispatch(t, r, ctx, latest)
				if got := outbound.Messages[0].Content[0].Text; got == original || !strings.Contains(got, "context omitted by route compression") {
					t.Fatal("old bracket-heading history was not compressed")
				}
				if !reflect.DeepEqual(request.Instructions, before.Instructions) || !reflect.DeepEqual(request.Tools, before.Tools) || !reflect.DeepEqual(request.Messages[1:], before.Messages[1:]) {
					t.Fatal("history compression changed instructions, schema, latest user or tool exchange")
				}
				if !ctx.ContextCompressionApplied || ctx.ContextCompressionAfter >= ctx.ContextCompressionBefore {
					t.Fatal("missing history compression receipt")
				}
			}
		})
	}
}

func encodedOverflowDispatch(t *testing.T, router *OpenAIRouter, ctx *RequestContext, input string) llmprotocol.Request {
	t.Helper()
	d := ctx.VSRSelectedDecision
	_, _, _, model, err := router.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: d, Confidence: 1}, "auto", input, ctx)
	if err != nil || model == "" {
		t.Fatalf("selection: model=%q err=%v", model, err)
	}
	if _, response, prepareErr := router.prepareRequestForModelRouting(ctx.SemanticRequest, input, ctx); prepareErr != nil || response != nil {
		t.Fatalf("plugin preparation: %v response=%v", prepareErr, response != nil)
	}
	if dispatch, dispatchErr := router.prepareProviderDispatch(ctx.SemanticRequest, model, d.Name, false, ctx); dispatchErr != nil || dispatch == nil {
		t.Fatalf("dispatch: %v", dispatchErr)
	}
	body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(*ctx.SemanticRequest, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	outbound, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	bound, _ := (overflowTokenCounter{}).CountRequest(model, &contextcompression.RequestIR{Semantic: &outbound})
	if bound+int(*outbound.Sampling.MaxOutputTokens) > 32768 {
		t.Fatalf("dispatch exceeds context: input bound=%d", bound)
	}
	return outbound
}

func TestContextOverflowRejectsUntrimmableBudgetWithoutMutation(t *testing.T) {
	for _, kind := range []string{"system", "schema", "json", "tool_result", "multimodal", "authorization"} {
		t.Run(kind, func(t *testing.T) {
			r, ctx := overflowFixture(t, strings.Repeat(" a", 40000))
			req := ctx.SemanticRequest
			switch kind {
			case "system":
				req.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat("mandatory ", 10000)}}}}
			case "schema":
				req.Tools = []llmprotocol.Tool{{Name: "required_tool", InputSchema: json.RawMessage(`{"type":"object","description":"` + strings.Repeat(" a", 40000) + `"}`)}}
			case "json":
				req.Messages[0].Content[0].Text = `{"data":"` + req.Messages[0].Content[0].Text + `"}`
			case "tool_result":
				req.Messages = append(req.Messages,
					llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{}`}}}},
					llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat(" a", 40000)}}}}}})
			case "multimodal":
				model := ctx.VSRSelectedDecision.ModelRefs[0].Model
				params := r.Config.ModelConfig[model]
				params.Capabilities = append(params.Capabilities, "vision")
				r.Config.ModelConfig[model] = params
				req.Messages[0].Content = append(req.Messages[0].Content, llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
			case "authorization":
				ctx.ProtectedContextMessages = map[int]contextcompression.Protection{0: contextcompression.ProtectAuthorization}
			}
			before, _ := json.Marshal(req)
			err := r.prepareDecisionContextOverflow(ctx, "auto")
			var budgetErr *selection.RequestBudgetError
			if !errors.As(err, &budgetErr) {
				t.Fatalf("not client budget error: %v", err)
			}
			after, _ := json.Marshal(req)
			if string(after) != string(before) {
				t.Fatal("rejected request partially mutated")
			}
		})
	}
}

func TestContextOverflowDefaultsAndBypassKeepRequest(t *testing.T) {
	for _, setting := range []string{"disabled", "preserve", "bypass", "short"} {
		t.Run(setting, func(t *testing.T) {
			r, ctx := overflowFixture(t, strings.Repeat(" a", 40000))
			cfg := map[string]any{"enabled": true, "targets": map[string]any{"current_user": map[string]any{"mode": "truncate"}}}
			switch setting {
			case "disabled":
				cfg["enabled"] = false
			case "preserve":
				cfg["targets"] = map[string]any{"current_user": map[string]any{"mode": "preserve"}}
			case "bypass":
				cfg["request_controls"] = map[string]any{"enabled": true, "allowed": []string{"bypass"}}
				ctx.Headers = map[string]string{"x-vsr-compression-control": "bypass"}
			case "short":
				ctx.SemanticRequest.Messages[0].Content[0].Text = "hello"
			}
			payload, _ := config.NewStructuredPayload(cfg)
			ctx.VSRSelectedDecision.Plugins[0].Configuration = payload
			before, _ := json.Marshal(ctx.SemanticRequest)
			if err := r.prepareDecisionContextOverflow(ctx, "auto"); err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(ctx.SemanticRequest)
			if string(before) != string(after) {
				t.Fatal("unexpected opt-out/short mutation")
			}
		})
	}
}

func TestContextOverflowRechecksActualDispatchGrowth(t *testing.T) {
	r, ctx := overflowFixture(t, "hello")
	if err := r.prepareDecisionContextOverflow(ctx, "auto"); err != nil {
		t.Fatal(err)
	}
	model := ctx.VSRSelectedDecision.ModelRefs[0].Model
	ctx.SemanticRequest.Messages[0].Content[0].Text = "HEAD " + strings.Repeat(" a", 40000) + " TAIL"
	if _, err := r.prepareProviderDispatch(ctx.SemanticRequest, model, ctx.VSRSelectedDecision.Name, false, ctx); err != nil {
		t.Fatal(err)
	}
	if !ctx.ContextCompressionApplied {
		t.Fatal("late dispatch growth bypassed preparation")
	}
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"description":"` + strings.Repeat(" a", 40000) + `"}`)}}
	if _, err := r.prepareProviderDispatch(ctx.SemanticRequest, model, ctx.VSRSelectedDecision.Name, false, ctx); err == nil {
		t.Fatal("late immutable schema bypassed budget")
	}
}

func TestContextOverflowEffectivePolicyAndRecipeOptInStayIsolated(t *testing.T) {
	r, ctx := overflowFixture(t, "HEAD "+strings.Repeat(" a", 40000)+" TAIL")
	ctx.SemanticRequest.Sampling.MaxOutputTokens = nil
	prompt := strings.Repeat("system rule ", 250)
	system, _ := config.NewStructuredPayload(map[string]any{"system_prompt": prompt})
	params, _ := config.NewStructuredPayload(map[string]any{"default_max_tokens": 8192})
	ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: "system_prompt", Configuration: system}, config.DecisionPlugin{Type: "request_params", Configuration: params})
	untouched, err := selection.EffectiveCandidateRequest(ctx.SemanticRequest, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = r.prepareDecisionContextOverflow(ctx, "auto"); err != nil {
		t.Fatal(err)
	}
	if len(ctx.SemanticRequest.Instructions) != 0 || ctx.SemanticRequest.Sampling.MaxOutputTokens != nil {
		t.Fatal("deterministic policy preview mutated original instruction/output fields")
	}
	effective, err := selection.EffectiveCandidateRequest(ctx.SemanticRequest, ctx.VSRSelectedDecision)
	if err != nil {
		t.Fatal(err)
	}
	bound, _ := (overflowTokenCounter{}).CountRequest("model", &contextcompression.RequestIR{Semantic: effective})
	if bound+8192 > 32768 {
		t.Fatal("effective system/output not reserved")
	}
	other := &RequestContext{SemanticRequest: untouched, VSRSelectedDecision: &config.Decision{Name: "other", ModelRefs: ctx.VSRSelectedDecision.ModelRefs}}
	before, _ := json.Marshal(untouched)
	if err = r.prepareDecisionContextOverflow(other, "auto"); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(untouched)
	if string(before) != string(after) {
		t.Fatal("opt-in leaked between decisions/recipes")
	}
}

func TestContextOverflowRejectsFramingHeavyRequest(t *testing.T) {
	for _, kind := range []string{"messages", "tools"} {
		t.Run(kind, func(t *testing.T) {
			r, ctx := overflowFixture(t, "hello")
			switch kind {
			case "messages":
				for range 1000 {
					ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a"}}})
				}
			case "tools":
				for range 256 {
					ctx.SemanticRequest.Tools = append(ctx.SemanticRequest.Tools, llmprotocol.Tool{Name: "tool", InputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`)})
				}
			}
			var budget *selection.RequestBudgetError
			if err := r.prepareDecisionContextOverflow(ctx, "auto"); !errors.As(err, &budget) {
				t.Fatalf("many %s escaped framing budget: %v", kind, err)
			}
		})
	}
}

func TestContextOverflowPreparedStreamCompletesAndReplays(t *testing.T) {
	r, ctx := overflowFixture(t, "HEAD "+strings.Repeat(" a", 40000)+" TAIL")
	ctx.SemanticRequest.Stream = true
	ctx.ExpectStreamingResponse = true
	d := ctx.VSRSelectedDecision
	_, _, _, model, err := r.finalizeDecisionEvaluation(&decision.DecisionResult{Decision: d, Confidence: 1}, "auto", "original routing input", ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = r.prepareProviderDispatch(ctx.SemanticRequest, model, d.Name, false, ctx); err != nil {
		t.Fatal(err)
	}
	if !ctx.SemanticRequest.Stream || !ctx.ContextCompressionApplied {
		t.Fatal("stream request lost overflow preparation")
	}
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	r.ReplayRecorder = recorder
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.CaptureResponseBody = true
	ctx.RouterReplayPluginConfig = &replayConfig
	ctx.RequestID = "overflow-stream"
	ctx.UpstreamStatusCode = 200
	ctx.IsStreamingResponse = true
	r.startRouterReplay(ctx, "auto", model, d.Name)
	if _, err = r.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}, {Key: "content-type", Value: "text/event-stream"}}}}}, ctx); err != nil {
		t.Fatal(err)
	}
	body := extProcStreamFixture(llmprotocol.OpenAIChatV1)
	response := r.handleSemanticStreamingResponseBody(body, true, ctx)
	publicBody := body
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		publicBody = mutation.GetBody()
	}
	if !strings.Contains(string(publicBody), "data: [DONE]") || ctx.SemanticResponse == nil {
		t.Fatal("prepared request did not reach a successful terminal stream")
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.LifecycleState != routerreplay.LifecycleCompleted || record.ResponseStatus != 200 {
		t.Fatalf("stream replay: found=%v state=%s status=%d reason=%s", ok, record.LifecycleState, record.ResponseStatus, record.TerminalReason)
	}
}

func TestContextOverflowCompressesMultipleOldUserAndAssistantTurns(t *testing.T) {
	r, ctx := overflowFixture(t, "Return only BLUE43:42")
	latest := ctx.SemanticRequest.Messages[0]
	messages := make([]llmprotocol.Message, 0, 13)
	for range 6 {
		messages = append(messages,
			llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Older user context " + strings.Repeat(" a", 4000)}}},
			llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Older assistant context " + strings.Repeat(" a", 4000)}}})
	}
	ctx.SemanticRequest.Messages = messages
	ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, latest)
	before, err := selection.EffectiveCandidateRequest(ctx.SemanticRequest, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = r.prepareDecisionContextOverflow(ctx, "auto"); err != nil {
		t.Fatal(err)
	}
	userChanged, assistantChanged := false, false
	for i, message := range ctx.SemanticRequest.Messages[:len(messages)-1] {
		if message.Content[0].Text != before.Messages[i].Content[0].Text {
			switch message.Role {
			case llmprotocol.RoleUser:
				userChanged = true
			case llmprotocol.RoleAssistant:
				assistantChanged = true
			}
		}
	}
	if !userChanged || !assistantChanged {
		t.Fatal("old user and assistant history did not both compress")
	}
	if !reflect.DeepEqual(ctx.SemanticRequest.Messages[len(messages):], before.Messages[len(messages):]) {
		t.Fatal("short current instruction changed")
	}
	count, _ := (overflowTokenCounter{}).CountRequest("model", &contextcompression.RequestIR{Semantic: ctx.SemanticRequest})
	if count+8192 > 32768 {
		t.Fatalf("multi-turn dispatch bound=%d", count)
	}
}
