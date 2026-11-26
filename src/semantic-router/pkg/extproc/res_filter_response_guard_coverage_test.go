package extproc

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const (
	inventoryRule = "inventory_check"
	inventoryKey  = "systemone:" + inventoryRule
	inventoryText = "The stockroom has twelve notebooks."
)

func inventoryToolResult(callID string, blocks ...llmprotocol.Content) llmprotocol.Content {
	return llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: callID, Content: blocks}}
}

func inventoryRequest(contents ...llmprotocol.Content) *llmprotocol.Request {
	return &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleTool, Content: contents}}}
}

func inventoryResponseGuard(t *testing.T, input config.SystemOneInput, coverage, onUnscanned, onUnknown string) (*OpenAIRouter, *RequestContext, *[]modelservice.Request) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = []config.SystemOneRule{{
		Name: inventoryRule, Stage: config.SystemOneStageResponse, Coverage: coverage, OnUnscanned: onUnscanned, Input: input,
		Question: config.SystemOneQuestion{Type: config.SystemOneQuestionNoul, Instructions: "Does the answer report the inventory count?"},
	}}
	cfg.Decisions = []config.Decision{keywordDecision(responseStageRouteName, guardPlugin(config.ResponseGuardPluginConfig{
		OnUnknown: onUnknown,
		Rules: []config.ResponseGuardRule{{
			When: config.RuleNode{Type: config.SignalTypeSystemOne, Name: inventoryRule}, Action: config.ResponseGuardAnnotate,
		}},
	}))}
	var calls []modelservice.Request
	decider := nativeRoutingDecider(func(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
		calls = append(calls, request)
		answers := make(map[string]modelservice.Answer)
		for _, question := range request.Questions {
			answers[question.ID] = modelservice.Answer{Type: "noul", Noul: 0.95, InputCoverage: "complete"}
		}
		return modelservice.Response{Answers: answers}, nil
	})
	router, ctx := routerWithSelectedDecision(cfg, responseStageClassifier(t, cfg, decider), responseStageRouteName)
	return router, ctx, &calls
}

func TestResponseGuardGroundingCoverage(t *testing.T) {
	readable := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: inventoryText}
	unreadable := llmprotocol.Content{Kind: llmprotocol.ContentImage}
	complete := inventoryRequest(inventoryToolResult("inventory", readable))
	partial := inventoryRequest(inventoryToolResult("inventory", readable, unreadable))
	unread := inventoryRequest(inventoryToolResult("inventory", unreadable))
	const coverageError = "systemone_unreadable_tool_result"
	for _, tc := range []struct {
		name, onUnscanned, coverage, answer string
		request                             *llmprotocol.Request
		explicit, toolIncomplete            bool
		wantState, wantError                string
		wantCall                            bool
	}{
		{name: "complete", request: complete, answer: inventoryText, wantState: "true", wantCall: true},
		{name: "partial unknown", request: partial, answer: inventoryText, onUnscanned: config.SystemOneUnscannedUnknown, wantState: "unknown", wantError: coverageError, wantCall: true},
		{name: "partial match", request: partial, answer: inventoryText, onUnscanned: config.SystemOneUnscannedMatch, wantState: "true", wantCall: true},
		{name: "partial no match", request: partial, answer: inventoryText, onUnscanned: config.SystemOneUnscannedNoMatch, wantState: "false", wantCall: true},
		{name: "partial best effort", request: partial, answer: inventoryText, onUnscanned: config.SystemOneUnscannedBestEffort, wantState: "unknown", wantError: coverageError, wantCall: true},
		{name: "wholly unreadable match", request: unread, answer: inventoryText, onUnscanned: config.SystemOneUnscannedMatch, wantState: "true"},
		{name: "wholly unreadable no match", request: unread, answer: inventoryText, onUnscanned: config.SystemOneUnscannedNoMatch, wantState: "false"},
		{name: "wholly unreadable unknown", request: unread, answer: inventoryText, wantState: "unknown", wantError: coverageError},
		{name: "wholly unreadable best effort", request: unread, answer: inventoryText, onUnscanned: config.SystemOneUnscannedBestEffort, wantState: "unknown", wantError: coverageError},
		{name: "absent context", answer: inventoryText, onUnscanned: config.SystemOneUnscannedMatch, wantState: "unknown", wantError: "systemone_context_unavailable"},
		{name: "missing answer", request: unread, onUnscanned: config.SystemOneUnscannedMatch, wantState: "unknown", wantError: "systemone_answer_unavailable"},
		{name: "truncated context", request: partial, answer: inventoryText, coverage: config.SystemOneCoverageTruncate, wantState: "true", wantCall: true},
		{name: "explicit tool coverage", request: partial, answer: inventoryText, explicit: true, toolIncomplete: true, wantState: "unknown", wantError: coverageError, wantCall: true},
		{name: "complete explicit tool coverage", request: partial, answer: inventoryText, explicit: true, wantState: "true", wantCall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			coverage := tc.coverage
			if coverage == "" {
				coverage = config.SystemOneCoverageFull
			}
			input := config.SystemOneInput{Answer: true, Context: !tc.explicit, ToolResults: tc.explicit}
			router, ctx, calls := inventoryResponseGuard(t, input, coverage, tc.onUnscanned, config.ResponseGuardNone)
			ctx.SemanticRequest = tc.request
			ctx.ResponseStage.ToolResultsIncomplete = tc.toolIncomplete
			if tc.explicit {
				ctx.ResponseStage.ToolResultTexts = []string{inventoryText}
			}
			plan := guardAnswer(t, router, ctx, tc.answer)
			run := ctx.ResponseGuard
			if state := run.rules[0].state.String(); state != tc.wantState {
				t.Errorf("guard rule = %s, want %s", state, tc.wantState)
			}
			if reason := run.observed().Errors[inventoryKey]; reason != tc.wantError {
				t.Errorf("signal error = %q, want %q", reason, tc.wantError)
			}
			warnings := ""
			if tc.wantState == "true" {
				warnings = headers.WarningGuard
				if got := run.observed().Reported[config.SignalTypeSystemOne]; len(got) != 1 || got[0] != inventoryRule {
					t.Errorf("reported matches = %v, want [%s]", got, inventoryRule)
				}
			}
			assertDelivered(t, ctx, plan, warnings)
			if !tc.wantCall {
				if len(*calls) != 0 {
					t.Errorf("model calls = %d, want none", len(*calls))
				}
				return
			}
			part := "context"
			if tc.explicit {
				part = "tool_results"
			}
			if len(*calls) != 1 || (*calls)[0].Parts[part] != inventoryText {
				t.Errorf("model calls = %+v, want one with the readable %s", *calls, part)
			}
		})
	}
}

func TestResponseGuardGroundsOnTheBackendToolResult(t *testing.T) {
	const format = llmprotocol.AnthropicMessagesV1
	const pdfData = "JVBERi0="
	const fileNote = "A file (application/pdf) was not sent because the selected model cannot read files."
	const history = "The earlier count was twelve notebooks."
	const requestBody = `{
		"model":"inventory","max_tokens":64,"messages":[
			{"role":"user","content":"Read the inventory."},
			{"role":"assistant","content":[{"type":"tool_use","id":"read-1","name":"read_inventory","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"read-1","content":[
				{"type":"text","text":"` + history + `"},
				{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + pdfData + `"}}
			]}]},
			{"role":"assistant","content":"I have the inventory count."},
			{"role":"user","content":"Repeat the count."}
		]
	}`
	for _, tc := range []struct {
		name         string
		capabilities []string
		wantContext  string
		wantError    string
		fallback     bool
	}{
		{name: "file replaced with note", capabilities: []string{"tools"}, wantContext: history + " " + fileNote},
		{name: "file retained", capabilities: []string{"tools", "file_input"}, wantContext: history, wantError: "systemone_unreadable_tool_result"},
		{name: "fallback replaces the file", capabilities: []string{"tools", "file_input"}, wantContext: history + " " + fileNote, fallback: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := config.SystemOneInput{Answer: true, Context: true}
			router, ctx, calls := inventoryResponseGuard(t, input, config.SystemOneCoverageFull, config.SystemOneUnscannedUnknown, config.ResponseGuardBlock)
			router.Config.ModelConfig = map[string]config.ModelParams{"inventory": {APIFormat: config.APIFormatAnthropic, Capabilities: tc.capabilities}}
			router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldUnreadableFile}
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, []byte(requestBody))
			if err != nil {
				t.Fatal(err)
			}
			ctx.SemanticRequest, ctx.ProtocolEnvelope = &request, envelope
			ctx.SourceFormat, ctx.TargetFormat, ctx.RequestModel = format, format, request.Model
			prepared, err := router.materializePreparedProviderRequest(&providerDispatch{logicalModel: request.Model, targetFormat: format}, ctx)
			if err != nil {
				t.Fatal(err)
			}
			wantFile := tc.fallback || tc.wantError != ""
			if strings.Contains(string(prepared.body), pdfData) != wantFile || strings.Contains(string(prepared.body), fileNote) == wantFile {
				t.Fatalf("backend request = %s, want the original PDF: %t", prepared.body, wantFile)
			}
			var plan *responsePolicyPlan
			if tc.fallback {
				fallbackRouter, fallbackConfig := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
				router.FallbackOrchestrator = fallbackRouter.FallbackOrchestrator
				router.CredentialResolver = fallbackRouter.CredentialResolver
				router.Config.ProviderProfiles = fallbackConfig.ProviderProfiles
				router.Config.VLLMEndpoints = fallbackConfig.VLLMEndpoints
				candidate := fallbackConfig.ModelConfig["model-fallback-2"]
				candidate.Capabilities = []string{"tools"}
				router.Config.ModelConfig["model-fallback-2"] = candidate
				ctx.VSREligibleModelRefs = []config.ModelRef{{Model: request.Model}, {Model: "model-fallback-2"}}
				ctx.VSRSelectedModel = request.Model
				ctx.UpstreamStatusCode = http.StatusServiceUnavailable
				router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
					if model != "model-fallback-2" || strings.Contains(string(body), pdfData) || !strings.Contains(string(body), fileNote) {
						t.Fatalf("fallback %s request = %s, want the replacement note", model, body)
					}
					return []byte(`{"id":"msg_inventory","type":"message","role":"assistant","model":"model-fallback-2","content":[{"type":"text","text":"` + inventoryText + `"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":8}}`), http.StatusOK, nil
				}
				response := router.handleUpstreamTransportError([]byte("primary unavailable"), ctx)
				if got := response.GetImmediateResponse().GetStatus().GetCode(); got != http.StatusOK {
					t.Errorf("fallback status = %d, want 200", got)
				}
			} else {
				plan = guardAnswer(t, router, ctx, inventoryText)
			}
			if len(*calls) != 1 || (*calls)[0].Parts["context"] != tc.wantContext {
				t.Fatalf("model calls = %+v, want one with context %q", *calls, tc.wantContext)
			}
			if reason := ctx.ResponseGuard.observed().Errors[inventoryKey]; reason != tc.wantError {
				t.Fatalf("signal error = %q, want %q", reason, tc.wantError)
			}
			if tc.fallback {
				return
			}
			if wantFile {
				assertBlockedWith(t, plan, "Response blocked by the response guard")
			} else {
				assertDelivered(t, ctx, plan, headers.WarningGuard)
			}
		})
	}
}

func TestResponseGuardHoldsStreamWithIncompleteGrounding(t *testing.T) {
	for _, partial := range []bool{false, true} {
		name := "complete"
		blocks := []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: inventoryText}}
		if partial {
			name = "partial"
			blocks = append(blocks, llmprotocol.Content{Kind: llmprotocol.ContentImage})
		}
		t.Run(name, func(t *testing.T) {
			input := config.SystemOneInput{Answer: true, Context: true}
			router, ctx, calls := inventoryResponseGuard(t, input, config.SystemOneCoverageFull, config.SystemOneUnscannedUnknown, config.ResponseGuardBlock)
			ctx.SemanticRequest = inventoryRequest(inventoryToolResult("inventory", blocks...))
			streamingClient(ctx, llmprotocol.OpenAIChatV1)
			answerHeaders(t, router, ctx, "text/event-stream")
			if ctx.ResponseGuardHold == nil {
				t.Fatal("the blocking guard did not hold the stream")
			}
			client, _ := answerBody(t, router, ctx, whole(streamOf(llmprotocol.OpenAIChatV1, inventoryText)))
			if len(*calls) != 1 || (*calls)[0].Parts["context"] != inventoryText {
				t.Fatalf("model calls = %+v, want one with the readable context", *calls)
			}
			if partial {
				assertRefusedStream(t, llmprotocol.OpenAIChatV1, client, "Response blocked by the response guard", inventoryText)
				if reason := ctx.ResponseGuard.observed().Errors[inventoryKey]; reason != "systemone_unreadable_tool_result" {
					t.Fatalf("signal error = %q, want unreadable tool result", reason)
				}
			} else if !strings.Contains(chatStreamContent(client), inventoryText) {
				t.Fatalf("client stream = %s, want the inventory answer", client)
			}
		})
	}
}

func TestResponseInputGroundingUsesFinalSemanticRequest(t *testing.T) {
	text := func(s string) llmprotocol.Content { return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: s} }
	image := llmprotocol.Content{Kind: llmprotocol.ContentImage}
	for _, tc := range []struct {
		name           string
		results        []llmprotocol.Content
		wantText       string
		wantIncomplete bool
	}{
		{name: "only retrieved material and empty text", results: []llmprotocol.Content{inventoryToolResult("empty", text(""))}},
		{name: "tool text", results: []llmprotocol.Content{inventoryToolResult("inventory", text("first"), text("second")), inventoryToolResult("more", text("third"))}, wantText: "\n\nfirst second\n\nthird"},
		{name: "multimodal tool", results: []llmprotocol.Content{inventoryToolResult("inventory", text("first"), image, text("second"))}, wantText: "\n\nfirst second", wantIncomplete: true},
		{name: "unreadable tool", results: []llmprotocol.Content{inventoryToolResult("inventory", image)}, wantIncomplete: true},
		{name: "missing tool payload", results: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult}}, wantIncomplete: true},
		{name: "refusal and reasoning omitted", results: []llmprotocol.Content{inventoryToolResult("inventory",
			llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: "Unavailable."},
			llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: "Checking the supply."})}, wantIncomplete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			duplicates := []llmprotocol.Content{inventoryToolResult("rag", text("document"), image), inventoryToolResult("memory", text("memory"), image)}
			ctx := &RequestContext{
				SemanticRequest:     inventoryRequest(append(duplicates, tc.results...)...),
				RAGRetrievedContext: "document",
				MemoryContext:       "memory",
				RAGToolCallIDs:      map[string]struct{}{"rag": {}},
				MemoryToolCallIDs:   map[string]struct{}{"memory": {}},
			}
			ctx.ResponseStage.ToolResultsIncomplete = true
			input := ctx.responseInput(inventoryText)
			if want := "document\n\nmemory" + tc.wantText; input.Context != want {
				t.Errorf("context = %q, want %q", input.Context, want)
			}
			if input.ContextIncomplete != tc.wantIncomplete || !input.ToolResultsIncomplete {
				t.Errorf("context/tool coverage incomplete = %v/%v, want %v/true", input.ContextIncomplete, input.ToolResultsIncomplete, tc.wantIncomplete)
			}
		})
	}
}
