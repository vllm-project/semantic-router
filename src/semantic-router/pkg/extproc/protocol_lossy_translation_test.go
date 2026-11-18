package extproc

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A coding agent reports a command that exited nonzero as a failed tool
// result. Without the opt-in the dispatch to a Chat backend fails closed, and
// with it the backend gets the command output and the caller gets the warning.
func TestAllowLossyTranslationDispatchesFailedToolResultWithWarning(t *testing.T) {
	for _, allow := range []bool{false, true} {
		router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
		if allow {
			router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldToolResultError}
		}
		body := []byte(`{"model":"` + model + `","max_tokens":64,"messages":[
			{"role":"user","content":"List /nope"},
			{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls /nope"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":"Exit code 2\nls: cannot access '/nope': No such file or directory"}]}]}`)
		request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
		if err != nil {
			t.Fatal(err)
		}
		ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
		ctx.ProtocolEnvelope = envelope
		if _, err = router.prepareProviderDispatch(&request, model, "", false, ctx); err != nil {
			t.Fatal(err)
		}
		encoded, err := router.encodeDispatchRequest(ctx)
		if !allow {
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "lossy_translation" {
				t.Fatalf("default dispatch error = %v, want lossy_translation", err)
			}
			continue
		}
		if err != nil || !bytes.Contains(encoded, []byte("No such file or directory")) || bytes.Contains(encoded, []byte("is_error")) {
			t.Fatalf("opted-in dispatch: body=%s error=%v", encoded, err)
		}
		builder := newResponseHeaderMutationBuilder()
		builder.addWarnings(ctx)
		warning := ""
		for _, header := range builder.setHeaders {
			if header.Header.Key == headers.VSRWarnings {
				warning = string(header.Header.RawValue)
			}
		}
		if !strings.HasPrefix(warning, "protocol;action=approximated;") || !strings.HasSuffix(warning, `;field="tool_result.is_error"`) {
			t.Fatalf("opted-in loss is not visible to the caller: %q", warning)
		}
	}
}

func TestAnthropicThinkingBudgetToChatUsesLossyPolicy(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(map[bool]string{false: "strict", true: "opt-in"}[allow], func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			if allow {
				router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldReasoningBudget}
			}
			body := []byte(`{"model":"client-model","max_tokens":4096,"messages":[{"role":"user","content":"Claude Code reasoning probe"}],"thinking":{"type":"enabled","budget_tokens":1024}}`)
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			if err != nil {
				t.Fatalf("decode Anthropic request: %v", err)
			}
			request.Trusted.SourceFormat = llmprotocol.AnthropicMessagesV1
			if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 1024 {
				t.Fatalf("decoded reasoning budget = %v, want 1024", request.ReasoningBudgetTokens)
			}

			candidateErr := router.candidateCapabilityMismatch(
				config.ModelRef{Model: model}, &request, nil, nil, nil, true,
			)
			if allow && candidateErr != nil {
				t.Fatalf("opted-in candidate qualification: %v", candidateErr)
			}
			if !allow && !hasReasoningBudgetLossError(candidateErr) {
				t.Fatalf("strict candidate qualification = %v, want actionable lossy_translation", candidateErr)
			}

			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
			ctx.ProtocolEnvelope = envelope
			_, err = router.prepareProviderDispatch(&request, model, "", false, ctx)
			if !allow {
				if !hasReasoningBudgetLossError(err) {
					t.Fatalf("strict dispatch error = %v, want actionable lossy_translation", err)
				}
				if ctx.ImmediateProtocolError == nil || !strings.Contains(ctx.ImmediateProtocolError.Message, "global.router.allow_lossy_translation") {
					t.Fatalf("strict dispatch did not retain the actionable error: %+v", ctx.ImmediateProtocolError)
				}
				if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 1024 || len(ctx.ProtocolDiagnostics) != 0 {
					t.Fatalf("strict rejection changed semantic state or emitted a drop: request=%+v diagnostics=%+v", request, ctx.ProtocolDiagnostics)
				}
				return
			}
			if err != nil {
				t.Fatalf("opted-in dispatch: %v", err)
			}
			encoded, err := router.encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatalf("encode opted-in request: %v", err)
			}
			if strings.Contains(string(encoded), `"reasoning_budget_tokens"`) || !strings.Contains(string(encoded), "Claude Code reasoning probe") {
				t.Fatalf("provider request did not omit only the budget: %s", encoded)
			}
			if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 1024 {
				t.Fatalf("provider projection changed the semantic request: %+v", request)
			}
			want := llmprotocol.Diagnostic{
				Source: llmprotocol.AnthropicMessagesV1, Target: llmprotocol.OpenAIChatV1,
				Field: llmprotocol.LossyFieldReasoningBudget, Action: llmprotocol.DiagnosticDropped,
				Reason: "the selected Chat provider cannot honor the budget, so its default reasoning behavior applies",
			}
			if len(ctx.ProtocolDiagnostics) != 1 || ctx.ProtocolDiagnostics[0] != want {
				t.Fatalf("projection diagnostics = %+v, want one dropped-budget diagnostic", ctx.ProtocolDiagnostics)
			}
			builder := newResponseHeaderMutationBuilder()
			builder.addWarnings(ctx)
			warning := warningHeaderValue(builder)
			if !strings.HasPrefix(warning, "protocol;action=dropped;") || !strings.HasSuffix(warning, `;field="reasoning_budget_tokens"`) {
				t.Fatalf("x-vsr-warnings = %q, want the dropped budget", warning)
			}

			if _, err := router.encodeDispatchRequest(ctx); err != nil {
				t.Fatalf("repeat opted-in encoding: %v", err)
			}
			if len(ctx.ProtocolDiagnostics) != 1 {
				t.Fatalf("repeated encoding duplicated the projection diagnostic: %+v", ctx.ProtocolDiagnostics)
			}
		})
	}
}

func TestNullChatReasoningBudgetIsOmittedBeforeNonVLLMDispatch(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(map[bool]string{false: "strict", true: "opt-in"}[allow], func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			if allow {
				router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldReasoningBudget}
			}
			body := []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"hello"}],"reasoning_budget_tokens":null}`)
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body)
			if err != nil {
				t.Fatalf("decode Chat request: %v", err)
			}
			if request.ReasoningBudgetTokens != nil {
				t.Fatalf("decoded null reasoning budget = %v, want omitted", request.ReasoningBudgetTokens)
			}

			ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
			ctx.ProtocolEnvelope = envelope
			ctx.TargetFormat = llmprotocol.OpenAIChatV1
			ctx.RequestModel = model
			dispatch := &providerDispatch{
				logicalModel: model, backendModel: model, targetFormat: llmprotocol.OpenAIChatV1,
				profile: &config.ProviderProfile{Type: "openai"},
			}
			replayed, err := router.encodeDispatchRequest(ctx)
			if err != nil || !bytes.Equal(replayed, body) {
				t.Fatalf("encode reused original request: body=%s error=%v", replayed, err)
			}
			prepared, err := router.materializePreparedProviderRequest(dispatch, ctx)
			if err != nil {
				t.Fatalf("materialize provider request: %v", err)
			}
			if bytes.Contains(prepared.body, []byte(`"reasoning_budget_tokens"`)) ||
				bytes.Contains(prepared.body, []byte(`"thinking_token_budget"`)) {
				t.Fatalf("provider request retained a null or projected budget: %s", prepared.body)
			}
			if len(ctx.ProtocolDiagnostics) != 0 {
				t.Fatalf("null budget should not report a lossy translation: %+v", ctx.ProtocolDiagnostics)
			}
		})
	}
}

func TestChatReasoningBudgetOptInPreservesEffortAndNoBudgetRequests(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldReasoningBudget}
	for _, withBudget := range []bool{true, false} {
		t.Run(map[bool]string{true: "budget", false: "no budget"}[withBudget], func(t *testing.T) {
			body := `{"model":"` + model + `","max_tokens":64,"messages":[{"role":"user","content":"preserve effort"}],"reasoning_effort":"high"}`
			if withBudget {
				body = strings.TrimSuffix(body, "}") + `,"reasoning_budget_tokens":1024}`
			}
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
			if err != nil {
				t.Fatalf("decode Chat request: %v", err)
			}
			request.Trusted.SourceFormat = llmprotocol.OpenAIChatV1
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
			ctx.ProtocolEnvelope = envelope
			if _, dispatchErr := router.prepareProviderDispatch(&request, model, "", false, ctx); dispatchErr != nil {
				t.Fatalf("prepare Chat dispatch: %v", dispatchErr)
			}
			encoded, err := router.encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatalf("encode Chat request: %v", err)
			}
			observed := unmarshalReasoningRequest(t, encoded)
			if observed["reasoning_effort"] != "high" || strings.Contains(string(encoded), `"reasoning_budget_tokens"`) {
				t.Fatalf("provider request = %s, want effort and no neutral budget", encoded)
			}
			if withBudget {
				if len(ctx.ProtocolDiagnostics) != 1 || ctx.ProtocolDiagnostics[0].Field != llmprotocol.LossyFieldReasoningBudget {
					t.Fatalf("budget diagnostics = %+v", ctx.ProtocolDiagnostics)
				}
			} else if len(ctx.ProtocolDiagnostics) != 0 {
				t.Fatalf("budget-free request emitted diagnostics: %+v", ctx.ProtocolDiagnostics)
			}
		})
	}
}

func TestReasoningBudgetQualificationKeepsVLLMFallbackCandidate(t *testing.T) {
	router, openAIModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	vllmModel := "target-vllm-fallback"
	router.Config.ModelConfig[vllmModel] = config.ModelParams{
		PreferredEndpoints: []string{"vllm-backend"},
		APIFormat:          config.APIFormatOpenAI,
		ExternalModelIDs:   map[string]string{"vllm": "vllm-provider-model"},
	}
	router.Config.ProviderProfiles["vllm-provider"] = config.ProviderProfile{
		Type: "vllm", BaseURL: "http://127.0.0.1:8001/v1",
	}
	router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, config.VLLMEndpoint{
		Name: "vllm-backend", Address: "127.0.0.1", Port: 8001,
		ProviderProfileName: "vllm-provider",
	})
	body := []byte(`{"model":"` + openAIModel + `","max_tokens":64,"messages":[{"role":"user","content":"fallback reasoning budget"}],"reasoning_budget_tokens":1024}`)
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("decode Chat request: %v", err)
	}
	request.Trusted.SourceFormat = llmprotocol.OpenAIChatV1

	openAICandidateErr := router.candidateCapabilityMismatch(config.ModelRef{Model: openAIModel}, &request, nil, nil, nil, true)
	if !hasReasoningBudgetLossError(openAICandidateErr) {
		t.Fatalf("strict non-vLLM candidate error = %v, want lossy_translation", openAICandidateErr)
	}
	vllmCandidateErr := router.candidateCapabilityMismatch(config.ModelRef{Model: vllmModel}, &request, nil, nil, nil, true)
	if vllmCandidateErr != nil {
		t.Fatalf("strict vLLM fallback candidate was rejected: %v", vllmCandidateErr)
	}

	ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
	ctx.ProtocolEnvelope = envelope
	useReasoning := true
	candidateRef := config.ModelRef{
		Model: vllmModel, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &useReasoning},
	}
	decision := config.Decision{Name: "reasoning-budget-fallback", ModelRefs: []config.ModelRef{candidateRef}}
	ctx.VSRSelectedDecision = &decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: openAIModel, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &useReasoning}}, candidateRef,
	}
	candidate, _, err := router.prepareFallbackCandidate(context.Background(), &candidateRef, ctx, false)
	if err != nil {
		t.Fatalf("prepare vLLM fallback candidate: %v", err)
	}
	observed := unmarshalReasoningRequest(t, candidate.body)
	if observed["thinking_token_budget"] != float64(1024) || strings.Contains(string(candidate.body), `"reasoning_budget_tokens"`) {
		t.Fatalf("final vLLM fallback request = %s, want native budget 1024 only", candidate.body)
	}
}

func hasReasoningBudgetLossError(err error) bool {
	var protocolErr *llmprotocol.ProtocolError
	return errors.As(err, &protocolErr) && protocolErr.Code == "lossy_translation" &&
		strings.Contains(protocolErr.Message, llmprotocol.OptInHint(llmprotocol.LossyFieldReasoningBudget))
}

func warningHeaderValue(builder *responseHeaderMutationBuilder) string {
	for _, header := range builder.setHeaders {
		if header.Header.Key == headers.VSRWarnings {
			return string(header.Header.RawValue)
		}
	}
	return ""
}
