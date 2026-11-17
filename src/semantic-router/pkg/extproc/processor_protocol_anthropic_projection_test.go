package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const adaptiveClaudeCodeRequest = `{
 "model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],
 "thinking":{"type":"adaptive","display":"omitted"},
 "output_config":{"effort":"medium"},
 "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}
}`

func TestAnthropicAdaptiveRequestDispatchesToChatAndResponses(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			router := routingTestRouter("chat")
			if target == llmprotocol.OpenAIResponsesV1 {
				model := router.Config.ModelConfig["chat"]
				model.APIFormat = config.APIFormatResponses
				model.ReasoningFamily = "reasoning"
				router.Config.ModelConfig["chat"] = model
				router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
					"reasoning": {Type: config.ReasoningFamilyTypeReasoningEffort, Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}},
				}
			}
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(adaptiveClaudeCodeRequest))
			if err != nil {
				t.Fatal(err)
			}
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
			ctx.ProtocolEnvelope = envelope
			dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
			if err != nil {
				t.Fatalf("dispatch rejected adaptive request: %v", err)
			}
			if dispatch.targetFormat != target {
				t.Fatalf("target = %q, want %q", dispatch.targetFormat, target)
			}
			if err = router.candidateCapabilityMismatch(config.ModelRef{Model: "chat"}, &request, nil, nil, nil, true); err != nil {
				t.Fatalf("selection disagrees with dispatch: %v", err)
			}
			body, err := router.encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			if _, found := wire["thinking"]; found || len(wire["context_management"]) > 0 {
				t.Fatalf("Anthropic-only controls reached backend: %s", body)
			}
			if target == llmprotocol.OpenAIChatV1 && string(wire["reasoning_effort"]) != `"medium"` {
				t.Fatalf("Chat effort was lost: %s", body)
			}
			if target == llmprotocol.OpenAIResponsesV1 && !bytes.Contains(wire["reasoning"], []byte(`"medium"`)) {
				t.Fatalf("Responses effort was lost: %s", body)
			}
			if request.ReasoningMode != llmprotocol.ReasoningModeAdaptive || request.ReasoningDisplay != "omitted" || len(request.ContextManagement) == 0 {
				t.Fatalf("public request was mutated: %+v", request)
			}
		})
	}
}

func TestAnthropicAdaptiveStrictCandidatesUseProjectedDemand(t *testing.T) {
	router := routingTestRouter("chat")
	router.Config.CandidateRequirements = &config.CandidateRequirements{
		Capabilities: config.CandidateCapabilitiesDeclared,
		Context:      config.CandidateContextKnownLimits,
	}
	model := router.Config.ModelConfig["chat"]
	model.Capabilities = []string{"chat", "reasoning"}
	model.ContextWindowSize = 8192
	model.MaxOutputTokens = 1024
	router.Config.ModelConfig["chat"] = model
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(adaptiveClaudeCodeRequest))
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{Name: "default", ModelRefs: []config.ModelRef{{Model: "chat"}}}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.VSRSelectedDecision = decision
	refs, err := router.decisionEligibleModelRefs(decision, ctx)
	if err != nil || len(refs) != 1 || refs[0].Model != "chat" {
		t.Fatalf("strict prefilter rejected projected request: refs=%+v err=%v", refs, err)
	}
	decision.Action = &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "chat"}
	modelName, found, err := router.decisionRouteActionDestination(decision, ctx)
	if err != nil || !found || modelName != "chat" {
		t.Fatalf("strict route action rejected projected request: model=%q found=%t err=%v", modelName, found, err)
	}
}

func TestAnthropicAdaptiveResponsesDropsCacheDirectiveWithWarning(t *testing.T) {
	router := routingTestRouter("chat")
	model := router.Config.ModelConfig["chat"]
	model.APIFormat = config.APIFormatResponses
	router.Config.ModelConfig["chat"] = model
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
		"model":"chat","max_tokens":64,
		"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}],
		"thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"medium"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = llmprotocol.Envelope{}
	if _, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx); err != nil {
		t.Fatalf("Responses rejected an Anthropic cache hint: %v", err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte("cache_control")) || !bytes.Contains(body, []byte("hello")) {
		t.Fatalf("Responses backend received the cache hint or lost prompt content: %s", body)
	}
	if len(ctx.ProtocolDiagnostics) != 1 ||
		ctx.ProtocolDiagnostics[0].Field != "cache_control" ||
		ctx.ProtocolDiagnostics[0].Action != llmprotocol.DiagnosticDropped ||
		ctx.ProtocolDiagnostics[0].Target != llmprotocol.OpenAIResponsesV1 {
		t.Fatalf("Responses cache drop was not reported once: %+v", ctx.ProtocolDiagnostics)
	}
	if !llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityCacheDirectives) {
		t.Fatal("dispatch modified the public Anthropic request")
	}
}

// Claude Code places its environment section in a system message after the
// first user turn and later notices after the tool results they refer to.
// Qwen3.5 and later templates raise on that position, and gpt-oss, MiniMax-M2
// and Nemotron templates drop it. Chat and Responses backends receive the
// messages in place by default, and as user text in the same place when the
// operator lists messages[].role in allow_lossy_translation.
func TestAnthropicMidConversationSystemMessagesBecomeUserTextOnlyWhenOptedIn(t *testing.T) {
	for _, check := range []struct {
		target   llmprotocol.WireFormat
		key      string
		inPlace  []string
		userText []string
	}{
		{
			llmprotocol.OpenAIChatV1, "messages",
			[]string{"system", "user", "system", "assistant", "tool", "system"},
			[]string{"system", "user", "assistant", "tool", "user"},
		},
		{
			llmprotocol.OpenAIResponsesV1, "input",
			[]string{"system", "user", "system", "function_call", "function_call_output", "system"},
			[]string{"system", "user", "function_call", "function_call_output", "user"},
		},
	} {
		for _, optedIn := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/opt_in=%t", check.target, optedIn), func(t *testing.T) {
				router := routingTestRouter("chat")
				if check.target == llmprotocol.OpenAIResponsesV1 {
					model := router.Config.ModelConfig["chat"]
					model.APIFormat = config.APIFormatResponses
					router.Config.ModelConfig["chat"] = model
				}
				want, environment, notice := check.inPlace, 2, 5
				if optedIn {
					router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldSystemMessageRole}
					want, environment, notice = check.userText, 1, 4
				}
				request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
					"model":"chat","max_tokens":64,"system":[{"type":"text","text":"You are a coding agent."}],
					"tools":[{"name":"Read","input_schema":{"type":"object"}}],
					"messages":[
						{"role":"user","content":"Fix the build"},
						{"role":"system","content":[{"type":"text","text":"# Environment\nWorking directory: /work"}]},
						{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Read","input":{"file_path":"/work/Makefile"}}]},
						{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"all: build"}]},
						{"role":"system","content":"Note: /work/Makefile changed on disk since you last read it."}
					]
				}`))
				if err != nil {
					t.Fatal(err)
				}
				ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
				ctx.ProtocolEnvelope = envelope
				if _, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx); err != nil {
					t.Fatal(err)
				}
				body, err := router.encodeDispatchRequest(ctx)
				if err != nil {
					t.Fatal(err)
				}
				var wire map[string]json.RawMessage
				var items []map[string]json.RawMessage
				if err = json.Unmarshal(body, &wire); err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(wire[check.key], &items); err != nil {
					t.Fatal(err)
				}
				var labels []string
				for _, item := range items {
					var label string
					if err = json.Unmarshal(item["role"], &label); err != nil {
						_ = json.Unmarshal(item["type"], &label)
					}
					labels = append(labels, label)
				}
				if !reflect.DeepEqual(labels, want) ||
					!bytes.Contains(items[environment]["content"], []byte("# Environment")) ||
					!bytes.Contains(items[notice]["content"], []byte("changed on disk")) ||
					bytes.Contains(items[0]["content"], []byte("# Environment")) {
					t.Fatalf("backend conversation = %v: %s", labels, body)
				}
				reported := len(ctx.ProtocolDiagnostics) == 1 &&
					ctx.ProtocolDiagnostics[0].Field == llmprotocol.LossyFieldSystemMessageRole &&
					ctx.ProtocolDiagnostics[0].Action == llmprotocol.DiagnosticApproximated
				if reported != optedIn || (!optedIn && len(ctx.ProtocolDiagnostics) != 0) {
					t.Fatalf("diagnostics = %+v, want one messages[].role entry only when opted in", ctx.ProtocolDiagnostics)
				}
				if request.Messages[1].Role != llmprotocol.RoleSystem {
					t.Fatal("dispatch modified the public Anthropic request")
				}
			})
		}
	}
}

func TestAnthropicCrossFormatProjectionRejectsMeaningfulEdits(t *testing.T) {
	router := routingTestRouter("chat")
	engine := protocolcodec.NewBuiltinEngine()
	for _, test := range []struct {
		name, body, code string
	}{
		{"clear_last", `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"last"}]}}`, "lossy_translation"},
		{"unknown_edit_option", `{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all","new_option":true}]}}`, "lossy_translation"},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			_, err = router.projectAnthropicRequestForBackend(request, "chat", llmprotocol.OpenAIChatV1)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.code {
				t.Fatalf("projection error = %v, want %s", err, test.code)
			}
		})
	}
}

func TestAnthropicDisabledRequiresConfiguredOffControl(t *testing.T) {
	router := routingTestRouter("chat")
	engine := protocolcodec.NewBuiltinEngine()
	body := []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	if _, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx); err == nil {
		t.Fatal("unconfigured model accepted disabled thinking")
	}
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"qwen3": {
			Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
			Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
		},
	}
	model := router.Config.ModelConfig["chat"]
	model.ReasoningFamily = "qwen3"
	router.Config.ModelConfig["chat"] = model
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = "vllm"
	router.Config.ProviderProfiles["provider"] = profile
	dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatalf("configured reasoning-off model rejected: %v", err)
	}
	encoded, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	adapted, err := router.adaptProviderRequest(encoded, dispatch, ctx)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		ChatTemplateKwargs map[string]bool `json:"chat_template_kwargs"`
	}
	if err := json.Unmarshal(adapted, &wire); err != nil {
		t.Fatal(err)
	}
	if enabled, found := wire.ChatTemplateKwargs["enable_thinking"]; !found || enabled {
		t.Fatalf("backend did not receive an explicit off control: %s", adapted)
	}
}

func TestAnthropicDisabledRejectsFamiliesWithoutOffControl(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		family config.ReasoningFamilyConfig
	}{
		{"always_on", config.ReasoningFamilyConfig{Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking", Modes: []string{config.ReasoningModeEnabled}}},
		{"effort_without_off_value", config.ReasoningFamilyConfig{Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort", Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := routingTestRouter("chat")
			router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"family": test.family}
			model := router.Config.ModelConfig["chat"]
			model.ReasoningFamily = "family"
			router.Config.ModelConfig["chat"] = model
			_, projectionErr := router.projectAnthropicRequestForBackend(request, "chat", llmprotocol.OpenAIChatV1)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(projectionErr, &protocolError) || protocolError.Code != "unsupported_capability" {
				t.Fatalf("unsupported off control accepted: %v", projectionErr)
			}
		})
	}
}

func TestAnthropicDisabledRejectsProviderWithoutEffectiveOffControl(t *testing.T) {
	router := routingTestRouter("chat")
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"family": {
			Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
			ActivationParameter: "enable_thinking", Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
		},
	}
	model := router.Config.ModelConfig["chat"]
	model.ReasoningFamily = "family"
	router.Config.ModelConfig["chat"] = model
	profile := router.Config.ProviderProfiles["provider"]
	profile.ReasoningTransport = modelcatalog.ReasoningTransportTopLevelEffort
	router.Config.ProviderProfiles["provider"] = profile
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	dispatch, err := router.prepareProviderDispatch(&request, "chat", "", false, ctx)
	if err != nil {
		t.Fatal(err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_, err = router.adaptProviderRequest(body, dispatch, ctx)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" {
		t.Fatalf("provider silently ignored disabled thinking: %v", err)
	}
}

func TestAnthropicOmittedDisplayFiltersBufferedChatReasoning(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	body := []byte(`{"id":"response_1","model":"chat","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"private thinking","content":"public answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`)
	response, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatal(err)
	}
	public, err := router.encodeClientResponse(*response, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte(`"thinking"`)) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("omitted display leaked reasoning or lost answer: %s", public)
	}
}

func TestAnthropicOmittedDisplayFiltersBufferedResponsesReasoning(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	body := []byte(`{
		"id":"resp_1","model":"chat","status":"completed",
		"output":[
			{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"private summary"}],"content":[{"type":"reasoning_text","text":"private thinking"}]},
			{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"public answer"}]}
		],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
	}`)
	response, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatal(err)
	}
	public, err := router.encodeClientResponse(*response, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte("private summary")) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("omitted display leaked reasoning or lost answer: %s", public)
	}
}

func TestAnthropicOmittedDisplayFiltersStreamingChatReasoning(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{
			ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted",
		},
	}
	engine := protocolcodec.NewBuiltinEngine()
	stream, err := engine.NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: t.Context()}, clientStreamMutation(ctx, llmprotocol.OpenAIChatV1))
	if err != nil {
		t.Fatal(err)
	}
	chunks := []string{
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"reasoning_content":"private thinking"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{"content":"public answer"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	var public []byte
	for _, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		for _, frame := range frames {
			public = append(public, frame...)
		}
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		public = append(public, frame...)
	}
	if bytes.Contains(public, []byte("private thinking")) || bytes.Contains(public, []byte("thinking_delta")) || !bytes.Contains(public, []byte("public answer")) {
		t.Fatalf("stream leaked reasoning or lost answer: %s", public)
	}
}

// TestSystemMessageProjectionKeepsTheCountOutOfTheMetricLabel sends requests
// with one, two and three mid-conversation system messages. x-vsr-warnings
// names each request's count, while llm_translation_lossy_total keeps one
// series for the projection, so a conversation's length never adds a series.
func TestSystemMessageProjectionKeepsTheCountOutOfTheMetricLabel(t *testing.T) {
	router := routingTestRouter("chat")
	router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldSystemMessageRole}
	const label = "N system messages sent as user text in place"
	counter := metrics.TranslationLossyTotal.WithLabelValues("anthropic", "openai", "approximated", label)
	before := testutil.ToFloat64(counter)
	for count := 1; count <= 3; count++ {
		messages := []string{`{"role":"user","content":"Fix the build"}`}
		for range count {
			messages = append(messages, `{"role":"system","content":"Note: a file changed on disk."}`, `{"role":"user","content":"go on"}`)
		}
		request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1,
			[]byte(`{"model":"chat","max_tokens":64,"messages":[`+strings.Join(messages, ",")+`]}`))
		if err != nil {
			t.Fatal(err)
		}
		ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
		ctx.ProtocolEnvelope = envelope
		if _, err = router.prepareProviderDispatch(&request, "chat", "", false, ctx); err != nil {
			t.Fatal(err)
		}
		if _, err = router.encodeDispatchRequest(ctx); err != nil {
			t.Fatal(err)
		}
		want := fmt.Sprintf(`protocol;action=approximated;reason="%d system messages sent as user text in place";field="messages[].role"`, count)
		if got := responseWarnings(ctx); got != want {
			t.Fatalf("x-vsr-warnings = %s, want %s", got, want)
		}
	}
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != "llm_translation_lossy_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, pair := range metric.GetLabel() {
				if pair.GetName() == "reason" && strings.Contains(pair.GetValue(), "system messages") && pair.GetValue() != label {
					t.Errorf("llm_translation_lossy_total has a per-request series reason=%q", pair.GetValue())
				}
			}
		}
	}
	if got := testutil.ToFloat64(counter) - before; got != 3 {
		t.Fatalf("llm_translation_lossy_total{reason=%q} rose by %v, want 3", label, got)
	}
}

func TestAnthropicOmittedDisplaySurvivesReasoningOnForAnotherFormat(t *testing.T) {
	for target, format := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIResponsesV1: config.APIFormatResponses,
		llmprotocol.OpenAIChatV1:      config.APIFormatChatCompletions,
	} {
		router := routingTestRouter("chat")
		model := router.Config.ModelConfig["chat"]
		model.APIFormat = format
		model.ReasoningFamily = "reasoning"
		router.Config.ModelConfig["chat"] = model
		router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
			"reasoning": {
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "medium", "high"}, Default: "high",
				Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
			},
		}
		request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(adaptiveClaudeCodeRequest))
		if err != nil {
			t.Fatal(err)
		}
		decision := &config.Decision{Name: "route", ModelRefs: []config.ModelRef{{
			Model:                 "chat",
			ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(true), ReasoningEffort: "high"},
		}}}
		router.applySemanticReasoningMode(&request, "chat", target, true, decision)
		ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
		// With reasoning on, the client's omitted display still keeps the
		// backend's reasoning out of its response.
		if !omitCrossFormatReasoning(ctx, target) {
			t.Fatalf("%s: display = %q, want the client's omitted display kept", target, request.ReasoningDisplay)
		}
	}
}

// TestAnthropicThinkingDisplayOnChatAndResponsesBackends sends the summarized
// and updates displays Claude Code asks for to backends that cannot shape
// thinking. The backend's reasoning comes back as it does without a display,
// reported as approximated only when the backend reasons.
func TestAnthropicThinkingDisplayOnChatAndResponsesBackends(t *testing.T) {
	const approximated = `protocol;action=approximated;reason="the backend's reasoning is returned as it is, not summarized";field="thinking.display"`
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, display := range []string{"summarized", "updates"} {
			for _, test := range []struct {
				name              string
				family, reasoning bool
			}{{"reasoning_on", true, true}, {"reasoning_off", true, false}, {"no_family", false, true}} {
				t.Run(fmt.Sprintf("%s/%s/%s", target, display, test.name), func(t *testing.T) {
					router := routingTestRouter("chat")
					model := router.Config.ModelConfig["chat"]
					if target == llmprotocol.OpenAIResponsesV1 {
						model.APIFormat = config.APIFormatResponses
					}
					if test.family {
						model.ReasoningFamily = "qwen3"
						router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"qwen3": {
							Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
							Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
						}}
					}
					router.Config.ModelConfig["chat"] = model
					profile := router.Config.ProviderProfiles["provider"]
					profile.Type = "vllm"
					router.Config.ProviderProfiles["provider"] = profile
					request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(
						`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","display":"`+display+`"}}`))
					if err != nil {
						t.Fatal(err)
					}
					ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
					ctx.ProtocolEnvelope = envelope
					ctx.VSRSelectedDecision = &config.Decision{Name: "route", ModelRefs: []config.ModelRef{{
						Model: "chat", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(test.reasoning)},
					}}}
					dispatch, err := router.prepareProviderDispatch(&request, "chat", "route", test.reasoning, ctx)
					if err != nil {
						t.Fatalf("dispatch rejected the display: %v", err)
					}
					encoded, err := router.encodeDispatchRequest(ctx)
					if err != nil {
						t.Fatal(err)
					}
					adapted, err := router.adaptProviderRequest(encoded, dispatch, ctx)
					if err != nil {
						t.Fatal(err)
					}
					if bytes.Contains(adapted, []byte("display")) {
						t.Fatalf("the display reached the backend: %s", adapted)
					}
					want := ""
					if test.family && test.reasoning {
						want = approximated
					}
					if got := responseWarnings(ctx); got != want {
						t.Fatalf("x-vsr-warnings = %q, want %q", got, want)
					}
				})
			}
		}
	}
}

// TestAnthropicOmittedDisplayKeepsAnAnswerThatIsAllReasoning sends an answer
// whose output limit went to reasoning alone. With the reasoning omitted the
// client gets an empty message with the backend's stop reason, not a 502.
func TestAnthropicOmittedDisplayKeepsAnAnswerThatIsAllReasoning(t *testing.T) {
	for target, body := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"id":"response_1","model":"chat","choices":[{"index":0,"message":{"role":"assistant","reasoning":"private thinking","content":null},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":40,"total_tokens":42}}`,
		llmprotocol.OpenAIResponsesV1: `{"id":"resp_1","model":"chat","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private thinking"}]}],
			"usage":{"input_tokens":2,"output_tokens":40,"total_tokens":42}}`,
	} {
		t.Run(string(target), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: target,
				SemanticRequest: &llmprotocol.Request{ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted"},
			}
			response, err := router.decodeClientResponse([]byte(body), ctx)
			if err != nil {
				t.Fatal(err)
			}
			public, err := router.encodeClientResponse(*response, ctx)
			if err != nil {
				t.Fatal(err)
			}
			var message struct {
				Content    []json.RawMessage `json:"content"`
				StopReason string            `json:"stop_reason"`
			}
			if err := json.Unmarshal(public, &message); err != nil {
				t.Fatal(err)
			}
			if message.Content == nil || len(message.Content) != 0 || message.StopReason != "max_tokens" {
				t.Fatalf("want an empty message stopped at max_tokens, got %s", public)
			}
		})
	}
}

// TestAnthropicOmittedDisplayStreamsAnAnswerThatIsAllReasoning streams the
// same answer from Chat and Responses backends. Like the buffered reply, the
// stream carries no content block, not even an empty one.
func TestAnthropicOmittedDisplayStreamsAnAnswerThatIsAllReasoning(t *testing.T) {
	chunk := `{"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":%s,"finish_reason":%s}]}`
	item := `{"id":"rs_1","type":"reasoning","summary":[],"content":[]}`
	done := `{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private thinking"}]}`
	resource := `{"id":"resp_1","object":"response","model":"chat","status":"%s","incomplete_details":%s,"output":%s,"usage":%s}`
	usage := `{"input_tokens":2,"output_tokens":40,"total_tokens":42}`
	for backend, events := range map[llmprotocol.WireFormat][]string{
		llmprotocol.OpenAIChatV1: {
			fmt.Sprintf(chunk, `{"role":"assistant"}`, "null"),
			fmt.Sprintf(chunk, `{"reasoning":"private thinking"}`, "null"),
			fmt.Sprintf(chunk, `{}`, `"length"`),
			"[DONE]",
		},
		llmprotocol.OpenAIResponsesV1: {
			`{"type":"response.created","sequence_number":0,"response":` + fmt.Sprintf(resource, "in_progress", "null", "[]", "null") + `}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":` + item + `}`,
			`{"type":"response.content_part.added","sequence_number":2,"item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":""}}`,
			`{"type":"response.reasoning_text.delta","sequence_number":3,"item_id":"rs_1","output_index":0,"content_index":0,"delta":"private thinking"}`,
			`{"type":"response.reasoning_text.done","sequence_number":4,"item_id":"rs_1","output_index":0,"content_index":0,"text":"private thinking"}`,
			`{"type":"response.content_part.done","sequence_number":5,"item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":"private thinking"}}`,
			`{"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":` + done + `}`,
			`{"type":"response.incomplete","sequence_number":7,"response":` + fmt.Sprintf(resource, "incomplete", `{"reason":"max_output_tokens"}`, "["+done+"]", usage) + `}`,
		},
	} {
		t.Run(string(backend), func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: backend,
				SemanticRequest: &llmprotocol.Request{ReasoningMode: llmprotocol.ReasoningModeAdaptive, ReasoningDisplay: "omitted"},
			}
			stream, err := protocolcodec.NewBuiltinEngine().NewStreamWithMutation(backend, llmprotocol.AnthropicMessagesV1,
				llmprotocol.StreamContext{Context: t.Context()}, clientStreamMutation(ctx, backend))
			if err != nil {
				t.Fatal(err)
			}
			var public []byte
			for _, event := range events {
				frames, _, _, pushErr := stream.Push([]byte("data: " + event + "\n\n"))
				if pushErr != nil {
					t.Fatal(pushErr)
				}
				public = append(public, bytes.Join(frames, nil)...)
			}
			frames, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			public = append(public, bytes.Join(frames, nil)...)
			if bytes.Contains(public, []byte("content_block_start")) || bytes.Contains(public, []byte(`"error"`)) ||
				!bytes.Contains(public, []byte(`"stop_reason":"max_tokens"`)) || !bytes.Contains(public, []byte("message_stop")) {
				t.Fatalf("want a stream with no content stopped at max_tokens, got %s", public)
			}
		})
	}
}

// Without a display the same Chat stream keeps its thinking block and text.
func TestAnthropicChatReasoningStreamKeepsItsThinkingBlock(t *testing.T) {
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{ReasoningMode: llmprotocol.ReasoningModeAdaptive},
	}
	stream, err := protocolcodec.NewBuiltinEngine().NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: t.Context()}, clientStreamMutation(ctx, llmprotocol.OpenAIChatV1))
	if err != nil {
		t.Fatal(err)
	}
	chunk := `data: {"id":"response_1","object":"chat.completion.chunk","model":"chat","choices":[{"index":0,"delta":%s,"finish_reason":%s}]}` + "\n\n"
	var public []byte
	for _, frame := range []string{
		fmt.Sprintf(chunk, `{"role":"assistant"}`, "null"), fmt.Sprintf(chunk, `{"reasoning":"private thinking"}`, "null"),
		fmt.Sprintf(chunk, `{"content":"ok"}`, "null"), fmt.Sprintf(chunk, `{}`, `"stop"`), "data: [DONE]\n\n",
	} {
		frames, _, _, pushErr := stream.Push([]byte(frame))
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		public = append(public, bytes.Join(frames, nil)...)
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	public = append(public, bytes.Join(frames, nil)...)
	for _, want := range []string{
		`"content_block":{"thinking":"","type":"thinking"}`, `"thinking":"private thinking"`,
		`"content_block":{"text":"","type":"text"}`, `"text":"ok"`,
	} {
		if !bytes.Contains(public, []byte(want)) {
			t.Fatalf("stream lost %s: %s", want, public)
		}
	}
}
