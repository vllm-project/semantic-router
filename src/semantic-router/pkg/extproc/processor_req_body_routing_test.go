package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestProviderDispatchEncodesEveryClientBackendProtocolPair(t *testing.T) {
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, source := range formats {
		for _, target := range formats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				assertProviderDispatchPair(t, source, target)
			})
		}
	}
}

func TestProviderDispatchAppliesReasoningBeforeExternalModelIDRewrite(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	params := router.Config.ModelConfig[logicalModel]
	params.ReasoningFamily = "openai"
	router.Config.ModelConfig[logicalModel] = params
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"openai": {
			Type:      config.ReasoningFamilyTypeTopLevelReasoningEffort,
			Parameter: "reasoning_effort",
		},
	}
	decision := config.Decision{
		Name: "Complex",
		ModelRefs: []config.ModelRef{{
			Model: logicalModel,
			ModelReasoningControl: config.ModelReasoningControl{
				ReasoningEffort: "high",
			},
		}},
	}
	request := testNeutralRequest("virtual", "solve this")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &decision

	response, err := router.handleEntrypointModelRouting(
		request,
		"virtual",
		decision.Name,
		entropy.ReasoningDecision{UseReasoning: true},
		logicalModel,
		ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var wire struct {
		Model           string `json:"model"`
		ReasoningEffort string `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode provider request: %v", err)
	}
	if wire.Model != "provider-model" {
		t.Fatalf("provider model = %q, want provider-model", wire.Model)
	}
	if wire.ReasoningEffort != "high" {
		t.Fatalf("reasoning effort = %q, want high", wire.ReasoningEffort)
	}
}

func TestToolSelectionMutatesNeutralRequestBeforeEveryProviderEncoding(t *testing.T) {
	semanticSelection := false
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:           true,
		Mode:              config.ToolsPluginModeFiltered,
		AllowTools:        []string{"lookup_weather"},
		SemanticSelection: &semanticSelection,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "tool-selection",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginTools, Configuration: payload,
		}},
	}
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, target := range formats {
		t.Run(string(target), func(t *testing.T) {
			router, logicalModel := routingTestRouterForFormat(target)
			request := testNeutralRequest("entrypoint", "What is the weather?")
			request.Tools = []llmprotocol.Tool{
				{Name: "lookup_weather", InputSchema: json.RawMessage(`{"type":"object"}`)},
				{Name: "unrelated_tool", InputSchema: json.RawMessage(`{"type":"object"}`)},
			}
			request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.VSRSelectedDecision = decision

			response, routeErr := router.handleEntrypointModelRouting(
				request, "entrypoint", decision.Name, entropy.ReasoningDecision{}, logicalModel, ctx,
			)
			if routeErr != nil {
				t.Fatal(routeErr)
			}
			if len(request.Tools) != 1 || request.Tools[0].Name != "lookup_weather" || ctx.SemanticRequest != request {
				t.Fatalf("tool selection did not commit exactly one neutral request: %+v", request.Tools)
			}
			body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			var wire map[string]any
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			tools, ok := wire["tools"].([]any)
			if !ok || len(tools) != 1 {
				t.Fatalf("provider body has unexpected tools: %s", body)
			}
			tool, ok := tools[0].(map[string]any)
			if !ok {
				t.Fatalf("provider tool is malformed: %s", body)
			}
			name, _ := tool["name"].(string)
			if target == llmprotocol.OpenAIChatV1 {
				function, _ := tool["function"].(map[string]any)
				name, _ = function["name"].(string)
			}
			if name != "lookup_weather" {
				t.Fatalf("provider body encoded stale tools: %s", body)
			}
		})
	}
}

func TestProviderDispatchKeepsModelServerReasoningExtensionsAtProviderBoundary(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	params := router.Config.ModelConfig[logicalModel]
	params.ReasoningFamily = "qwen"
	router.Config.ModelConfig[logicalModel] = params
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"qwen": {
			Type:      config.ReasoningFamilyTypeChatTemplateKwargs,
			Parameter: "enable_thinking",
		},
	}
	decision := config.Decision{Name: "Agentic"}
	request := testNeutralRequest("virtual", "plan this")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &decision

	response, err := router.handleEntrypointModelRouting(
		request,
		"virtual",
		decision.Name,
		entropy.ReasoningDecision{UseReasoning: true},
		logicalModel,
		ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var wire struct {
		Model              string          `json:"model"`
		ChatTemplateKwargs json.RawMessage `json:"chat_template_kwargs"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode provider request: %v", err)
	}
	if wire.Model != "provider-model" {
		t.Fatalf("provider model = %q, want provider-model", wire.Model)
	}
	var kwargs map[string]bool
	if err := json.Unmarshal(wire.ChatTemplateKwargs, &kwargs); err != nil {
		t.Fatalf("decode chat_template_kwargs: %v", err)
	}
	if !kwargs["enable_thinking"] {
		t.Fatalf("reasoning extension = %s, want enable_thinking=true", wire.ChatTemplateKwargs)
	}
}

func assertProviderDispatchPair(t *testing.T, source, target llmprotocol.WireFormat) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(target)
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: source,
		RequestID: "protocol-matrix-request", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest(protocolRequestFixture(source), ctx)
	if immediate != nil {
		t.Fatalf("prepareProtocolRequest(%s) returned immediate response", source)
	}
	response, err := router.handleSpecifiedModelRouting(request, logicalModel, "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting(%s -> %s): %v", source, target, err)
	}
	assertDispatchedRequest(t, response, target)
}

func TestChatDispatchForcesBackendUsageWithoutChangingClientPreference(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1,
		RequestID: "stream-usage-contract", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(
		`{"model":"virtual","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":false}}`,
	), ctx)
	if immediate != nil {
		t.Fatal("prepareProtocolRequest returned an immediate response")
	}
	if ctx.SemanticRequest.StreamOptions.IncludeUsage == nil || *ctx.SemanticRequest.StreamOptions.IncludeUsage {
		t.Fatalf("client usage preference = %+v, want explicit false", ctx.SemanticRequest.StreamOptions)
	}
	response, err := router.handleSpecifiedModelRouting(request, logicalModel, "", ctx)
	if err != nil {
		t.Fatal(err)
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var wire struct {
		StreamOptions struct {
			IncludeUsage *bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.StreamOptions.IncludeUsage == nil || !*wire.StreamOptions.IncludeUsage {
		t.Fatalf("backend dispatch did not request authoritative usage: %s", body)
	}
	if ctx.SemanticRequest.StreamOptions.IncludeUsage == nil || *ctx.SemanticRequest.StreamOptions.IncludeUsage {
		t.Fatalf("backend accounting override mutated client preference: %+v", ctx.SemanticRequest.StreamOptions)
	}
}

func assertDispatchedRequest(t *testing.T, response *ext_proc.ProcessingResponse, target llmprotocol.WireFormat) {
	t.Helper()
	common := response.GetRequestBody().GetResponse()
	body := common.GetBodyMutation().GetBody()
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(target, body)
	if err != nil {
		t.Fatalf("decode dispatched %s body: %v\n%s", target, err, body)
	}
	if decoded.Model != "provider-model" || len(decoded.Messages) != 1 ||
		len(decoded.Messages[0].Content) != 1 || decoded.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("dispatched request changed semantics: %+v", decoded)
	}
	wantPath := requestWirePath(target)
	gotPath := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]
	if gotPath != wantPath {
		t.Fatalf("dispatch path = %q, want %q", gotPath, wantPath)
	}
}

func protocolRequestFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"virtual","messages":[{"role":"user","content":"hello"}],"max_tokens":8}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"virtual","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"max_output_tokens":8}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"virtual","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)
	default:
		panic(fmt.Sprintf("unsupported fixture format %q", format))
	}
}

func TestProviderProtocolPath(t *testing.T) {
	tests := []struct {
		name         string
		baseURL      string
		protocolPath string
		want         string
	}{
		{name: "root", baseURL: "https://api.example.com", protocolPath: "/v1/messages", want: "/v1/messages"},
		{name: "version root", baseURL: "https://api.example.com/v1", protocolPath: "/v1/responses", want: "/v1/responses"},
		{name: "nested version root", baseURL: "https://api.example.com/openai/v1", protocolPath: "/v1/responses", want: "/openai/v1/responses"},
		{name: "custom base", baseURL: "https://api.example.com/proxy", protocolPath: "/v1/messages", want: "/proxy/v1/messages"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := providerProtocolPath(test.baseURL, test.protocolPath); got != test.want {
				t.Fatalf("providerProtocolPath(%q, %q) = %q, want %q", test.baseURL, test.protocolPath, got, test.want)
			}
		})
	}
}

func TestSetProviderRequestPathCoversProtocolAndBaseURLMatrix(t *testing.T) {
	tests := []struct {
		name    string
		profile config.ProviderProfile
		format  llmprotocol.WireFormat
		want    string
	}{
		{
			name:    "Chat OpenAI version root",
			profile: config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/v1"},
			format:  llmprotocol.OpenAIChatV1,
			want:    "/v1/chat/completions",
		},
		{
			name:    "Chat OpenAI nested version root",
			profile: config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/compatible/v1"},
			format:  llmprotocol.OpenAIChatV1,
			want:    "/compatible/v1/chat/completions",
		},
		{
			name:    "Chat explicit override",
			profile: config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/v1", ChatPath: "/custom/chat"},
			format:  llmprotocol.OpenAIChatV1,
			want:    "/custom/chat",
		},
		{
			name:    "Responses version root",
			profile: config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/v1"},
			format:  llmprotocol.OpenAIResponsesV1,
			want:    "/v1/responses",
		},
		{
			name:    "Responses nested version root ignores chat override",
			profile: config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/compatible/v1", ChatPath: "/custom/chat"},
			format:  llmprotocol.OpenAIResponsesV1,
			want:    "/compatible/v1/responses",
		},
		{
			name:    "Anthropic version root",
			profile: config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.example.com/v1"},
			format:  llmprotocol.AnthropicMessagesV1,
			want:    "/v1/messages",
		},
		{
			name:    "Anthropic nested version root ignores chat override",
			profile: config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.example.com/compatible/v1", ChatPath: "/custom/chat"},
			format:  llmprotocol.AnthropicMessagesV1,
			want:    "/compatible/v1/messages",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var mutations []*core.HeaderValueOption
			setProviderRequestPath(&mutations, &test.profile, test.format)
			if got := headerValuesByName(mutations)[":path"]; got != test.want {
				t.Fatalf("provider path = %q, want %q", got, test.want)
			}
		})
	}
}

func routingTestRouterForFormat(format llmprotocol.WireFormat) (*OpenAIRouter, string) {
	logicalModel := "target-" + string(format)
	apiFormat := config.APIFormatOpenAI
	providerType := "openai"
	baseURL := "http://127.0.0.1:8000/v1"
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		apiFormat = config.APIFormatResponses
	case llmprotocol.AnthropicMessagesV1:
		apiFormat = config.APIFormatAnthropic
		providerType = "anthropic"
		baseURL = "http://127.0.0.1:8000"
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: logicalModel,
			ModelConfig: map[string]config.ModelParams{
				logicalModel: {
					PreferredEndpoints: []string{"backend"},
					APIFormat:          apiFormat,
					ExternalModelIDs:   map[string]string{"vllm": "provider-model"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: providerType, BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}, logicalModel
}

func TestHandleAutoModelRoutingEmitsSelectedModelAndEncodedBody(t *testing.T) {
	router := routingTestRouter("qwen14b-dev")
	request := testNeutralRequest("MoM", "hello from routed model")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality: ModalityBoth, Confidence: 0.97, Method: "signal",
	}

	response, err := router.handleEntrypointModelRouting(
		request, "MoM", "", entropy.ReasoningDecision{}, "qwen14b-dev", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	headersByName := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := headersByName[headers.SelectedModel]; got != "qwen14b-dev" {
		t.Fatalf("selected model header = %q", got)
	}
	var body map[string]any
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	if got := body["model"]; got != "qwen14b-dev" {
		t.Fatalf("logical model = %#v", got)
	}
	if request.Model != "qwen14b-dev" || ctx.SemanticRequest != request {
		t.Fatalf("neutral request was not updated in place: %+v", request)
	}
	if ctx.ModalityClassification.Modality != ModalityBoth {
		t.Fatalf("backend dispatch lost modality evidence: %+v", ctx.ModalityClassification)
	}
}

func TestHandleAutoModelRoutingSameModelEncodesCurrentSemanticRequest(t *testing.T) {
	router := routingTestRouter("auto")
	request := testNeutralRequest("auto", "short semantic content")
	request.Generation = 4
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	response, err := router.handleEntrypointModelRouting(
		request, "auto", "", entropy.ReasoningDecision{}, "auto", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if body.Model != "auto" || len(body.Messages) != 1 || body.Messages[0].Content != "short semantic content" {
		t.Fatalf("unexpected encoded semantic request: %+v", body)
	}
}

func TestSpecifiedModelPreservesAnthropicWireAtExtProcBoundary(t *testing.T) {
	router := routingTestRouter("test-model")
	request := testNeutralRequest("test-model", "Explain the incident.")
	request.Stream = true
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true

	response, err := router.handleSpecifiedModelRouting(request, "test-model", "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}
	if _, ok := body["messages"]; !ok {
		t.Fatalf("Anthropic request omitted messages: %s", common.GetBodyMutation().GetBody())
	}
	if string(body["stream"]) != "true" || string(body["max_tokens"]) != "256" {
		t.Fatalf("Anthropic options were not preserved: %s", common.GetBodyMutation().GetBody())
	}
	if got := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]; got != "/v1/messages" {
		t.Fatalf("ExtProc source-format path = %q", got)
	}
}

func routingTestRouter(model string) *OpenAIRouter {
	apiFormat := config.APIFormatOpenAI
	profileType := "openai"
	baseURL := "http://127.0.0.1:8000/v1"
	if model == "test-model" {
		apiFormat = config.APIFormatAnthropic
		profileType = "anthropic"
		baseURL = "http://127.0.0.1:8000"
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: model,
			ModelConfig: map[string]config.ModelParams{
				model: {PreferredEndpoints: []string{"backend"}, APIFormat: apiFormat},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: profileType, BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
}

func routingTestContext(format llmprotocol.WireFormat, request *llmprotocol.Request) *RequestContext {
	return &RequestContext{
		Headers:         map[string]string{},
		SourceFormat:    format,
		SemanticRequest: request,
		RequestID:       "routing-test-request",
		TraceContext:    context.Background(),
	}
}
