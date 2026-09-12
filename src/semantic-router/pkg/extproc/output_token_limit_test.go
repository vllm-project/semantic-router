package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outputtokens"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestPrepareProviderDispatchComposesPerModelRefLimit(t *testing.T) {
	router, low, high, decision := confidenceTokenLimitFixture()
	client := int64(8000)
	for _, attempt := range []struct {
		model     string
		wantLimit int64
	}{
		{model: low, wantLimit: 256},
		{model: high, wantLimit: 1024},
	} {
		request := testNeutralRequest(attempt.model, "hello")
		request.Sampling.MaxOutputTokens = llmprotocol.Int64(client)
		ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
		ctx.ClientMaxOutputTokens = llmprotocol.Int64(client)
		ctx.VSRSelectedDecision = decision

		if _, err := router.prepareProviderDispatch(request, attempt.model, "", false, ctx); err != nil {
			t.Fatalf("prepareProviderDispatch(%s): %v", attempt.model, err)
		}
		if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != attempt.wantLimit {
			t.Fatalf("%s MaxOutputTokens = %v, want %d", attempt.model, request.Sampling.MaxOutputTokens, attempt.wantLimit)
		}
		if ctx.EffectiveMaxOutputTokens == nil || *ctx.EffectiveMaxOutputTokens != attempt.wantLimit {
			t.Fatalf("%s effective = %v, want %d", attempt.model, ctx.EffectiveMaxOutputTokens, attempt.wantLimit)
		}
		if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourceModelRef {
			t.Fatalf("%s source = %q, want %s", attempt.model, ctx.EffectiveMaxOutputTokensSource, outputtokens.SourceModelRef)
		}
		if got := encodedChatMaxCompletionTokens(t, request); got != attempt.wantLimit {
			t.Fatalf("%s encoded max_completion_tokens = %d, want %d", attempt.model, got, attempt.wantLimit)
		}
	}
}

func TestPrepareProviderDispatchClientCannotWidenModelRef(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(100)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(100)
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "route",
		ModelRefs: []config.ModelRef{{
			Model:               model,
			MaxCompletionTokens: outputTokenTestInt(512),
		}},
	}

	if _, err := router.prepareProviderDispatch(request, model, "route", false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 100 {
		t.Fatalf("MaxOutputTokens = %v, want client 100", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourceClient {
		t.Fatalf("source = %q, want %s", ctx.EffectiveMaxOutputTokensSource, outputtokens.SourceClient)
	}
}

func TestPrepareProviderDispatchInjectsPluginCeilingWhenClientOmitted(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := outputTokenRequestParamsDecision(t, model, map[string]interface{}{
		"max_tokens_limit": 500,
	})
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	if _, err := router.prepareProviderDispatch(request, model, decision.Name, false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 500 {
		t.Fatalf("MaxOutputTokens = %v, want plugin 500", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourcePlugin {
		t.Fatalf("source = %q, want %s", ctx.EffectiveMaxOutputTokensSource, outputtokens.SourcePlugin)
	}
}

func TestPrepareProviderDispatchBlockedParamClearsClientBound(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := outputTokenRequestParamsDecision(t, model, map[string]interface{}{
		"blocked_params": []string{"max_tokens"},
	})
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(9000)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(9000)
	ctx.VSRSelectedDecision = decision

	if _, err := router.prepareProviderDispatch(request, model, decision.Name, false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.Sampling.MaxOutputTokens != nil {
		t.Fatalf("MaxOutputTokens = %v, want unset after blocked_params", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensFallback != outputtokens.FallbackBlockedParam {
		t.Fatalf("fallback = %q, want %s", ctx.EffectiveMaxOutputTokensFallback, outputtokens.FallbackBlockedParam)
	}
}

func TestPrepareProviderDispatchStageBoundCanOnlyNarrow(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(9000)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(256)
	ctx.AlgorithmStageMaxOutputTokens = llmprotocol.Int64(9000)
	ctx.VSRSelectedDecision = &config.Decision{
		Name:      "route",
		ModelRefs: []config.ModelRef{{Model: model}},
	}

	if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 256 {
		t.Fatalf("MaxOutputTokens = %v, want original client 256", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourceClient {
		t.Fatalf("source = %q, want %s", ctx.EffectiveMaxOutputTokensSource, outputtokens.SourceClient)
	}
}

func TestApplyDispatchOutputTokenLimitDropsUnsupportedWire(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(256)
	ctx.VSRSelectedDecision = &config.Decision{
		Name: "route",
		ModelRefs: []config.ModelRef{{
			Model:               model,
			MaxCompletionTokens: outputTokenTestInt(128),
		}},
	}
	dispatch := &providerDispatch{logicalModel: model, targetFormat: llmprotocol.OpenAIImagesV1}

	router.applyDispatchOutputTokenLimit(request, dispatch, ctx)
	if request.Sampling.MaxOutputTokens != nil {
		t.Fatalf("MaxOutputTokens = %v, want dropped on images wire", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensFallback != outputtokens.FallbackCodecUnsupported {
		t.Fatalf("fallback = %q, want %s", ctx.EffectiveMaxOutputTokensFallback, outputtokens.FallbackCodecUnsupported)
	}
	if ctx.EffectiveMaxOutputTokens != nil || ctx.EffectiveMaxOutputTokensSource != "" {
		t.Fatalf("effective bound must be cleared on codec fallback")
	}
}

func TestPrepareProviderDispatchRerouteUsesNewModelRefBound(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	fallback := "fallback-responses"
	router.Config.ModelConfig[fallback] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		ExternalModelIDs:   map[string]string{"vllm": "provider-fallback"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary, MaxCompletionTokens: outputTokenTestInt(256)},
			{Model: fallback, MaxCompletionTokens: outputTokenTestInt(1024)},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8000)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(8000)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if dispatch.logicalModel != fallback {
		t.Fatalf("logical model = %s, want %s", dispatch.logicalModel, fallback)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 1024 {
		t.Fatalf("MaxOutputTokens = %v, want rerouted model_ref 1024", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourceModelRef {
		t.Fatalf("source = %q, want %s", ctx.EffectiveMaxOutputTokensSource, outputtokens.SourceModelRef)
	}
}

func TestPrepareProviderDispatchDoesNotReapplyLooperReasoning(t *testing.T) {
	router, low, _, decision := confidenceTokenLimitFixture()
	request := testNeutralRequest(low, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8000)
	request.ReasoningMode = llmprotocol.ReasoningModeEnabled
	request.ReasoningEffort = "high"
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(8000)
	ctx.VSRSelectedDecision = decision

	if _, err := router.prepareProviderDispatch(request, low, "", false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeEnabled || request.ReasoningEffort != "high" {
		t.Fatalf("reasoning mutated at dispatch: mode=%q effort=%q", request.ReasoningMode, request.ReasoningEffort)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 256 {
		t.Fatalf("MaxOutputTokens = %v, want model_ref 256", request.Sampling.MaxOutputTokens)
	}
}

func TestPrepareProviderDispatchBlockedLooperClientLetsModelRefWin(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := outputTokenRequestParamsDecision(t, model, map[string]interface{}{
		"blocked_params": []string{"max_tokens"},
	})
	tokens := 1024
	decision.ModelRefs[0].MaxCompletionTokens = &tokens
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.LooperRequest = true
	ctx.Headers[headers.VSRLooperClientMaxOutputTokens] = "256"
	ctx.ClientMaxOutputTokens = llmprotocol.Int64(256)
	ctx.VSRSelectedDecision = decision

	parseLooperOutputTokenBoundHeaders(ctx)
	if ctx.AlgorithmStageMaxOutputTokens != nil {
		t.Fatalf("missing stage header must not populate AlgorithmStage, got %v", ctx.AlgorithmStageMaxOutputTokens)
	}

	if _, err := router.prepareProviderDispatch(request, model, decision.Name, false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 1024 {
		t.Fatalf("MaxOutputTokens = %v, want model_ref 1024 after blocked client", request.Sampling.MaxOutputTokens)
	}
	if ctx.EffectiveMaxOutputTokensSource != outputtokens.SourceModelRef {
		t.Fatalf("source = %q, want %s", ctx.EffectiveMaxOutputTokensSource, outputtokens.SourceModelRef)
	}
}

func TestParseLooperOutputTokenBoundHeaders(t *testing.T) {
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			headers.VSRLooperClientMaxOutputTokens: "256",
			headers.VSRLooperStageMaxOutputTokens:  "512",
		},
	}
	parseLooperOutputTokenBoundHeaders(ctx)
	if ctx.ClientMaxOutputTokens == nil || *ctx.ClientMaxOutputTokens != 256 {
		t.Fatalf("client bound = %v, want 256", ctx.ClientMaxOutputTokens)
	}
	if ctx.AlgorithmStageMaxOutputTokens == nil || *ctx.AlgorithmStageMaxOutputTokens != 512 {
		t.Fatalf("stage bound = %v, want 512", ctx.AlgorithmStageMaxOutputTokens)
	}
}

func TestBuildReplayRouteDiagnosticsCopiesOutputTokenLimit(t *testing.T) {
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, testNeutralRequest("m", "hello"))
	ctx.EffectiveMaxOutputTokens = llmprotocol.Int64(256)
	ctx.EffectiveMaxOutputTokensSource = outputtokens.SourceModelRef
	ctx.EffectiveMaxOutputTokensFallback = outputtokens.FallbackCodecUnsupported
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "m", "route", 0, 0)
	if diagnostics.EffectiveMaxOutputTokens == nil || *diagnostics.EffectiveMaxOutputTokens != 256 {
		t.Fatalf("replay effective = %v, want 256", diagnostics.EffectiveMaxOutputTokens)
	}
	if diagnostics.EffectiveMaxOutputTokensSource != outputtokens.SourceModelRef {
		t.Fatalf("replay source = %q", diagnostics.EffectiveMaxOutputTokensSource)
	}
	if diagnostics.EffectiveMaxOutputTokensFallback != outputtokens.FallbackCodecUnsupported {
		t.Fatalf("replay fallback = %q", diagnostics.EffectiveMaxOutputTokensFallback)
	}
}

func TestBuildLooperRequestForwardsBlockedClientTokenLimit(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	tokens := 1024
	decision := outputTokenRequestParamsDecision(t, model, map[string]interface{}{
		"blocked_params": []string{"max_tokens"},
	})
	decision.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmConfidence}
	decision.ModelRefs[0].MaxCompletionTokens = &tokens
	request := testNeutralRequest(model, "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	looperReq, errResp := router.buildLooperRequest(request, decision, ctx)
	if errResp != nil {
		t.Fatalf("buildLooperRequest error response: %+v", errResp)
	}
	if looperReq == nil || !looperReq.ClientMaxOutputTokensBlocked {
		t.Fatalf("looper request = %+v, want ClientMaxOutputTokensBlocked", looperReq)
	}
}

func TestParseLooperOutputTokenBoundHeadersIgnoresNonPositive(t *testing.T) {
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			headers.VSRLooperClientMaxOutputTokens: "0",
			headers.VSRLooperStageMaxOutputTokens:  "-4",
		},
	}
	parseLooperOutputTokenBoundHeaders(ctx)
	if ctx.ClientMaxOutputTokens != nil || ctx.AlgorithmStageMaxOutputTokens != nil {
		t.Fatalf("non-positive headers must be ignored: %+v", ctx)
	}
}

func confidenceTokenLimitFixture() (*OpenAIRouter, string, string, *config.Decision) {
	router, low := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	high := "confidence-model-high"
	lowParams := router.Config.ModelConfig[low]
	router.Config.ModelConfig[high] = config.ModelParams{
		PreferredEndpoints: append([]string(nil), lowParams.PreferredEndpoints...),
		APIFormat:          lowParams.APIFormat,
		ExternalModelIDs:   map[string]string{"vllm": "provider-high"},
	}
	decision := &config.Decision{
		Name: "confidence-route",
		ModelRefs: []config.ModelRef{
			{Model: low, MaxCompletionTokens: outputTokenTestInt(256)},
			{Model: high, MaxCompletionTokens: outputTokenTestInt(1024)},
		},
	}
	return router, low, high, decision
}

func outputTokenRequestParamsDecision(t *testing.T, model string, plugin map[string]interface{}) *config.Decision {
	t.Helper()
	payload, err := config.NewStructuredPayload(plugin)
	if err != nil {
		t.Fatalf("NewStructuredPayload: %v", err)
	}
	return &config.Decision{
		Name: "params-route",
		ModelRefs: []config.ModelRef{
			{Model: model},
		},
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: payload},
		},
	}
}

func encodedChatMaxCompletionTokens(t *testing.T, request *llmprotocol.Request) int64 {
	t.Helper()
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(
		llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{},
	)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded.Body, &body); err != nil {
		t.Fatalf("decode encoded body: %v", err)
	}
	value, ok := body["max_completion_tokens"].(float64)
	if !ok {
		t.Fatalf("encoded body missing max_completion_tokens: %s", encoded.Body)
	}
	return int64(value)
}

func outputTokenTestInt(value int) *int {
	return &value
}
