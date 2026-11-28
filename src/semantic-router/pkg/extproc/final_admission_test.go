package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestFinalAdmissionRefusesMaskedOverflow(t *testing.T) {
	r, ctx := overflowFixture(t, strings.Repeat("x ", 30))
	r.Config.CandidateRequirements = nil
	decision := ctx.VSRSelectedDecision
	decision.Plugins = maskingDecision(true).Plugins
	model := decision.ModelRefs[0].Model
	const reserve = 12
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(reserve)
	useMaskingDetector(t, &stubPIIDetector{values: []string{"x"}})

	tokenizer, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	before := countPromptTokens(tokenizer, ctx.SemanticRequest)
	params := r.Config.ModelConfig[model]
	params.ContextWindowSize = before + reserve + 1
	r.Config.ModelConfig[model] = params
	eligible, err := r.decisionEligibleModelRefs(decision, ctx)
	require.NoError(t, err)
	require.Len(t, eligible, 1)

	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "vllm-sr/auto", decision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	require.EqualValues(t, 400, response.GetImmediateResponse().GetStatus().GetCode())
	require.Equal(t, "context_length_exceeded", ctx.ImmediateProtocolError.Code)
	require.Nil(t, response.GetRequestBody())
}

func TestFinalAdmissionBudgetsMaskedAutomaticOutput(t *testing.T) {
	text := strings.Repeat("x ", 30)
	calls := 0
	r, ctx := automaticFixture(t, text, renderMock(t, &calls, 0, 0))
	decision := ctx.VSRSelectedDecision
	decision.Plugins = append(decision.Plugins, maskingDecision(true).Plugins...)
	model := decision.ModelRefs[0].Model
	useMaskingDetector(t, &stubPIIDetector{values: []string{"x"}})

	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "vllm-sr/auto"))
	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "vllm-sr/auto", decision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	wire := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	dispatched, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(wire, llmprotocol.DefaultPolicy())
	require.NoError(t, err)
	masked := dispatched.Messages[0].Content[0].Text
	require.Contains(t, masked, "[EMAIL_ADDRESS_0]")
	// renderMock counts each message rune plus a framing constant.
	maskedInput := 17 + len([]rune(masked))
	window := r.Config.ModelConfig[model].ContextWindowSize
	require.EqualValues(t, window-maskedInput, *dispatched.Sampling.MaxOutputTokens)
	require.EqualValues(t, maskedInput, *ctx.SemanticRequest.Sampling.AutomaticInputTokens)
}

func TestFinalAdmissionRendersAfterMaskingCanShrinkLateInput(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hi", renderMock(t, &calls, 0, 0))
	decision := ctx.VSRSelectedDecision
	decision.Plugins = append(decision.Plugins[1:], maskingDecision(true).Plugins...)
	model := decision.ModelRefs[0].Model
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "vllm-sr/auto"))

	late := strings.Repeat("ordinary account text ", 1900)
	detector := &stubPIIDetector{values: []string{late}}
	useMaskingDetector(t, detector)
	ctx.SemanticRequest.Messages[0].Content[0].Text = late
	ctx.SemanticRequest.Generation++

	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "vllm-sr/auto", decision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	require.Equal(t, 2, calls)
	require.Equal(t, 1, detector.calls)
	wire := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	require.Contains(t, string(wire), "[EMAIL_ADDRESS_0]")
	require.Equal(t, "provider_render", ctx.finalAdmission.Source)
	require.Equal(t, 17+len("[EMAIL_ADDRESS_0]"), ctx.finalAdmission.PromptTokens)
}

func TestFinalAdmissionCompressesLateAutomaticOverflow(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "HEAD private-value", renderMock(t, &calls, 0, 0))
	decision := ctx.VSRSelectedDecision
	decision.Plugins = append(decision.Plugins, maskingDecision(true).Plugins...)
	model := decision.ModelRefs[0].Model
	detector := &stubPIIDetector{values: []string{"private-value"}}
	useMaskingDetector(t, detector)

	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "vllm-sr/auto"))
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, model, decision.Name, false, ctx)
	require.NoError(t, err)
	require.False(t, ctx.ContextCompressionApplied)
	ctx.SemanticRequest.Messages[0].Content[0].Text += strings.Repeat(" late content", 4000) + " TAIL"
	ctx.SemanticRequest.Generation++

	response, err := r.finalizeProviderDispatchResponse(dispatch, buildRequestBodyContinueResponse(&routeHeaderState{}, nil, false), ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	require.True(t, ctx.ContextCompressionApplied)
	require.Equal(t, 1, detector.calls)
	require.Equal(t, 3, calls)
	require.Equal(t, "fits", ctx.finalAdmission.Verdict)
	require.Zero(t, ctx.finalAdmission.ExcludedCandidates)

	wire := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	dispatched, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(wire, llmprotocol.DefaultPolicy())
	require.NoError(t, err)
	text := dispatched.Messages[0].Content[0].Text
	require.Contains(t, text, "[EMAIL_ADDRESS_0]")
	require.NotContains(t, text, "private-value")
	require.Contains(t, text, "context omitted by route compression")
	require.True(t, strings.HasSuffix(text, " TAIL"))
	require.Equal(t, 32768, 17+len([]rune(text))+int(*dispatched.Sampling.MaxOutputTokens))
}

func TestFallbackFinalAdmissionCompressesAutomaticOverflow(t *testing.T) {
	calls := 0
	text := "HEAD" + strings.Repeat(" long passage", 4000) + " TAIL"
	r, ctx := automaticFixture(t, text, renderMock(t, &calls, 0, 0))
	decision := ctx.VSRSelectedDecision
	narrow := decision.ModelRefs[0]
	wide := r.Config.ModelConfig[narrow.Model]
	wide.ContextWindowSize = 65536
	wide.MaxOutputTokens = 65536
	wide.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wide
	decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: "wide"})

	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, "wide", decision.Name, false, ctx)
	require.NoError(t, err)
	_, err = r.finalizeProviderDispatchResponse(dispatch, buildRequestBodyContinueResponse(&routeHeaderState{}, nil, false), ctx)
	require.NoError(t, err)
	require.False(t, ctx.ContextCompressionApplied)
	ctx.FallbackRequest, err = cloneSemanticRequestForReplay(ctx.SemanticRequest)
	require.NoError(t, err)

	candidate, _, err := r.prepareFallbackCandidate(t.Context(), &narrow, ctx, false)
	require.NoError(t, err)
	require.Equal(t, 3, calls)
	require.Equal(t, text, ctx.FallbackRequest.Messages[0].Content[0].Text)
	dispatched, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(candidate.body, llmprotocol.DefaultPolicy())
	require.NoError(t, err)
	compressed := dispatched.Messages[0].Content[0].Text
	require.Contains(t, compressed, "context omitted by route compression")
	require.True(t, strings.HasSuffix(compressed, " TAIL"))
	require.Equal(t, 32768, 17+len([]rune(compressed))+int(*dispatched.Sampling.MaxOutputTokens))
}

func TestFinalAdmissionChecksMediaOutputReserve(t *testing.T) {
	for _, tt := range []struct {
		name    string
		reserve int64
		verdict string
	}{
		{"known overflow", 100, "exceeds"},
		{"no capacity for media", 64, "exceeds"},
		{"unknown prompt demand", 12, "unknown"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, ctx := overflowFixture(t, "Describe the image.")
			r.Config.CandidateRequirements = nil
			decision := ctx.VSRSelectedDecision
			decision.Plugins = nil
			model := decision.ModelRefs[0].Model
			params := r.Config.ModelConfig[model]
			params.ContextWindowSize = 64
			params.Capabilities = nil
			r.Config.ModelConfig[model] = params
			ctx.SemanticRequest.Model = model
			ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(tt.reserve)
			ctx.SemanticRequest.Messages[0].Content = append(ctx.SemanticRequest.Messages[0].Content, llmprotocol.Content{
				Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image.png",
			})

			response, err := r.handleSpecifiedModelRouting(ctx.SemanticRequest, model, decision.Name, ctx)
			require.NoError(t, err)
			require.Equal(t, tt.verdict, ctx.finalAdmission.Verdict)
			if tt.verdict == "exceeds" {
				require.Equal(t, "output_reserve", ctx.finalAdmission.Source)
				require.EqualValues(t, 400, response.GetImmediateResponse().GetStatus().GetCode())
				require.Equal(t, "context_length_exceeded", ctx.ImmediateProtocolError.Code)
				require.Nil(t, response.GetRequestBody())
				return
			}
			require.Nil(t, response.GetImmediateResponse())
			require.NotEmpty(t, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody())
		})
	}
}

func TestFinalAdmissionUsesReducedPrompt(t *testing.T) {
	raw := strings.Repeat("quarterly report ", 40)
	r, ctx := overflowFixture(t, raw)
	r.Config.CandidateRequirements = nil
	decision := ctx.VSRSelectedDecision
	decision.Plugins = maskingDecision(true).Plugins
	model := decision.ModelRefs[0].Model
	params := r.Config.ModelConfig[model]
	params.ContextWindowSize = 64
	r.Config.ModelConfig[model] = params
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	useMaskingDetector(t, &stubPIIDetector{values: []string{raw}})
	before, _, ok := r.contextTokens(ctx, nil, model)
	require.True(t, ok)
	require.Greater(t, before+12, 64)
	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "auto", decision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	wire := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	require.Contains(t, string(wire), "[EMAIL_ADDRESS_0]")
	require.NotContains(t, string(wire), "quarterly report")
}

func TestFinalAdmissionRespectsSelectorEligibility(t *testing.T) {
	r, ctx := overflowFixture(t, strings.Repeat("x ", 30))
	r.Config.CandidateRequirements = nil
	d := ctx.VSRSelectedDecision
	d.Plugins = maskingDecision(true).Plugins
	model := d.ModelRefs[0].Model
	params := r.Config.ModelConfig[model]
	r.Config.ModelConfig["wide"] = params
	params.ContextWindowSize = 64
	r.Config.ModelConfig[model] = params
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	d.ModelRefs = append(d.ModelRefs, config.ModelRef{Model: "wide"})
	ctx.VSREligibleModelRefs = d.ModelRefs
	ctx.VSRPolicyEligibleModelRefs = d.ModelRefs[:1]
	detector := &stubPIIDetector{values: []string{"x"}}
	useMaskingDetector(t, detector)
	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "auto", d.Name, entropy.ReasoningDecision{}, model, ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	require.Equal(t, 1, detector.calls)
	require.Nil(t, ctx.preparedDispatchReceipt)
	require.Equal(t, "exceeds", ctx.finalAdmission.Verdict)
	require.Equal(t, 1, ctx.finalAdmission.ExcludedCandidates)
}
