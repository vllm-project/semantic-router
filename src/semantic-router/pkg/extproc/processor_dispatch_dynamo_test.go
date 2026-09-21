package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestFinalDynamoDispatchAfterCapabilityReroute(t *testing.T) {
	for _, backendType := range []string{"vllm", "dynamo"} {
		for _, source := range []string{"nvext", "header"} {
			t.Run(backendType+"/"+source, func(t *testing.T) {
				router, primary := routingTestRouterForFormat(llmprotocol.OpenAIResponsesV1)
				params := router.Config.ModelConfig[primary]
				params.Capabilities = []string{"image_input"}
				router.Config.ModelConfig[primary] = params
				router.Config.VLLMEndpoints[0].Type = "dynamo"
				fallback := "fallback-responses"
				endpoint := router.Config.VLLMEndpoints[0]
				endpoint.Name, endpoint.Type = "fallback", backendType
				router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, endpoint)
				router.Config.ModelConfig[fallback] = config.ModelParams{
					PreferredEndpoints: []string{"fallback"}, APIFormat: config.APIFormatResponses,
				}
				request := testNeutralRequest(primary, "draw a cat")
				request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
				ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
				ctx.ProtocolEnvelope.Format = llmprotocol.OpenAIResponsesV1
				switch source {
				case "nvext":
					ctx.ProtocolEnvelope.Dynamo = &llmprotocol.DynamoEnvelope{RequestNVExt: &llmprotocol.DynamoRequestNVExt{GreedSampling: llmprotocol.Bool(true)}}
				case "header":
					ctx.Headers[headers.DynamoDPRank] = "1"
				}
				ctx.VSRSelectedDecision = &config.Decision{Name: "reroute", ModelRefs: []config.ModelRef{{Model: primary}, {Model: fallback}}}
				if err := validateDynamoBackendPool(router.Config, primary, ctx, ctx.ProtocolEnvelope); err != nil {
					t.Fatalf("initial Dynamo target rejected: %v", err)
				}
				dispatch, err := router.prepareProviderDispatch(request, primary, "reroute", false, ctx)
				if err != nil {
					t.Fatal(err)
				}
				if dispatch.logicalModel != fallback {
					t.Fatalf("expected actual reroute, got %s", dispatch.logicalModel)
				}
				wantCode := ""
				if backendType != "dynamo" {
					wantCode = "unsupported_dynamo_nvext_backend"
				}
				assertFinalDynamoDispatch(t, router, dispatch, ctx, wantCode)
			})
		}
	}
}

func TestFinalDynamoDispatchWithConfiguredHeaders(t *testing.T) {
	for _, source := range []string{"profile", "decision"} {
		for _, tc := range []struct{ name, backendType, value, wantCode string }{
			{"non-dynamo", "vllm", "1", "unsupported_dynamo_nvext_backend"},
			{"dynamo", "dynamo", "1", ""},
			{"invalid-value", "dynamo", "-1", "invalid_dynamo_routing_header"},
		} {
			t.Run(source+"/"+tc.name, func(t *testing.T) {
				router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
				router.Config.VLLMEndpoints[0].Type = tc.backendType
				request := testNeutralRequest(model, "hello")
				ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
				if source == "profile" {
					profile := router.Config.ProviderProfiles["provider"]
					profile.ExtraHeaders = map[string]string{headers.DynamoDPRank: tc.value}
					router.Config.ProviderProfiles["provider"] = profile
				} else {
					payload, err := config.NewStructuredPayload(map[string]interface{}{
						"update": []map[string]string{{"name": headers.DynamoDPRank, "value": tc.value}},
					})
					if err != nil {
						t.Fatal(err)
					}
					ctx.VSRSelectedDecision = &config.Decision{Name: "headers", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginHeaderMutation, Configuration: payload}}}
				}
				if hasDynamoRequestExtension(ctx, ctx.ProtocolEnvelope) {
					t.Fatal("original request unexpectedly contains Dynamo state")
				}
				if err := validateDynamoBackendPool(router.Config, model, ctx, ctx.ProtocolEnvelope); err != nil {
					t.Fatal(err)
				}
				dispatch, err := router.prepareProviderDispatch(request, model, "", false, ctx)
				if err != nil {
					t.Fatal(err)
				}
				assertFinalDynamoDispatch(t, router, dispatch, ctx, tc.wantCode)
			})
		}
	}
}

func assertFinalDynamoDispatch(t *testing.T, router *OpenAIRouter, dispatch *providerDispatch, ctx *RequestContext, wantCode string) {
	t.Helper()
	response := router.buildProviderDispatchResponse(dispatch, ctx)
	finalized, err := router.finalizeProviderDispatchResponse(dispatch, response, ctx)
	if wantCode == "" {
		if err != nil {
			t.Fatalf("valid Dynamo dispatch rejected: %v", err)
		}
		if len(finalized.GetRequestBody().GetResponse().GetBodyMutation().GetBody()) == 0 {
			t.Fatal("missing outbound body")
		}
		return
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != wantCode {
		t.Fatalf("error = %v, want %s", err, wantCode)
	}
	if finalized != nil {
		t.Fatal("rejected request was released")
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != wantCode {
		t.Fatalf("immediate error = %v", ctx.ImmediateProtocolError)
	}
	rejected, converted := router.processBodyRoutingError(err, ctx)
	if !converted || rejected.GetImmediateResponse().GetStatus().GetCode() != 400 {
		t.Fatal("expected HTTP 400")
	}
}
