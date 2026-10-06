package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDynamoBoundariesUseSelectedSharedLoRAOwner(t *testing.T) {
	for _, selectedType := range []string{"vllm", "dynamo"} {
		for _, source := range []string{"nvext", "header"} {
			for _, boundary := range []string{"early", "final"} {
				t.Run(selectedType+"/"+source+"/"+boundary, func(t *testing.T) {
					router, original := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
					params := router.Config.ModelConfig[original]
					delete(router.Config.ModelConfig, original)
					params.LoRAs = []config.LoRAAdapter{{Name: "shared"}}
					params.PreferredEndpoints = []string{"base-a-backend"}
					router.Config.ModelConfig["base-a"] = params
					params.PreferredEndpoints = []string{"base-b-backend"}
					router.Config.ModelConfig["base-b"] = params
					otherType := "dynamo"
					if selectedType == "dynamo" {
						otherType = "vllm"
					}
					endpoint := router.Config.VLLMEndpoints[0]
					endpoint.Name, endpoint.Type = "base-a-backend", otherType
					router.Config.VLLMEndpoints = []config.VLLMEndpoint{endpoint}
					endpoint.Name, endpoint.Type = "base-b-backend", selectedType
					router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, endpoint)
					// The old global alias lookup selects base-a, but this request selected base-b.
					if pool := router.Config.GetEndpointsForModel("shared"); len(pool) != 1 || pool[0].Name != "base-a-backend" {
						t.Fatalf("fixture must resolve shared globally to base-a: %+v", pool)
					}
					request := testNeutralRequest("shared", "hello")
					ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
					ctx.ProtocolEnvelope.Format = llmprotocol.OpenAIChatV1
					candidate := config.ModelRef{Model: "base-b", LoRAName: "shared"}
					ctx.VSRSelectedCandidate = &candidate
					ctx.VSRSelectedDecision = &config.Decision{Name: "shared-lora", ModelRefs: []config.ModelRef{candidate}}
					if source == "nvext" {
						ctx.ProtocolEnvelope.Dynamo = &llmprotocol.DynamoEnvelope{RequestNVExt: &llmprotocol.DynamoRequestNVExt{GreedSampling: llmprotocol.Bool(true)}}
					} else {
						ctx.Headers[headers.DynamoDPRank] = "1"
					}
					wantCode := ""
					if selectedType != "dynamo" {
						wantCode = "unsupported_dynamo_nvext_backend"
					}
					if boundary == "early" {
						defer func() { inflight.End(ctx.InflightModel, ctx.InflightToken) }()
						response := router.runPostDecisionImmediateStages("shared", "shared", "shared-lora", ctx)
						if wantCode == "" {
							if response != nil {
								t.Fatalf("valid selected Dynamo owner rejected: %+v", response)
							}
						} else if response.GetImmediateResponse().GetStatus().GetCode() != 400 || ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != wantCode {
							t.Fatalf("non-Dynamo owner was not rejected: response=%+v error=%v", response, ctx.ImmediateProtocolError)
						}
						return
					}
					dispatch, err := router.prepareProviderDispatch(request, "shared", "shared-lora", false, ctx)
					if err != nil {
						t.Fatal(err)
					}
					if dispatch.logicalModel != "shared" || dispatch.effectiveBackendModel() != "base-b" || dispatch.backendName != "base-b-backend" {
						t.Fatalf("wrong selected owner: %+v", dispatch)
					}
					assertFinalDynamoDispatch(t, router, dispatch, ctx, wantCode)
				})
			}
		}
	}
}
