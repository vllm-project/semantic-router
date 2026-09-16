package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestPrepareProviderDispatchChecksPrimaryModelTaskCapabilities(t *testing.T) {
	for _, capabilities := range [][]string{
		{"image_input"},
		{"chat", "vision", "structured_output", "long_context"},
		{"image_input", "custom_descriptive_label"},
	} {
		for _, fallback := range []bool{false, true} {
			router, primary := routingTestRouterForFormat(llmprotocol.OpenAIResponsesV1)
			params := router.Config.ModelConfig[primary]
			params.Capabilities = capabilities
			router.Config.ModelConfig[primary] = params
			decision := &config.Decision{Name: "generation", ModelRefs: []config.ModelRef{{Model: primary}}}
			if fallback {
				params.Capabilities = []string{"image_generation"}
				params.ExternalModelIDs = map[string]string{"vllm": "provider-generator"}
				router.Config.ModelConfig["generator"] = params
				decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: "generator"})
			}
			request := testNeutralRequest(primary, "draw a cat")
			request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
			ctx.VSRSelectedDecision = decision
			dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
			if fallback {
				if err != nil || dispatch.logicalModel != "generator" || ctx.RequestModel != "generator" {
					t.Fatalf("capabilities=%v: dispatch=%+v error=%v", capabilities, dispatch, err)
				}
				if ctx.ImmediateProtocolError != nil {
					t.Fatal("successful same-wire reroute retained the primary rejection")
				}
			} else {
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" || dispatch != nil {
					t.Fatalf("capabilities=%v: dispatch=%+v error=%v, want capability rejection", capabilities, dispatch, err)
				}
			}
		}
	}
}

func TestPrepareProviderDispatchPreservesContextEligibilityOnReroute(t *testing.T) {
	for _, largeFallback := range []bool{false, true} {
		router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
		params := router.Config.ModelConfig[primary]
		params.ContextWindowSize = 32000
		router.Config.ModelConfig[primary] = params
		params.APIFormat = config.APIFormatResponses
		params.Capabilities = []string{"image_generation"}
		params.ContextWindowSize = 1000
		router.Config.ModelConfig["too-small"] = params
		decision := &config.Decision{Name: "context-gated", ModelRefs: []config.ModelRef{{Model: primary}, {Model: "too-small"}}}
		if largeFallback {
			params.ContextWindowSize = 32000
			router.Config.ModelConfig["large-generator"] = params
			decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: "large-generator"})
		}
		request := testNeutralRequest(primary, "draw a cat")
		request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
		ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
		ctx.VSRSelectedDecision = decision
		ctx.VSRContextTokenCount = 8000
		if _, err := router.contextEligibleDecisionModelRefs(decision.ModelRefs, decision.Name, ctx.VSRContextTokenCount, ctx); err != nil {
			t.Fatal(err)
		}
		dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
		if largeFallback {
			if err != nil || dispatch.logicalModel != "large-generator" {
				t.Fatalf("dispatch=%+v error=%v, want eligible large generator", dispatch, err)
			}
		} else if err == nil || dispatch != nil {
			t.Fatalf("excluded small model resurrected: dispatch=%+v error=%v", dispatch, err)
		}
	}
}

func TestPrepareProviderDispatchDoesNotExpandFilteredInventory(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	params := router.Config.ModelConfig[primary]
	params.APIFormat = config.APIFormatResponses
	router.Config.ModelConfig["excluded"] = params
	request := testNeutralRequest(primary, "draw a cat")
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &config.Decision{Name: "filtered", ModelRefs: []config.ModelRef{{Model: primary}, {Model: "excluded"}}}
	ctx.VSREligibleModelRefs = []config.ModelRef{{Model: primary}}
	if dispatch, err := router.prepareProviderDispatch(request, primary, "filtered", false, ctx); err == nil || dispatch != nil {
		t.Fatalf("filtered candidate resurrected: dispatch=%+v error=%v", dispatch, err)
	}
}
