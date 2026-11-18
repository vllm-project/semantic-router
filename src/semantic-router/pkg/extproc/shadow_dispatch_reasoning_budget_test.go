package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestShadowChatRequestProjectsReasoningBudgetForVLLM(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	profile := config.ProviderProfile{Type: "vllm", BaseURL: "http://127.0.0.1:8000/v1"}
	router.Config.ProviderProfiles["provider"] = profile
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	engine, err := router.protocolEngine()
	if err != nil {
		t.Fatalf("protocol engine: %v", err)
	}
	encode := router.shadowRequestEncoder(ctx, &providerDispatch{}, engine)
	budget := int64(1024)
	request := llmprotocol.Request{
		Model: model, ReasoningBudgetTokens: &budget,
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "shadow budget"}}}},
		Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIChatV1},
	}
	target := &shadowTarget{
		logicalModel: model, upstreamModel: "provider-model",
		profile: &profile, format: llmprotocol.OpenAIChatV1,
	}

	body, err := encode(request, target)
	if err != nil {
		t.Fatalf("encode shadow request: %v", err)
	}
	observed := unmarshalReasoningRequest(t, body)
	if observed["thinking_token_budget"] != float64(1024) {
		t.Fatalf("shadow thinking_token_budget = %v, want 1024: %s", observed["thinking_token_budget"], body)
	}
	if _, found := observed["reasoning_budget_tokens"]; found {
		t.Fatalf("shadow request retained the neutral budget: %s", body)
	}
}

func TestShadowChatRequestAppliesNonVLLMReasoningBudgetPolicy(t *testing.T) {
	for _, test := range []struct {
		name       string
		allow      bool
		legacyType bool
	}{
		{name: "strict", allow: false},
		{name: "opt-in", allow: true},
		{name: "legacy endpoint without profile", allow: true, legacyType: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			primaryProfile := router.Config.ProviderProfiles["provider"]
			primaryProfile.Type = "vllm"
			router.Config.ProviderProfiles["provider"] = primaryProfile
			if test.allow {
				router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldReasoningBudget}
			}
			shadowModel := "target-shadow-ollama"
			shadowEndpoint := config.VLLMEndpoint{
				Name: "shadow-backend", Address: "127.0.0.1", Port: 8001,
			}
			var shadowProfile *config.ProviderProfile
			if test.legacyType {
				shadowEndpoint.Type = "ollama"
			} else {
				profile := config.ProviderProfile{Type: "ollama", BaseURL: "http://127.0.0.1:8001/v1"}
				router.Config.ProviderProfiles["shadow-provider"] = profile
				shadowEndpoint.ProviderProfileName = "shadow-provider"
				shadowProfile = &profile
			}
			router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, shadowEndpoint)
			router.Config.ModelConfig[shadowModel] = config.ModelParams{
				PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI,
			}
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
			engine, err := router.protocolEngine()
			if err != nil {
				t.Fatalf("protocol engine: %v", err)
			}
			encode := router.shadowRequestEncoder(ctx, &providerDispatch{}, engine)
			budget := int64(1024)
			request := llmprotocol.Request{
				Model: model, ReasoningBudgetTokens: &budget,
				Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "shadow budget"}}}},
				Trusted:  llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.OpenAIChatV1},
			}
			target := &shadowTarget{
				logicalModel: shadowModel, upstreamModel: "shadow-model",
				profile: shadowProfile, format: llmprotocol.OpenAIChatV1,
			}

			body, err := encode(request, target)
			if !test.allow {
				if !hasReasoningBudgetLossError(err) {
					t.Fatalf("strict shadow encode error = %v, want actionable lossy_translation", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("opted-in shadow encode: %v", err)
			}
			observed := unmarshalReasoningRequest(t, body)
			if _, found := observed["reasoning_budget_tokens"]; found {
				t.Fatalf("opted-in shadow request retained the neutral budget: %s", body)
			}
			if _, found := observed["thinking_token_budget"]; found {
				t.Fatalf("non-vLLM shadow request received a vLLM budget: %s", body)
			}
			if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != budget {
				t.Fatalf("shadow projection changed the semantic request: %+v", request)
			}
		})
	}
}
