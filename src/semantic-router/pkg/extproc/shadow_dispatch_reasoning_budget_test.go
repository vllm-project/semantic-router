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
