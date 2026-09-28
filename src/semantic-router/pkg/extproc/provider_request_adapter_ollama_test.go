package extproc

import (
	"encoding/json"
	"maps"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func routerWithChatProvider(providerType string) (*OpenAIRouter, string) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = providerType
	router.Config.ProviderProfiles["provider"] = profile
	return router, model
}

// dispatchBody runs the provider dispatch seam: codec encode, then the
// provider adapter, exactly as finalizeProviderDispatchResponse does.
func dispatchBody(t *testing.T, providerType string, client llmprotocol.WireFormat) map[string]json.RawMessage {
	t.Helper()
	router, model := routerWithChatProvider(providerType)
	request := testNeutralRequest(model, "Count from 1 to 50.")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8)
	ctx := routingTestContext(client, request)
	dispatch, err := router.prepareProviderDispatch(request, model, "default-route", false, ctx)
	if err != nil {
		t.Fatal(err)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	body, err = router.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	return fields
}

// Ollama's OpenAI layer maps only max_tokens to num_predict, so an output
// limit sent as max_completion_tokens is ignored.
func TestOllamaDispatchCarriesOutputLimitAsMaxTokens(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(client), func(t *testing.T) {
			fields := dispatchBody(t, "ollama", client)
			if string(fields["max_tokens"]) != "8" {
				t.Fatalf("Ollama dispatch has no max_tokens of 8; body keys %v", slices.Sorted(maps.Keys(fields)))
			}
			if _, ok := fields["max_completion_tokens"]; ok {
				t.Fatal("Ollama dispatch still carries max_completion_tokens")
			}
		})
	}
}

func TestOpenAIDispatchKeepsMaxCompletionTokens(t *testing.T) {
	fields := dispatchBody(t, "openai", llmprotocol.OpenAIChatV1)
	if string(fields["max_completion_tokens"]) != "8" {
		t.Fatalf("OpenAI dispatch max_completion_tokens = %s", fields["max_completion_tokens"])
	}
	if _, ok := fields["max_tokens"]; ok {
		t.Fatal("OpenAI dispatch gained max_tokens")
	}
}

func TestOllamaShadowRequestCarriesOutputLimitAsMaxTokens(t *testing.T) {
	router, model := routerWithChatProvider("ollama")
	request := testNeutralRequest(model, "Count from 1 to 50.")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	dispatch, err := router.prepareProviderDispatch(request, model, "default-route", false, ctx)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := router.protocolEngine()
	if err != nil {
		t.Fatal(err)
	}
	encode := router.shadowRequestEncoder(ctx, dispatch, engine)
	body, err := encode(*request, &shadowTarget{
		logicalModel: model, upstreamModel: "shadow-model", format: llmprotocol.OpenAIChatV1,
		profile: &config.ProviderProfile{Type: "ollama"},
	})
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields["max_tokens"]) != "8" {
		t.Fatalf("Ollama shadow request has no max_tokens of 8; body keys %v", slices.Sorted(maps.Keys(fields)))
	}
	if _, ok := fields["max_completion_tokens"]; ok {
		t.Fatal("Ollama shadow request still carries max_completion_tokens")
	}
}
