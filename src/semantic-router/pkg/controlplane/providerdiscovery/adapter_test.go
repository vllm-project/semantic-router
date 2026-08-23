package providerdiscovery

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestOpenAIAdapterProvidesDeterministicLocalPagination(t *testing.T) {
	plan := providercatalog.DiscoveryPlan{
		DiscoveryAdapterID: openAIModelsAdapterID, NormalizedOrigin: "https://example.com/v1",
		Path: "/models", PageSize: 2, Search: "model", ProviderCursor: "model-a",
	}
	page, err := (OpenAIModelsAdapter{}).Decode(plan, strings.NewReader(
		`{"object":"list","data":[{"id":"model-c"},{"id":"other"},{"id":"model-b"},{"id":"model-a"}]}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Models) != 2 || page.Models[0].ProviderModelID != "model-b" ||
		page.Models[1].ProviderModelID != "model-c" || page.HasMore {
		t.Fatalf("OpenAI page = %+v", page)
	}
}

func TestAnthropicAdapterUsesOpaqueProviderCursor(t *testing.T) {
	plan := providercatalog.DiscoveryPlan{
		DiscoveryAdapterID: anthropicModelsAdapterID, NormalizedOrigin: "https://api.anthropic.com",
		Path: "/v1/models", PageSize: 25, ProviderCursor: "model-before",
	}
	query, err := (AnthropicModelsAdapter{}).Query(plan)
	if err != nil {
		t.Fatal(err)
	}
	if query.Encode() != (url.Values{"after_id": {"model-before"}, "limit": {"25"}}).Encode() {
		t.Fatalf("Anthropic query = %s", query.Encode())
	}
	page, err := (AnthropicModelsAdapter{}).Decode(plan, strings.NewReader(
		`{"data":[{"id":"claude-a","display_name":"Claude A","type":"model"}],"has_more":true,"last_id":"claude-a"}`,
	))
	if err != nil || len(page.Models) != 1 || page.NextCursor != "claude-a" || !page.HasMore {
		t.Fatalf("Anthropic page = %+v, err = %v", page, err)
	}
}

func TestBuiltinRegistryAlsoValidatesCatalogDiscoveryPlans(t *testing.T) {
	registry, err := BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	validators, err := providercatalog.NewDiscoveryRegistry(registry.Validators())
	if err != nil || validators == nil {
		t.Fatalf("catalog validator registry = %+v, err = %v", validators, err)
	}
	adapter, _ := registry.Adapter(openAIModelsAdapterID)
	if err := adapter.ValidateDiscovery(context.Background(), providercatalog.DiscoveryPlan{
		DiscoveryAdapterID: openAIModelsAdapterID, NormalizedOrigin: "https://example.com",
		Path: "/models", PageSize: 10,
	}); err != nil {
		t.Fatal(err)
	}
}
