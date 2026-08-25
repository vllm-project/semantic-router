package managementserver

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func TestProviderCatalogReadContractsExposeRevisionWithoutExecutorInternals(t *testing.T) {
	definition := providercatalog.Definition{
		ID: "provider-a", Revision: "sha256:" + strings.Repeat("a", 64),
		Display: providercatalog.Display{
			Name: "Provider A", Description: "A provider.", Category: "Model APIs",
			Icon: providercatalog.Icon{Source: "lobe", Value: "provider-a", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "default", Label: "Default", Default: true, WireFormat: "private.wire.v1",
			Compiler: providercatalog.Compiler{
				AdapterID: providercatalog.StaticBackendCompilerID,
				Config: map[string]any{
					"path": "/invoke", "headers": map[string]any{"X-Internal-Invocation": "hidden"},
				},
			},
		}},
		Credential: providercatalog.Credential{
			Mode: providercatalog.CredentialRequired, AdapterID: "credential-adapter", Label: "API key",
		},
		Origin:       providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com/v1"},
		Discovery:    &providercatalog.Discovery{AdapterID: "discovery-adapter", Path: "/models"},
		Capabilities: []string{"streaming", "tools"},
		ConnectionFields: []providercatalog.ConnectionField{{
			Name: "region", Label: "Region", Kind: providercatalog.FieldSelect,
			Options: []providercatalog.FieldOption{{Value: "east", Label: "East"}},
		}},
	}
	page := providerCatalogPageDTO(providercatalog.ListResult{
		CatalogRevision: "sha256:revision", Providers: []providercatalog.Definition{definition},
		Categories: []string{"Model APIs"}, PageSize: 50,
	})
	encoded, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}
	wire := string(encoded)
	for _, required := range []string{
		`"revision":"sha256:` + strings.Repeat("a", 64) + `"`,
		`"discoverySupported":true`,
		`"icon":{"source":"lobe","value":"provider-a","color":false}`,
		`"interfaces":[{"id":"default","label":"Default","default":true,"capabilities":[]}]`,
	} {
		if !strings.Contains(wire, required) {
			t.Errorf("provider response %s does not contain %s", wire, required)
		}
	}
	for _, forbidden := range []string{"private.wire.v1", providercatalog.StaticBackendCompilerID, "credential-adapter", "discovery-adapter", "X-Internal-Invocation", "/invoke", "/models"} {
		if strings.Contains(wire, forbidden) {
			t.Errorf("provider response exposed executor-only value %q: %s", forbidden, wire)
		}
	}
	page.Data[0].Capabilities[0] = "mutated"
	page.Data[0].ConnectionFields[0].Options[0].Value = "mutated"
	if definition.Capabilities[0] == "mutated" || definition.ConnectionFields[0].Options[0].Value == "mutated" {
		t.Fatal("Management DTO retained mutable catalog storage")
	}
}

func TestDiscoveredModelCapabilitiesRequireModelSpecificEvidence(t *testing.T) {
	result := providerdiscovery.Result{Models: []providerdiscovery.Model{
		{CatalogItemID: "item-unknown", ProviderModelID: "unknown", DisplayName: "Unknown"},
		{
			CatalogItemID: "item-described", ProviderModelID: "described", DisplayName: "Described",
			Capabilities: []string{"image_input", "tools"},
		},
	}}
	encoded, err := json.Marshal(discoveredModelsPageDTO(result, 50))
	if err != nil {
		t.Fatal(err)
	}
	wire := string(encoded)
	if strings.Contains(wire, `"providerModelId":"unknown","displayName":"Unknown","capabilities"`) {
		t.Fatalf("unknown Model received a capability field: %s", wire)
	}
	if !strings.Contains(wire, `"providerModelId":"described","displayName":"Described","capabilities":["image_input","tools"]`) {
		t.Fatalf("model-specific capabilities were not preserved: %s", wire)
	}
}

func TestProviderDiscoveryRequestPreservesTypedConnectionPrimitives(t *testing.T) {
	request := managementapi.DiscoverModelsRequest{
		CredentialID: "22222222-2222-4222-8222-222222222222",
		BaseURL:      "https://api.example.com",
		ConnectionFields: map[string]json.RawMessage{
			"region": json.RawMessage(`"east"`), "preview": json.RawMessage(`true`),
			"shard": json.RawMessage(`9007199254740993`),
		},
	}
	converted, err := providerDiscoveryRequest(request, "11111111-1111-4111-8111-111111111111")
	if err != nil {
		t.Fatal(err)
	}
	if converted.ConnectionFields["region"] != "east" || converted.ConnectionFields["preview"] != true {
		t.Fatalf("converted connection fields = %#v", converted.ConnectionFields)
	}
	if number, ok := converted.ConnectionFields["shard"].(json.Number); !ok || string(number) != "9007199254740993" {
		t.Fatalf("integer connection field = %#v, want exact json.Number", converted.ConnectionFields["shard"])
	}
	request.ConnectionFields["shard"] = json.RawMessage(`1 2`)
	if _, err := providerDiscoveryRequest(request, "11111111-1111-4111-8111-111111111111"); err == nil {
		t.Fatal("connection field with trailing JSON was accepted")
	}
}
