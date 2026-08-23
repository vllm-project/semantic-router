package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const humanAuthoringFixture = `
version: v0.4
billing_currency: USD
models:
  - name: local/primary
    card:
      description: Primary local model
      capabilities: [chat, tools]
      modality: text
    connections:
      - provider: private-test
        endpoint: http://model.example
        model: upstream-primary
    runtime:
      max_retries: 2
      request_timeout: 30s
      stream_timeout: 2m
    pricing:
      input_cost_per_million_tokens: "0.5"
recipes:
  - name: balance
    document:
      decisions:
        - name: simple
          rules: {operator: AND, conditions: []}
entrypoints:
  - name: vllm-sr/auto
    aliases: [auto]
    recipe: balance
    assignments:
      simple:
        models:
          - model: local/primary
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
`

func TestHumanV04AuthoringCompilesNamesToStrictSnapshot(t *testing.T) {
	parser := testAuthoringParser(t)
	cfg, err := parser.ParseYAMLBytes([]byte(humanAuthoringFixture))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.RoutingSnapshot == nil || len(cfg.RoutingSnapshot.Models) != 1 {
		t.Fatalf("compiled snapshot = %+v", cfg.RoutingSnapshot)
	}
	model := cfg.RoutingSnapshot.Models[0]
	if !strings.HasPrefix(model.ID, "mdl_") || model.Revision != 1 ||
		!strings.HasPrefix(model.CatalogRevision, "sha256:") || len(model.Backends) != 1 ||
		!strings.HasPrefix(model.Backends[0].ID, "be_") {
		t.Fatalf("machine identity was not compiled: %+v", model)
	}
	resolution, err := cfg.ResolveEntrypoint("auto", "", nil)
	if err != nil || resolution.Recipe == nil || resolution.Recipe.Name != "balance" {
		t.Fatalf("ResolveEntrypoint() = %+v, %v", resolution, err)
	}
	decision := resolution.Recipe.Profile.Decisions[0]
	if len(decision.ModelRefs) != 1 || decision.ModelRefs[0].Model != "local/primary" {
		t.Fatalf("compiled Decision assignment = %+v", decision.ModelRefs)
	}
}

func TestHumanV04AuthoringFailsClosedWithoutProviderCompiler(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(humanAuthoringFixture))
	if err == nil || !strings.Contains(err.Error(), "require an injected Provider Integration compiler") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestHumanV04AuthoringRejectsMachineFields(t *testing.T) {
	tests := []struct {
		name   string
		needle string
		value  string
	}{
		{"model id", "  - name: local/primary", "  - id: mdl_primary\n    name: local/primary"},
		{"catalog revision", "  - name: local/primary", "  - provider_catalog_revision: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n    name: local/primary"},
		{"compiled backends", "    connections:", "    backends: []\n    connections:"},
		{"recipe id", "  - name: balance", "  - id: rcp_balance\n    name: balance"},
		{"decision id", "        - name: simple", "        - id: dec_simple\n          name: simple"},
		{"entrypoint id", "  - name: vllm-sr/auto", "  - id: ep_auto\n    name: vllm-sr/auto"},
		{"model id reference", "          - model: local/primary", "          - model_id: mdl_primary"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			document := strings.Replace(humanAuthoringFixture, test.needle, test.value, 1)
			_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
			if err == nil || !strings.Contains(err.Error(), "unsupported v0.4 authoring fields") {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}
		})
	}
}

func testAuthoringParser(t *testing.T) *Parser {
	t.Helper()
	return NewParser(testAuthoringConnectionCompiler(t))
}

func testAuthoringConnectionCompiler(t *testing.T) providercatalog.AuthoringCompiler {
	t.Helper()
	privateTest := providercatalog.IntegrationFunc(func() providercatalog.Definition {
		return providercatalog.Definition{
			ID: "private-test",
			Display: providercatalog.Display{
				Name: "Private test", Description: "Private test endpoint", Category: "Test",
				Icon: providercatalog.Icon{Source: "lobe", Value: "openai"},
			},
			Interfaces: []providercatalog.Interface{{
				ID: "chat", Label: "Chat Completions", Default: true, WireFormat: "openai.chat.v1",
				Compiler: providercatalog.Compiler{
					AdapterID: providercatalog.StaticBackendCompilerID,
					Config:    map[string]any{"path": "/v1/chat/completions"},
				},
			}},
			Credential: providercatalog.Credential{Mode: providercatalog.CredentialOptional, AdapterID: "bearer", Label: "API key"},
			Origin:     providercatalog.Origin{Mode: providercatalog.OriginUserSupplied, Label: "Endpoint"},
		}
	})
	integrations := append(providercatalog.BuiltinIntegrations(), privateTest)
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: integrations,
		BackendCompilers: []providercatalog.BackendCompiler{
			providercatalog.StaticBackendCompiler{},
		},
		WireFormats:          []string{"openai.chat.v1", "openai.responses.v1", "anthropic.messages.v1"},
		CredentialAdapterIDs: []string{"bearer", "x-api-key"},
		DiscoveryAdapterIDs:  []string{"openai.models.v1", "anthropic.models.v1"},
	})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	return providercatalog.AuthoringCompiler{Registry: registry}
}
