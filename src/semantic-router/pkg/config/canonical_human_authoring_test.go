package config

import (
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const humanAuthoringFixture = `
version: v0.3
providers:
  models:
    - name: local/primary
      provider_model_id: upstream-primary
      backend_refs:
        - provider: private-test
          endpoint: http://model.example
      control:
        retry: {count: 2, on: [unavailable]}
        timeout: {request: 30s, stream: 2m}
      pricing:
        input_cost_per_million_tokens: "0.5"
routing:
  modelCards:
    - name: local/primary
      description: Primary local model
      capabilities: [chat, tools]
      modality: text
recipes:
  - name: balance
    routing:
      decisions:
        - name: simple
          rules: {operator: AND, conditions: []}
entrypoints:
  - model_names: [vllm-sr/auto, auto]
    recipe: balance
    assignments:
      simple:
        models:
          - model: local/primary
global:
  billing:
    currency: USD
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
`

func TestHumanV03AuthoringCompilesNamesToStrictSnapshot(t *testing.T) {
	parser := testAuthoringParser(t)
	cfg, err := parser.ParseYAMLBytes([]byte(humanAuthoringFixture))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.RoutingSnapshot == nil || len(cfg.RoutingSnapshot.Models) != 1 {
		t.Fatalf("compiled snapshot = %+v", cfg.RoutingSnapshot)
	}
	model := cfg.RoutingSnapshot.Models[0]
	if len(model.Backends) != 1 {
		t.Fatalf("machine identity was not compiled: %+v", model)
	}
	_, backendIDErr := uuid.Parse(model.Backends[0].ID)
	if !strings.HasPrefix(model.ID, "mdl_") || model.Revision != 1 ||
		!strings.HasPrefix(model.CatalogRevision, "sha256:") ||
		backendIDErr != nil {
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

func TestHumanV03FixedProviderUsesCatalogOrigin(t *testing.T) {
	document := strings.Replace(
		humanAuthoringFixture,
		"        - provider: private-test\n          endpoint: http://model.example\n",
		"        - provider: openai\n          api_key_env: OPENAI_API_KEY\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	backend := cfg.RoutingSnapshot.Models[0].Backends[0]
	if backend.ProviderID != "openai" || backend.Origin != "https://api.openai.com/v1" {
		t.Fatalf("fixed Provider origin = %+v", backend)
	}
}

func TestHumanV03AuthoringFailsClosedWithoutProviderCompiler(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(humanAuthoringFixture))
	if err == nil || !strings.Contains(err.Error(), "require an injected Provider Integration compiler") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestHumanV03AuthoringRejectsMachineFields(t *testing.T) {
	tests := []struct {
		name   string
		needle string
		value  string
	}{
		{"model id", "    - name: local/primary", "    - id: mdl_primary\n      name: local/primary"},
		{"catalog revision", "    - name: local/primary", "    - provider_catalog_revision: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n      name: local/primary"},
		{"compiled backends", "      backend_refs:", "      backends: []\n      backend_refs:"},
		{"recipe id", "  - name: balance", "  - id: rcp_balance\n    name: balance"},
		{"decision id", "        - name: simple", "        - id: dec_simple\n          name: simple"},
		{"entrypoint id", "  - model_names: [vllm-sr/auto, auto]", "  - id: ep_auto\n    model_names: [vllm-sr/auto, auto]"},
		{"model id reference", "          - model: local/primary", "          - model_id: mdl_primary"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			document := strings.Replace(humanAuthoringFixture, test.needle, test.value, 1)
			_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
			if err == nil || !strings.Contains(err.Error(), "unsupported v0.3 authoring fields") {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}
		})
	}
}

func TestHumanV03AuthoringRejectsRootBillingCurrency(t *testing.T) {
	document := strings.Replace(
		humanAuthoringFixture,
		"version: v0.3",
		"version: v0.3\nbilling_currency: USD",
		1,
	)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "unexpected top-level keys: billing_currency") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestHumanV03BackendLiteralCompilesToOpaqueCredentialIdentity(t *testing.T) {
	document := strings.Replace(
		humanAuthoringFixture,
		"          endpoint: http://model.example\n",
		"          endpoint: http://model.example\n          api_key: example-not-a-secret\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	credentialID := cfg.RoutingSnapshot.Models[0].Backends[0].ProviderCredentialID
	if _, err := uuid.Parse(credentialID); err != nil {
		t.Fatalf("provider credential ID = %q, want deterministic UUID", credentialID)
	}
	credential, found := cfg.BackendCredentials.File[credentialID]
	if !found || credential.SecretValue != "example-not-a-secret" || credential.SecretEnv != "" {
		t.Fatalf("compiled credential = %+v", credential)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if got := exported.Providers.Models[0].BackendRefs[0].APIKey; got != "example-not-a-secret" {
		t.Fatalf("public source api_key = %q", got)
	}

	invalid := strings.Replace(
		document,
		"          api_key: example-not-a-secret\n",
		"          api_key: example-not-a-secret\n          api_key_env: PROVIDER_API_KEY\n",
		1,
	)
	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(invalid)); err == nil ||
		!strings.Contains(err.Error(), "only one of api_key or api_key_env") {
		t.Fatalf("conflicting credential source error = %v", err)
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
