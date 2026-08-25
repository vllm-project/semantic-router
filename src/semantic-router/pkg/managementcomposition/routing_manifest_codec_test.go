package managementcomposition

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRoutingManifestCodecRoundTripsReadableCredentialName(t *testing.T) {
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return providercatalog.Definition{
				ID: "private-test",
				Display: providercatalog.Display{
					Name: "Private test", Description: "Private endpoint", Category: "Test",
					Icon: providercatalog.Icon{Source: "lobe", Value: "openai"},
				},
				Interfaces: []providercatalog.Interface{{
					ID: "chat", Label: "Chat", Default: true, WireFormat: llmprotocol.OpenAIChatV1,
					Compiler: providercatalog.Compiler{
						AdapterID: providercatalog.StaticBackendCompilerID,
						Config:    map[string]any{"path": "/v1/chat/completions"},
					},
				}},
				Credential: providercatalog.Credential{
					Mode: providercatalog.CredentialOptional, AdapterID: "bearer", Label: "API key",
				},
				Origin: providercatalog.Origin{Mode: providercatalog.OriginUserSupplied, Label: "Endpoint"},
			}
		})},
		BackendCompilers:     []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:          []string{string(llmprotocol.OpenAIChatV1)},
		CredentialAdapterIDs: []string{"bearer"},
	})
	if err != nil {
		t.Fatal(err)
	}
	codec, err := newV03RoutingManifestCodec(registry)
	if err != nil {
		t.Fatal(err)
	}
	document := []byte(`version: v0.3
providers:
  models:
    - name: remote/model
      provider_model_id: upstream-model
      backend_refs:
        - provider: private-test
          type: chat
          endpoint: https://models.example
          credential: Primary provider
routing:
  modelCards:
    - name: remote/model
recipes:
  - name: direct
    routing:
      decisions:
        - name: default
          rules: {operator: AND, conditions: []}
entrypoints:
  - model_names: [vllm-sr/test]
    recipe: direct
    assignments:
      default:
        models:
          - model: remote/model
`)
	snapshot, err := codec.Decode(document)
	if err != nil {
		t.Fatal(err)
	}
	if got := snapshot.Models[0].Backends[0].ProviderCredentialID; got != "Primary provider" {
		t.Fatalf("decoded credential reference = %q", got)
	}
	exported, err := codec.Encode(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), "credential: Primary provider") {
		t.Fatalf("exported manifest lost readable credential name:\n%s", exported)
	}
	roundTrip, err := codec.Decode(exported)
	if err != nil {
		t.Fatal(err)
	}
	if got := roundTrip.Models[0].Backends[0].ProviderCredentialID; got != "Primary provider" {
		t.Fatalf("round-trip credential reference = %q", got)
	}
	if _, err := codec.Decode([]byte("version: v0.3\n")); err == nil {
		t.Fatal("manifest codec accepted an incomplete routing closure")
	}
}

func TestRoutingManifestCredentialNamesAreStrictAndSecretFree(t *testing.T) {
	document := []byte(`version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - credential: Primary provider
        - credential: Backup provider
    - name: model-b
      backend_refs:
        - credential: Primary provider
`)
	source, err := decodeRoutingManifestSource(document)
	if err != nil {
		t.Fatal(err)
	}
	if got := source.Providers.Models[0].BackendRefs[0].Credential; got != "Primary provider" {
		t.Fatalf("credential name = %q", got)
	}
	for name, invalid := range map[string]string{
		"version":   "version: v0.4\n",
		"duplicate": "version: v0.3\nproviders: {}\nproviders: {}\n",
		"secret": `version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - api_key: secret
`,
		"unknown field": "version: v0.3\nidentity: {}\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeRoutingManifestSource([]byte(invalid)); err == nil {
				t.Fatal("manifest decoder accepted a non-routing or secret-bearing manifest")
			}
		})
	}
}
