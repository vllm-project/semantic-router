package managementcomposition

import (
	"net/url"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
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

func TestRoutingManifestCodecRoundTripsMultiEntrypointVLLMManifest(t *testing.T) {
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations:     providercatalog.BuiltinIntegrations(),
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats: []string{
			string(llmprotocol.OpenAIChatV1), string(llmprotocol.OpenAIResponsesV1),
			string(llmprotocol.AnthropicMessagesV1),
		},
		CredentialAdapterIDs: []string{"bearer", "x-api-key"},
		DiscoveryAdapterIDs:  []string{"openai.models.v1", "anthropic.models.v1"},
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
  defaults:
    reasoning_families:
      strong-reasoning:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: public/fast
      provider_model_id: public-fast-model
      backend_refs:
        - provider: vllm
          base_url: https://fast-vllm.example.com
          weight: 2
    - name: public/reasoning
      reasoning_family: strong-reasoning
      provider_model_id: public-reasoning-model
      backend_refs:
        - provider: vllm
          base_url: https://reasoning-vllm.example.org/v1
          weight: 3
routing:
  modelCards:
    - name: public/fast
      capabilities: [chat]
    - name: public/reasoning
      capabilities: [chat, reasoning]
      reasoning:
        type: reasoning_effort
        efforts: [high]
recipes:
  - name: choose-model
    routing:
      decisions:
        - name: default
          rules: {operator: AND, conditions: []}
entrypoints:
  - model_names: [vllm-sr/public]
    recipe: choose-model
    assignments:
      default:
        models:
          - model: public/fast
            weight: "1"
          - model: public/reasoning
            weight: "1"
            reasoning: {enabled: true, effort: high}
  - model_names: [vllm-sr/public-fast]
    recipe: choose-model
    assignments:
      default:
        models:
          - model: public/fast
            weight: "1"
`)
	snapshot, err := codec.Decode(document)
	if err != nil {
		t.Fatal(err)
	}
	assertVLLMRoundTripShape(t, snapshot)

	exported, err := codec.Encode(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{"reasoning_families:", "reasoning_family: public/reasoning"} {
		if !strings.Contains(string(exported), required) {
			t.Fatalf("exported manifest lost %q:\n%s", required, exported)
		}
	}
	roundTrip, err := codec.Decode(exported)
	if err != nil {
		t.Fatalf("decode exported manifest: %v\n%s", err, exported)
	}
	assertVLLMRoundTripShape(t, roundTrip)
}

func assertVLLMRoundTripShape(t *testing.T, snapshot *routingsnapshot.Snapshot) {
	t.Helper()
	if snapshot == nil || len(snapshot.Models) != 2 || len(snapshot.Entrypoints) != 2 {
		t.Fatalf("round-trip snapshot shape = %+v", snapshot)
	}
	models := make(map[string]routingsnapshot.Model, len(snapshot.Models))
	for _, model := range snapshot.Models {
		models[model.Name] = model
		if len(model.Backends) != 1 {
			t.Fatalf("model %q backends = %+v", model.Name, model.Backends)
		}
		backend := model.Backends[0]
		parsed, err := url.Parse(backend.Origin)
		if err != nil {
			t.Fatal(err)
		}
		if got := strings.TrimRight(parsed.Path, "/") + backend.Connection.Path; got != "/v1/chat/completions" {
			t.Fatalf("model %q target path = %q (origin %q, path %q)", model.Name, got, backend.Origin, backend.Connection.Path)
		}
	}
	reasoning, found := models["public/reasoning"]
	if !found || reasoning.Reasoning.Type != "reasoning_effort" ||
		len(reasoning.Reasoning.Efforts) != 1 || reasoning.Reasoning.Efforts[0] != "high" {
		t.Fatalf("reasoning model = %+v, found = %t", reasoning, found)
	}
	entrypoints := make(map[string]struct{}, len(snapshot.Entrypoints))
	for _, entrypoint := range snapshot.Entrypoints {
		entrypoints[entrypoint.Name] = struct{}{}
	}
	for _, name := range []string{"vllm-sr/public", "vllm-sr/public-fast"} {
		if _, found := entrypoints[name]; !found {
			t.Fatalf("round-trip entrypoints = %+v", snapshot.Entrypoints)
		}
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
