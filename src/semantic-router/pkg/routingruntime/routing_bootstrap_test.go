package routingruntime

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	bootstrapTestNamespaceA = "11111111-1111-4111-8111-111111111111"
	bootstrapTestNamespaceB = "22222222-2222-4222-8222-222222222222"
	bootstrapCanonicalYAML  = `
version: v0.3
providers:
  models:
    - name: remote/anthropic
      provider_model_id: claude-compatible
      backend_refs:
        - provider: anthropic-compatible
          endpoint: https://models.example
          credential: reference-anthropic
      api_format: anthropic
routing:
  modelCards:
    - name: remote/anthropic
      capabilities: [chat, image_input, tools]
      modality: omni
recipes:
  - name: balance
    routing:
      decisions:
        - name: omni
          rules: {operator: AND, conditions: []}
entrypoints:
  - model_names: [vllm-sr/balance]
    recipe: balance
    assignments:
      omni:
        models:
          - model: remote/anthropic
global:
  services:
    backend_credentials:
      reference-anthropic:
        credential_adapter_id: x-api-key
        secret_env: ROUTING_BOOTSTRAP_TEST_SECRET
    backend_egress:
      policy_file: /etc/vllm-sr/backend-egress-policy.yaml
`
)

func TestCompileDurableBootstrapSnapshotScopesReadableCredentialAliases(t *testing.T) {
	catalog := bootstrapAnthropicCatalog(t)
	source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
		bootstrapAliasBackend("backend-short", "x", "https://models.example"),
	})

	first, firstNames, err := compileDurableBootstrapSnapshot(source)
	if err != nil {
		t.Fatal(err)
	}
	second, secondNames, err := compileDurableBootstrapSnapshot(source)
	if err != nil {
		t.Fatal(err)
	}
	firstID := first.Models[0].Backends[0].ProviderCredentialID
	if _, err := uuid.Parse(firstID); err != nil {
		t.Fatalf("durable credential ID = %q: %v", firstID, err)
	}
	if firstID == "x" || firstNames[firstID] != "x" || secondNames[firstID] != "x" ||
		second.Models[0].Backends[0].ProviderCredentialID != firstID || first.Digest != second.Digest {
		t.Fatalf("stable alias mapping = first %q/%q, second %q/%q",
			firstID, firstNames[firstID], second.Models[0].Backends[0].ProviderCredentialID, secondNames[firstID])
	}
	if source.Models[0].Backends[0].ProviderCredentialID != "x" {
		t.Fatalf("file-authored snapshot was mutated: %q", source.Models[0].Backends[0].ProviderCredentialID)
	}
	shortConfig := &config.RouterConfig{BackendCredentials: config.BackendCredentialsConfig{File: map[string]config.BackendCredentialConfig{
		"x": {CredentialAdapterID: "x-api-key", SecretValue: "provider-secret"},
	}}}
	shortSeeds, err := materializeBootstrapCredentials(
		shortConfig, first, firstNames, bootstrapProviderCodec(), catalog,
	)
	if err != nil || len(shortSeeds) != 1 || shortSeeds[0].Credential.Name != "x" {
		t.Fatalf("short alias materialization = %#v, %v", shortSeeds, err)
	}

	otherSource := bootstrapAliasSnapshot(t, bootstrapTestNamespaceB, catalog.Revision(), []routingsnapshot.Backend{
		bootstrapAliasBackend("backend-short", "x", "https://models.example"),
	})
	other, _, err := compileDurableBootstrapSnapshot(otherSource)
	if err != nil {
		t.Fatal(err)
	}
	if other.Models[0].Backends[0].ProviderCredentialID == firstID {
		t.Fatal("durable provider credential identity was not isolated by Namespace")
	}
}

func TestMaterializeBootstrapCredentialsPreservesCanonicalAliasBinding(t *testing.T) {
	catalog := bootstrapAnthropicCatalog(t)
	alias := "reference-anthropic"
	source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
		bootstrapAliasBackend("backend-primary", alias, "https://models.example"),
		bootstrapAliasBackend("backend-replica", alias, "https://models.example"),
	})
	durable, names, err := compileDurableBootstrapSnapshot(source)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{BackendCredentials: config.BackendCredentialsConfig{File: map[string]config.BackendCredentialConfig{
		alias: {CredentialAdapterID: "x-api-key", SecretValue: "provider-secret"},
	}}}
	seeds, err := materializeBootstrapCredentials(cfg, durable, names, bootstrapProviderCodec(), catalog)
	if err != nil {
		t.Fatal(err)
	}
	if len(seeds) != 1 {
		t.Fatalf("materialized provider credentials = %d, want 1", len(seeds))
	}
	seed := seeds[0]
	credentialID := durable.Models[0].Backends[0].ProviderCredentialID
	if seed.Credential.ID != credentialID || seed.Credential.Name != alias ||
		seed.Version.CredentialID != credentialID || seed.Credential.Validate() != nil || seed.Version.Validate() != nil {
		t.Fatalf("durable provider credential = %#v, version = %#v", seed.Credential, seed.Version)
	}
	for _, backend := range durable.Models[0].Backends {
		if backend.ProviderCredentialID != credentialID {
			t.Fatalf("backend %q credential = %q, want %q", backend.ID, backend.ProviderCredentialID, credentialID)
		}
	}
	if _, exposedGeneratedID := cfg.BackendCredentials.File[credentialID]; exposedGeneratedID {
		t.Fatal("file-authored credential map exposed the generated durable UUID")
	}
}

func TestCanonicalAuthoringMaterializesReadableAliasIntoDurableIdentity(t *testing.T) {
	registry := bootstrapAnthropicRegistry(t)
	cfg, err := config.NewParser(providercatalog.AuthoringCompiler{Registry: registry}).ParseYAMLBytes(
		[]byte(bootstrapCanonicalYAML),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.RoutingSnapshot.Models[0].Backends[0].ProviderCredentialID; got != "reference-anthropic" {
		t.Fatalf("file-authored provider credential reference = %q", got)
	}

	t.Setenv("ROUTING_BOOTSTRAP_TEST_SECRET", "provider-secret")
	durable, names, err := compileDurableBootstrapSnapshot(cfg.RoutingSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	seeds, err := materializeBootstrapCredentials(
		cfg, durable, names, bootstrapProviderCodec(), registry.Snapshot(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(seeds) != 1 || seeds[0].Credential.Name != "reference-anthropic" ||
		seeds[0].Credential.ID != durable.Models[0].Backends[0].ProviderCredentialID {
		t.Fatalf("durable provider credential seed = %#v", seeds)
	}
	if _, err := uuid.Parse(seeds[0].Credential.ID); err != nil {
		t.Fatalf("durable provider credential ID = %q: %v", seeds[0].Credential.ID, err)
	}
}

func TestMaterializeBootstrapCredentialsRejectsInvalidAliasBindings(t *testing.T) {
	catalog := bootstrapAnthropicCatalog(t)

	t.Run("missing secret", func(t *testing.T) {
		source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
			bootstrapAliasBackend("backend-missing", "missing", "https://models.example"),
		})
		durable, names, err := compileDurableBootstrapSnapshot(source)
		if err != nil {
			t.Fatal(err)
		}
		_, err = materializeBootstrapCredentials(
			&config.RouterConfig{}, durable, names, bootstrapProviderCodec(), catalog,
		)
		if err == nil || !strings.Contains(err.Error(), `alias "missing" has no file secret source`) {
			t.Fatalf("missing secret error = %v", err)
		}
	})

	t.Run("incompatible reuse", func(t *testing.T) {
		source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
			bootstrapAliasBackend("backend-first", "shared", "https://first.example"),
			bootstrapAliasBackend("backend-second", "shared", "https://second.example"),
		})
		durable, names, err := compileDurableBootstrapSnapshot(source)
		if err != nil {
			t.Fatal(err)
		}
		cfg := &config.RouterConfig{BackendCredentials: config.BackendCredentialsConfig{File: map[string]config.BackendCredentialConfig{
			"shared": {CredentialAdapterID: "x-api-key", SecretValue: "provider-secret"},
		}}}
		_, err = materializeBootstrapCredentials(cfg, durable, names, bootstrapProviderCodec(), catalog)
		for _, fragment := range []string{"alias \"shared\"", "incompatible backends", "backend-first", "backend-second"} {
			if err == nil || !strings.Contains(err.Error(), fragment) {
				t.Fatalf("incompatible reuse error = %v, want %q", err, fragment)
			}
		}
	})
}

func TestSealBootstrapCredentialZeroesSecretOnEveryExit(t *testing.T) {
	catalog := bootstrapAnthropicCatalog(t)
	source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
		bootstrapAliasBackend("backend-zero", "zero-me", "https://models.example"),
	})
	durable, names, err := compileDurableBootstrapSnapshot(source)
	if err != nil {
		t.Fatal(err)
	}
	credentialID := durable.Models[0].Backends[0].ProviderCredentialID
	binding := bootstrapCredentialBinding{
		providerID: "anthropic-compatible", origin: "https://models.example",
		catalogRevision: durable.Models[0].CatalogRevision,
		credentialMode:  providercredential.ModeOptional, credentialAdapterID: "x-api-key",
		backendID: "backend-zero",
	}
	for _, test := range []struct {
		name       string
		definition config.BackendCredentialConfig
		codec      providercredential.Codec
	}{
		{name: "binding error", definition: config.BackendCredentialConfig{CredentialAdapterID: "bearer"}, codec: bootstrapProviderCodec()},
		{name: "seal error", definition: config.BackendCredentialConfig{CredentialAdapterID: "x-api-key"}, codec: providercredential.Codec{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			secret := []byte("provider-secret")
			_, err := sealBootstrapCredential(
				names[credentialID], credentialID, test.definition, binding, durable, test.codec,
				time.Date(2026, 8, 25, 0, 0, 0, 0, time.UTC), secret,
			)
			if err == nil {
				t.Fatal("sealBootstrapCredential() accepted invalid input")
			}
			if !bytes.Equal(secret, make([]byte, len(secret))) {
				t.Fatalf("secret was not zeroed after error: %x", secret)
			}
		})
	}
}

func TestGeneratedFileCredentialIdentityGetsReadableDurableName(t *testing.T) {
	catalog := bootstrapAnthropicCatalog(t)
	generatedSourceID := "33333333-3333-4333-8333-333333333333"
	source := bootstrapAliasSnapshot(t, bootstrapTestNamespaceA, catalog.Revision(), []routingsnapshot.Backend{
		bootstrapAliasBackend("backend-generated", generatedSourceID, "https://models.example"),
	})
	durable, names, err := compileDurableBootstrapSnapshot(source)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{BackendCredentials: config.BackendCredentialsConfig{File: map[string]config.BackendCredentialConfig{
		generatedSourceID: {CredentialAdapterID: "x-api-key", SecretValue: "provider-secret"},
	}}}
	seeds, err := materializeBootstrapCredentials(cfg, durable, names, bootstrapProviderCodec(), catalog)
	if err != nil || len(seeds) != 1 {
		t.Fatalf("materialize generated file credential = %#v, %v", seeds, err)
	}
	if seeds[0].Credential.Name == generatedSourceID {
		t.Fatalf("generated internal identity leaked into ProviderCredential name: %q", seeds[0].Credential.Name)
	}
	if err := providercredential.ValidateName(seeds[0].Credential.Name); err != nil {
		t.Fatalf("readable ProviderCredential name = %q: %v", seeds[0].Credential.Name, err)
	}
}

func bootstrapAliasSnapshot(
	t *testing.T,
	namespaceID string,
	catalogRevision string,
	backends []routingsnapshot.Backend,
) *routingsnapshot.Snapshot {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: namespaceID, Revision: 1,
		Models: []routingsnapshot.Model{{
			ID: "model-anthropic", Revision: 1, CatalogRevision: catalogRevision,
			Name: "anthropic/model", Backends: backends,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func bootstrapAliasBackend(id, credentialAlias, origin string) routingsnapshot.Backend {
	return routingsnapshot.Backend{
		ID: id, ProviderID: "anthropic-compatible", WireFormat: llmprotocol.AnthropicMessagesV1,
		Origin: origin, ProviderModelID: "claude-compatible", ProviderCredentialID: credentialAlias,
		Connection: routingsnapshot.BackendConnection{Path: "/v1/messages"}, Weight: "1",
	}
}

func bootstrapAnthropicCatalog(t *testing.T) *providercatalog.Snapshot {
	t.Helper()
	return bootstrapAnthropicRegistry(t).Snapshot()
}

func bootstrapAnthropicRegistry(t *testing.T) *providercatalog.Registry {
	t.Helper()
	var anthropicCompatible providercatalog.Integration
	for _, integration := range providercatalog.BuiltinIntegrations() {
		if integration.Definition().ID == "anthropic-compatible" {
			anthropicCompatible = integration
			break
		}
	}
	if anthropicCompatible == nil {
		t.Fatal("anthropic-compatible Provider Integration is unavailable")
	}
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{anthropicCompatible},
		BackendCompilers: []providercatalog.BackendCompiler{
			providercatalog.StaticBackendCompiler{},
		},
		WireFormats:          []string{string(llmprotocol.AnthropicMessagesV1)},
		CredentialAdapterIDs: []string{"x-api-key"},
		DiscoveryAdapterIDs:  []string{"anthropic.models.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func bootstrapProviderCodec() providercredential.Codec {
	return providercredential.Codec{Keyring: accesscredential.KEKKeyring{
		ActiveVersion: "bootstrap-provider-kek-v1",
		Keys: map[string][]byte{
			"bootstrap-provider-kek-v1": bytes.Repeat([]byte{0x5a}, 32),
		},
	}}
}
