package managedruntime

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

func TestProviderRegistryRequiresExplicitApplicationIntegrations(t *testing.T) {
	protocols := protocolcodec.NewBuiltinRegistry()
	credentials, err := backendresolver.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	discovery, err := providerdiscovery.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := composeProviderRegistry(
		nil, []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		protocols, credentials, discovery,
	); err == nil || !strings.Contains(err.Error(), "at least one provider integration") {
		t.Fatalf("missing application integrations error = %v", err)
	}
}

func TestProviderRegistryAcceptsApplicationProviderWithoutRuntimeProductBranch(t *testing.T) {
	protocols := protocolcodec.NewBuiltinRegistry()
	credentials, err := backendresolver.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	discovery, err := providerdiscovery.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	integration := providercatalog.IntegrationFunc(func() providercatalog.Definition {
		return providercatalog.Definition{
			ID: "application-provider", Order: 1,
			Display: providercatalog.Display{
				Name: "Application Provider", Description: "Connect an application Provider.",
				Category: "Model APIs",
				Icon:     providercatalog.Icon{Source: "lobe", Value: "application-provider", Color: false},
			},
			Interfaces: []providercatalog.Interface{{
				ID: "chat", Label: "Chat Completions", Default: true,
				WireFormat: llmprotocol.OpenAIChatV1,
				Compiler: providercatalog.Compiler{
					AdapterID: providercatalog.StaticBackendCompilerID,
					Config:    map[string]any{"path": "/chat/completions"},
				},
			}},
			Credential: providercatalog.Credential{
				Mode: providercatalog.CredentialOptional, AdapterID: "bearer", Label: "API key",
			},
			Origin: providercatalog.Origin{
				Mode: providercatalog.OriginUserSupplied, Label: "API base URL",
			},
			Discovery: &providercatalog.Discovery{AdapterID: "openai.models.v1", Path: "/models"},
		}
	})
	registry, err := composeProviderRegistry(
		[]providercatalog.Integration{integration},
		[]providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		protocols, credentials, discovery,
	)
	if err != nil {
		t.Fatal(err)
	}
	provider, found := registry.Snapshot().Get("application-provider")
	if !found || len(provider.Interfaces) != 1 || provider.Interfaces[0].WireFormat != llmprotocol.OpenAIChatV1 ||
		provider.Revision == "" {
		t.Fatalf("application Provider = %+v, found=%t", provider, found)
	}
}
