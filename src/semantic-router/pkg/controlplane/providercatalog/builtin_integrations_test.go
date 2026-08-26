package providercatalog

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
)

func TestBuiltinIntegrationsAreCompleteValidCatalogData(t *testing.T) {
	options := testRegistryOptions()
	options.Integrations = BuiltinIntegrations()
	registry, err := NewRegistry(options)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := registry.Snapshot()
	if got := len(snapshot.List()); got != 40 {
		t.Fatalf("built-in provider count = %d, want 40", got)
	}
	for _, providerID := range []string{"vllm", "sglang", "amd-atom", "openai-compatible", "anthropic"} {
		if _, found := snapshot.Get(providerID); !found {
			t.Fatalf("built-in provider %q is missing", providerID)
		}
	}
	atom, found := snapshot.Get("amd-atom")
	if !found || atom.Origin.Mode != OriginUserSupplied || atom.Origin.DefaultURL != "" {
		t.Fatalf("AMD ATOM must use the operator's local endpoint, got %+v", atom.Origin)
	}
	for _, provider := range snapshot.List() {
		if provider.Origin.Mode == OriginFixed && provider.Origin.DefaultURL == "" {
			t.Fatalf("fixed provider %q has no default URL", provider.ID)
		}
	}
	for _, providerID := range []string{"vllm", "anthropic"} {
		provider, found := snapshot.Get(providerID)
		if !found || !hasBuiltinCapability(provider.Capabilities, "text") {
			t.Fatalf("built-in provider %q does not advertise text", providerID)
		}
		for _, providerInterface := range provider.Interfaces {
			if !hasBuiltinCapability(providerInterface.Capabilities, "text") {
				t.Fatalf(
					"built-in provider %q interface %q does not advertise text",
					providerID, providerInterface.ID,
				)
			}
		}
	}
}

func TestBuiltinVLLMCompilesCompleteAPIPathsFromConventionalBaseURLs(t *testing.T) {
	options := testRegistryOptions()
	options.Integrations = BuiltinIntegrations()
	registry, err := NewRegistry(options)
	if err != nil {
		t.Fatal(err)
	}
	compiler := AuthoringCompiler{Registry: registry}
	for _, test := range []struct {
		name, endpoint, interfaceID, targetPath string
	}{
		{name: "chat from server origin", endpoint: "https://vllm.example.com", targetPath: "/v1/chat/completions"},
		{name: "chat from API base URL", endpoint: "https://vllm.example.com/v1", targetPath: "/v1/chat/completions"},
		{name: "responses from server origin", endpoint: "https://vllm.example.org", interfaceID: "responses", targetPath: "/v1/responses"},
		{name: "responses from API base URL", endpoint: "https://vllm.example.org/v1", interfaceID: "responses", targetPath: "/v1/responses"},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, compileErr := compiler.CompileConnection(context.Background(), modelauthoring.CompileRequest{
				BackendID: "backend-vllm",
				Connection: modelauthoring.Connection{
					Provider: "vllm", Interface: test.interfaceID,
					Endpoint: test.endpoint, Model: "public-example-model",
				},
			})
			if compileErr != nil {
				t.Fatal(compileErr)
			}
			if got := effectiveProviderPath(t, result.Backend.Origin, result.Backend.Connection.Path); got != test.targetPath {
				t.Fatalf("compiled target path = %q, want %q (origin %q, path %q)", got, test.targetPath, result.Backend.Origin, result.Backend.Connection.Path)
			}
		})
	}
	managementCompiler := ModelCompiler{
		Catalog: staticSnapshotSource(registry.Snapshot()), Registry: registry,
		Egress: compilerEgressPolicy(t, "management-vllm.example.net"),
	}
	managed, err := managementCompiler.CompileBackend(context.Background(), ModelBackendRequest{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		BackendID:   "22222222-2222-4222-8222-222222222222",
		ProviderID:  "vllm", ProviderModelID: "public-example-model",
		Origin: "https://management-vllm.example.net",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := effectiveProviderPath(t, managed.Backend.Origin, managed.Backend.Connection.Path); got != "/v1/chat/completions" {
		t.Fatalf("managed Model target path = %q, want /v1/chat/completions", got)
	}

	validator := &recordingDiscoveryValidator{id: "openai.models.v1"}
	discoveryRegistry, err := NewDiscoveryRegistry([]DiscoveryRequestValidator{validator})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(staticSnapshotSource(registry.Snapshot()), ServiceOptions{
		CursorKeyring: testCursorKeyring, DiscoveryPlugins: discoveryRegistry,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, endpoint := range []string{"https://inventory.example.com", "https://inventory.example.com/v1"} {
		plan, err := service.PrepareDiscovery(context.Background(), "vllm", DiscoverModelsRequest{
			NamespaceID: "11111111-1111-4111-8111-111111111111", Origin: endpoint,
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := effectiveProviderPath(t, plan.NormalizedOrigin, plan.Path); got != "/v1/models" {
			t.Fatalf("discovery target path = %q, want /v1/models (origin %q, path %q)", got, plan.NormalizedOrigin, plan.Path)
		}
	}
}

func effectiveProviderPath(t *testing.T, origin, path string) string {
	t.Helper()
	parsed, err := url.Parse(origin)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimRight(parsed.Path, "/") + path
}

func hasBuiltinCapability(capabilities []string, expected string) bool {
	for _, capability := range capabilities {
		if capability == expected {
			return true
		}
	}
	return false
}
