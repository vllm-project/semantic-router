package providercatalog

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRegistryIsDeterministicAndDefensive(t *testing.T) {
	first := validDefinition("provider-z", 20)
	second := validDefinition("provider-a", 10)
	one := mustTestRegistry(t, first, second)
	two := mustTestRegistry(t, second, first)
	if one.Snapshot().Revision() != two.Snapshot().Revision() ||
		!reflect.DeepEqual(one.Snapshot().List(), two.Snapshot().List()) {
		t.Fatal("integration source order changed the catalog")
	}
	providers := one.Snapshot().List()
	if len(providers) != 2 || providers[0].ID != "provider-a" || providers[1].ID != "provider-z" {
		t.Fatalf("unexpected UI ordering: %#v", providers)
	}
	if !validCatalogRevision(providers[0].Revision) {
		t.Fatalf("provider revision = %q", providers[0].Revision)
	}
	providers[0].Display.Name = "mutated"
	provider, _ := one.Snapshot().Get("provider-a")
	if provider.Display.Name == "mutated" {
		t.Fatal("Snapshot returned mutable registry storage")
	}
}

func TestRegistryEvaluatesTypedIntegrationsExactlyOnce(t *testing.T) {
	evaluations := 0
	options := testRegistryOptions()
	options.Integrations = []Integration{IntegrationFunc(func() Definition {
		evaluations++
		return validDefinition("provider", 1)
	})}
	registry, err := NewRegistry(options)
	if err != nil {
		t.Fatal(err)
	}
	_ = registry.Snapshot()
	_ = registry.Snapshot()
	if evaluations != 1 {
		t.Fatalf("Integration Definition() evaluations = %d, want 1", evaluations)
	}
}

func TestRegistryRejectsDuplicateProvidersAndUnknownCapabilities(t *testing.T) {
	if _, err := NewRegistry(testRegistryOptions(
		validDefinition("same-provider", 1), validDefinition("same-provider", 2),
	)); err == nil {
		t.Fatal("duplicate provider IDs were accepted")
	}
	unknown := validDefinition("provider", 1)
	unknown.Interfaces[0].WireFormat = "product-specific-runtime"
	if _, err := NewRegistry(testRegistryOptions(unknown)); err == nil {
		t.Fatal("unknown wire format was accepted")
	}
}

func TestRegistryRejectsCallerOwnedRevisionAndNonJSONCompilerConfiguration(t *testing.T) {
	definition := validDefinition("provider", 1)
	definition.Revision = "sha256:caller-owned"
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("caller-owned provider revision was accepted")
	}
	definition = validDefinition("provider", 1)
	cycle := map[string]any{}
	cycle["self"] = cycle
	definition.Interfaces[0].Compiler.Config = cycle
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("cyclic backend compiler config was accepted")
	}
}

func TestRegistryValidatesControlPlaneOwnedIconDescriptors(t *testing.T) {
	for name, icon := range map[string]Icon{
		"unknown source": {Source: "provider", Value: "example"},
		"unsafe asset":   {Source: "asset", Value: "/icons/../secret.svg", Color: true},
		"insecure URL":   {Source: "url", Value: "http://icons.example.com/logo.svg", Color: true},
		"credential URL": {Source: "url", Value: "https://user:secret@icons.example.com/logo.svg", Color: true},
		"query URL":      {Source: "url", Value: "https://icons.example.com/logo.svg?token=secret", Color: true},
	} {
		t.Run(name, func(t *testing.T) {
			definition := validDefinition("provider", 1)
			definition.Display.Icon = icon
			if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
				t.Fatalf("invalid icon was accepted: %+v", icon)
			}
		})
	}
}

func TestRegistryRejectsNilIntegration(t *testing.T) {
	options := testRegistryOptions(validDefinition("provider", 1))
	options.Integrations = []Integration{nil}
	if _, err := NewRegistry(options); err == nil {
		t.Fatal("nil integration was accepted")
	}
	var typedNil IntegrationFunc
	options.Integrations = []Integration{typedNil}
	if _, err := NewRegistry(options); err == nil {
		t.Fatal("typed nil integration was accepted")
	}
}

func TestSnapshotPersistenceRebuildsAndRevalidatesImmutableCatalog(t *testing.T) {
	registry := mustTestRegistry(t, validDefinition("provider", 1))
	snapshot := registry.Snapshot()
	payload, err := snapshot.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := RestoreSnapshot(payload, registry)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Revision() != snapshot.Revision() ||
		!reflect.DeepEqual(restored.List(), snapshot.List()) ||
		!reflect.DeepEqual(restored.IntegrationReferences(), snapshot.IntegrationReferences()) {
		t.Fatalf("restored snapshot differs: %#v %#v", restored, snapshot)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	integrations := envelope["integrations"].([]any)
	integrations[0].(map[string]any)["display"].(map[string]any)["name"] = "Tampered"
	tampered, _ := json.Marshal(envelope)
	if _, err := RestoreSnapshot(tampered, registry); err == nil {
		t.Fatal("snapshot with mismatched content revision was restored")
	}
	unknown := append(append([]byte(nil), payload[:len(payload)-1]...), []byte(`,"unknown":true}`)...)
	if _, err := RestoreSnapshot(unknown, registry); err == nil {
		t.Fatal("snapshot with an unknown field was restored")
	}
}

func TestSecuritySensitiveHeadersAndSecretFieldsFailClosed(t *testing.T) {
	definition := validDefinition("provider", 1)
	definition.Interfaces[0].Compiler.Config["headers"] = map[string]string{"Authorization": "Bearer catalog-secret"}
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("security-sensitive invocation header was accepted")
	}
	definition = validDefinition("provider", 1)
	definition.Discovery.Headers = map[string]string{"X-API-Key": "catalog-secret"}
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("security-sensitive discovery header was accepted")
	}
	definition = validDefinition("provider", 1)
	definition.Interfaces[0].Compiler = Compiler{AdapterID: "test.fields.v1", Config: map[string]any{"path": "/chat/completions"}}
	definition.ConnectionFields = []ConnectionField{{Name: "region", Label: "Region", Kind: "secret"}}
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("secret connection field was accepted")
	}
}

func TestConnectionFieldDefaultsAreTypedAndBounded(t *testing.T) {
	for _, field := range []ConnectionField{
		{Name: "enabled", Label: "Enabled", Kind: FieldBoolean, Default: "yes"},
		{Name: "shard", Label: "Shard", Kind: FieldInteger, Default: "01"},
	} {
		definition := validDefinition("provider", 1)
		definition.Interfaces[0].Compiler = Compiler{AdapterID: "test.fields.v1", Config: map[string]any{"path": "/chat/completions"}}
		definition.ConnectionFields = []ConnectionField{field}
		if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
			t.Fatalf("invalid typed default was accepted: %+v", field)
		}
	}
	definition := validDefinition("provider", 1)
	definition.Interfaces[0].Compiler = Compiler{AdapterID: "test.fields.v1", Config: map[string]any{"path": "/chat/completions"}}
	definition.ConnectionFields = make([]ConnectionField, 65)
	for index := range definition.ConnectionFields {
		definition.ConnectionFields[index] = ConnectionField{
			Name: fmt.Sprintf("field_%d", index), Label: "Field", Kind: FieldText,
		}
	}
	if _, err := NewRegistry(testRegistryOptions(definition)); err == nil {
		t.Fatal("more than 64 connection fields were accepted")
	}
}

func validDefinition(providerID string, order uint32) Definition {
	return Definition{
		ID: providerID, Order: order,
		Display: Display{
			Name: providerID, Description: "A declarative provider.",
			Category: "Model APIs", Icon: lobeIcon("provider", false),
		},
		Interfaces: []Interface{{
			ID: "chat", Label: "Chat Completions", Default: true,
			WireFormat: "openai.chat.v1",
			Compiler: Compiler{AdapterID: StaticBackendCompilerID, Config: map[string]any{
				"path": "/chat/completions", "headers": map[string]any{"X-Provider-Version": "1"},
			}},
		}},
		Credential:   Credential{Mode: CredentialRequired, AdapterID: "bearer", Label: "API key"},
		Origin:       Origin{Mode: OriginFixed, DefaultURL: "https://api.example.com/v1"},
		Discovery:    &Discovery{AdapterID: "openai.models.v1", Path: "/models"},
		Capabilities: []string{"tools", "streaming", "tools"},
	}
}

func testRegistryOptions(definitions ...Definition) RegistryOptions {
	integrations := make([]Integration, len(definitions))
	for index := range definitions {
		definition := definitions[index]
		integrations[index] = IntegrationFunc(func() Definition { return definition })
	}
	return RegistryOptions{
		Integrations: integrations,
		BackendCompilers: []BackendCompiler{
			StaticBackendCompiler{}, testFieldCompiler{}, unsafeTestCompiler{},
		},
		WireFormats:          []string{"anthropic.messages.v1", "openai.chat.v1", "openai.responses.v1"},
		CredentialAdapterIDs: []string{"bearer", "x-api-key"},
		DiscoveryAdapterIDs:  []string{"anthropic.models.v1", "openai.models.v1"},
	}
}

func mustTestRegistry(t testing.TB, definitions ...Definition) *Registry {
	t.Helper()
	registry, err := NewRegistry(testRegistryOptions(definitions...))
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

type testFieldCompiler struct{}

func (testFieldCompiler) AdapterID() string { return "test.fields.v1" }
func (testFieldCompiler) Validate(config map[string]any, fields []ConnectionField) error {
	if len(fields) == 0 {
		return fmt.Errorf("test field compiler requires fields")
	}
	_, err := decodeStaticBackendCompilerConfig(config)
	return err
}

func (testFieldCompiler) Compile(config map[string]any, values map[string]CanonicalConnectionValue) (routingsnapshot.BackendConnection, error) {
	decoded, err := decodeStaticBackendCompilerConfig(config)
	if err != nil {
		return routingsnapshot.BackendConnection{}, err
	}
	region, found := values["region"]
	if !found {
		return routingsnapshot.BackendConnection{}, fmt.Errorf("region is required")
	}
	headers := cloneStringMap(decoded.Headers)
	if headers == nil {
		headers = make(map[string]string)
	}
	headers["X-Region"] = region.Value
	return routingsnapshot.BackendConnection{Path: decoded.Path, Headers: headers}, nil
}

type unsafeTestCompiler struct{}

func (unsafeTestCompiler) AdapterID() string                                { return "test.unsafe.v1" }
func (unsafeTestCompiler) Validate(map[string]any, []ConnectionField) error { return nil }
func (unsafeTestCompiler) Compile(map[string]any, map[string]CanonicalConnectionValue) (routingsnapshot.BackendConnection, error) {
	return routingsnapshot.BackendConnection{
		Path: "/chat/completions", Headers: map[string]string{"Authorization": "Bearer forbidden"},
	}, nil
}
