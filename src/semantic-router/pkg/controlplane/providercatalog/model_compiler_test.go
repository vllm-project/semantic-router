package providercatalog

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const (
	compilerNamespaceID  = "11111111-1111-4111-8111-111111111111"
	compilerBackendID    = "22222222-2222-4222-8222-222222222222"
	compilerCredentialID = "33333333-3333-4333-8333-333333333333"
	compilerVersionID    = "44444444-4444-4444-8444-444444444444"
	oldCatalogRevision   = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

type compilerCredentialReader struct {
	credential providercredential.Credential
	err        error
	calls      int
}

func (reader *compilerCredentialReader) GetProviderCredential(
	_ context.Context,
	_ accesscontrol.NamespaceID,
	_ string,
) (providercredential.Credential, error) {
	reader.calls++
	return reader.credential, reader.err
}

func TestModelCompilerCompilesProviderDataIntoStableRuntimeContract(t *testing.T) {
	snapshot, registry := compilerSnapshot(t)
	reader := &compilerCredentialReader{credential: compilerCredential()}
	compiler := ModelCompiler{
		Catalog: staticSnapshotSource(snapshot), Registry: registry, Credentials: reader,
		Egress: compilerEgressPolicy(t, "api.example.com"),
	}
	result, err := compiler.CompileBackend(context.Background(), ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "provider", ProviderModelID: "provider-model-1",
		CredentialID:     compilerCredentialID,
		ConnectionFields: map[string]any{"region": "global"}, Weight: "2.5000",
	})
	if err != nil {
		t.Fatal(err)
	}
	backend := result.Backend
	if result.CatalogRevision != snapshot.Revision() || reader.calls != 1 ||
		backend.ProviderID != "provider" || backend.WireFormat != "openai.chat.v1" ||
		backend.Origin != "https://api.example.com/v1" || backend.ProviderModelID != "provider-model-1" ||
		backend.ProviderCredentialID != compilerCredentialID || backend.Connection.Path != "/chat/completions" ||
		backend.Connection.Headers["X-Provider-Version"] != "1" ||
		backend.Connection.Headers["X-Region"] != "global" || backend.Weight != "2.5" {
		t.Fatalf("compiled backend = %+v, revision = %q, reader calls = %d", backend, result.CatalogRevision, reader.calls)
	}
	// Provider display changes remain control-plane data and cannot mutate a
	// backend that was already compiled by value.
	provider, _ := snapshot.Get("provider")
	provider.Display.Name = "Changed"
	if backend.WireFormat != "openai.chat.v1" || backend.Connection.Path != "/chat/completions" {
		t.Fatalf("compiled backend changed with display metadata: %+v", backend)
	}
}

func TestModelCompilerFailsClosedUntilCatalogIsActive(t *testing.T) {
	unavailable := errors.New("active catalog is unavailable")
	credentials := &compilerCredentialReader{credential: compilerCredential()}
	compiler := ModelCompiler{
		Catalog:  SnapshotSourceFunc(func(context.Context) (*Snapshot, error) { return nil, unavailable }),
		Registry: compilerRegistry(t), Credentials: credentials,
		Egress: compilerEgressPolicy(t, "api.example.com"),
	}
	_, err := compiler.CompileBackend(context.Background(), ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "provider", ProviderModelID: "provider-model-1", CredentialID: compilerCredentialID,
	})
	if !errors.Is(err, unavailable) || credentials.calls != 0 {
		t.Fatalf("compile before catalog activation = %v, credential calls = %d", err, credentials.calls)
	}
}

func TestModelCompilerAcceptsCompatibleCredentialFromOlderCatalog(t *testing.T) {
	snapshot, registry := compilerSnapshot(t)
	credential := compilerCredential()
	credential.CatalogRevision = oldCatalogRevision
	compiler := ModelCompiler{
		Catalog: staticSnapshotSource(snapshot), Registry: registry, Credentials: &compilerCredentialReader{credential: credential},
		Egress: compilerEgressPolicy(t, "api.example.com"),
	}
	_, err := compiler.CompileBackend(context.Background(), ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "provider", ProviderModelID: "model", CredentialID: compilerCredentialID,
		ConnectionFields: map[string]any{"region": "global"},
	})
	if err != nil {
		t.Fatalf("compatible credential was rejected after an unrelated catalog revision: %v", err)
	}
}

func TestModelCompilerRejectsMissingOrMismatchedCredential(t *testing.T) {
	snapshot, registry := compilerSnapshot(t)
	base := ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "provider", ProviderModelID: "model",
		ConnectionFields: map[string]any{"region": "global"},
	}
	compiler := ModelCompiler{Catalog: staticSnapshotSource(snapshot), Registry: registry, Egress: compilerEgressPolicy(t, "api.example.com")}
	if _, err := compiler.CompileBackend(context.Background(), base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("missing required credential error = %v", err)
	}

	for name, mutate := range map[string]func(*providercredential.Credential){
		"provider": func(value *providercredential.Credential) { value.ProviderID = "another" },
		"mode":     func(value *providercredential.Credential) { value.CredentialMode = providercredential.ModeOptional },
		"adapter":  func(value *providercredential.Credential) { value.CredentialAdapterID = "x-api-key" },
		"origin":   func(value *providercredential.Credential) { value.NormalizedOrigin = "https://other.example.com" },
	} {
		t.Run(name, func(t *testing.T) {
			credential := compilerCredential()
			mutate(&credential)
			compiler.Credentials = &compilerCredentialReader{credential: credential}
			request := base
			request.CredentialID = compilerCredentialID
			if _, err := compiler.CompileBackend(context.Background(), request); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("mismatched credential error = %v", err)
			}
		})
	}
}

func TestModelCompilerCanonicalizesPrivateOriginAndEnforcesEgress(t *testing.T) {
	provider := validDefinition("private", 1)
	provider.Origin = Origin{Mode: OriginUserSupplied, Label: "Base URL"}
	provider.Credential = Credential{Mode: CredentialNone}
	registry := mustTestRegistry(t, provider)
	snapshot := registry.Snapshot()
	request := ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "private", ProviderModelID: "model",
		Origin: "HTTPS://Private.Example.com:443/v1/",
	}
	allowed := ModelCompiler{Catalog: staticSnapshotSource(snapshot), Registry: registry, Egress: compilerEgressPolicy(t, "private.example.com")}
	compiled, err := allowed.CompileBackend(context.Background(), request)
	if err != nil || compiled.Backend.Origin != "https://private.example.com/v1" {
		t.Fatalf("canonical private origin = (%+v, %v)", compiled, err)
	}
	denied := ModelCompiler{Catalog: staticSnapshotSource(snapshot), Registry: registry, Egress: compilerEgressPolicy(t, "another.example.com")}
	if _, err := denied.CompileBackend(context.Background(), request); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("denied egress error = %v", err)
	}
}

func TestModelCompilerRejectsUnsafeCompilerOutput(t *testing.T) {
	provider := validDefinition("provider", 1)
	provider.Interfaces[0].Compiler = Compiler{AdapterID: "test.unsafe.v1", Config: map[string]any{}}
	provider.Credential = Credential{Mode: CredentialNone}
	registry := mustTestRegistry(t, provider)
	snapshot := registry.Snapshot()
	compiler := ModelCompiler{
		Catalog: staticSnapshotSource(snapshot), Registry: registry,
		Egress: compilerEgressPolicy(t, "api.example.com"),
	}
	_, err := compiler.CompileBackend(context.Background(), ModelBackendRequest{
		NamespaceID: compilerNamespaceID, BackendID: compilerBackendID,
		ProviderID: "provider", ProviderModelID: "model",
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unsafe Provider compiler output error = %v", err)
	}
}

func TestProviderDefinitionRequiresCompilerAndDiscoveryPaths(t *testing.T) {
	provider := validDefinition("provider", 1)
	provider.Interfaces[0].Compiler.Config["path"] = ""
	if _, err := NewRegistry(testRegistryOptions(provider)); err == nil {
		t.Fatal("provider without a compiled connection path was accepted")
	}
	provider = validDefinition("provider", 1)
	provider.Discovery.Path = ""
	if _, err := NewRegistry(testRegistryOptions(provider)); err == nil {
		t.Fatal("provider discovery without a path was accepted")
	}
}

func compilerSnapshot(t *testing.T) (*Snapshot, *Registry) {
	t.Helper()
	provider := validDefinition("provider", 1)
	provider.Origin = Origin{Mode: OriginFixed, DefaultURL: "https://api.example.com/v1"}
	provider.Credential = Credential{
		Mode: CredentialRequired, AdapterID: "bearer", Label: "API key",
	}
	provider.Interfaces[0].Compiler = Compiler{
		AdapterID: "test.fields.v1",
		Config: map[string]any{
			"path":    "/chat/completions",
			"headers": map[string]any{"X-Provider-Version": "1"},
		},
	}
	provider.ConnectionFields = []ConnectionField{{
		Name: "region", Label: "Region", Kind: FieldText, Required: true,
	}}
	registry := mustTestRegistry(t, provider)
	return registry.Snapshot(), registry
}

func compilerRegistry(t *testing.T) *Registry {
	t.Helper()
	_, registry := compilerSnapshot(t)
	return registry
}

func compilerCredential() providercredential.Credential {
	now := time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC)
	versionID := compilerVersionID
	return providercredential.Credential{
		ID: compilerCredentialID, NamespaceID: compilerNamespaceID, Name: "Provider key",
		ProviderID: "provider", CredentialMode: providercredential.ModeRequired,
		CredentialAdapterID: "bearer", CatalogRevision: oldCatalogRevision,
		NormalizedOrigin: "https://api.example.com/v1", Status: providercredential.StatusActive,
		ActiveVersionID: &versionID, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
}

func compilerEgressPolicy(t *testing.T, host string) backendegress.Policy {
	t.Helper()
	policy, err := backendegress.Compile(backendegress.Config{
		Version: "v1", Schemes: []string{"https"},
		Hosts: []backendegress.HostConfig{{Host: host, Ports: []uint16{443}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return policy
}
