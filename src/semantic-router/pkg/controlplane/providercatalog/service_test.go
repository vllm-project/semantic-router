package providercatalog

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

var testCursorKeyring = securitykeyring.Symmetric{
	ActiveVersion: "cursor-v1",
	Keys:          map[string][]byte{"cursor-v1": []byte(strings.Repeat("c", 32))},
}

func staticSnapshotSource(snapshot *Snapshot) SnapshotSource {
	return SnapshotSourceFunc(func(context.Context) (*Snapshot, error) { return snapshot, nil })
}

type switchingSnapshotSource struct{ snapshot *Snapshot }

func (source *switchingSnapshotSource) ActiveSnapshot(context.Context) (*Snapshot, error) {
	return source.snapshot, nil
}

func TestServiceListUsesStableFilteredKeysetAndRevision(t *testing.T) {
	providerZ := validDefinition("provider-z", 20)
	providerA := validDefinition("provider-a", 10)
	providerA.Display.Category = "Private"
	providerA.Capabilities = []string{"streaming"}
	providerB := validDefinition("provider-b", 10)
	providerB.Display.Category = "Private"
	providerB.Capabilities = []string{"streaming", "tools"}
	snapshot := mustTestRegistry(t, providerZ, providerB, providerA).Snapshot()
	source := &switchingSnapshotSource{snapshot: snapshot}
	service, err := NewService(source, ServiceOptions{CursorKeyring: testCursorKeyring})
	if err != nil {
		t.Fatal(err)
	}

	first, err := service.List(context.Background(), ListRequest{
		PageSize: 1, Category: "Private", Capability: "streaming", Search: "provider",
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.CatalogRevision != snapshot.Revision() || len(first.Providers) != 1 ||
		first.Providers[0].ID != "provider-a" || !first.HasMore || first.NextCursor == "" {
		t.Fatalf("first page = %+v", first)
	}
	if len(first.Categories) != 2 || first.Categories[0] != "Model APIs" || first.Categories[1] != "Private" {
		t.Fatalf("categories = %v, want stable complete facets", first.Categories)
	}
	second, err := service.List(context.Background(), ListRequest{
		PageSize: 1, Cursor: first.NextCursor, Category: "Private", Capability: "streaming", Search: "provider",
	})
	if err != nil || len(second.Providers) != 1 || second.Providers[0].ID != "provider-b" || second.HasMore {
		t.Fatalf("second page = (%+v, %v)", second, err)
	}
	if _, err := service.List(context.Background(), ListRequest{
		PageSize: 1, Cursor: first.NextCursor, Category: "Model APIs", Capability: "streaming", Search: "provider",
	}); !errors.Is(err, ErrInvalidCursor) {
		t.Fatalf("cursor/filter mismatch error = %v, want ErrInvalidCursor", err)
	}

	tampered := []byte(first.NextCursor)
	if tampered[0] == 'A' {
		tampered[0] = 'B'
	} else {
		tampered[0] = 'A'
	}
	if _, err := service.List(context.Background(), ListRequest{
		PageSize: 1, Cursor: string(tampered), Category: "Private", Capability: "streaming", Search: "provider",
	}); !errors.Is(err, ErrInvalidCursor) {
		t.Fatalf("tampered cursor error = %v, want ErrInvalidCursor", err)
	}

	providerZ.Display.Description = "Changed provider metadata."
	changedSnapshot := mustTestRegistry(t, providerZ, providerB, providerA).Snapshot()
	source.snapshot = changedSnapshot
	if _, err := service.List(context.Background(), ListRequest{
		PageSize: 1, Cursor: first.NextCursor, Category: "Private", Capability: "streaming", Search: "provider",
	}); !errors.Is(err, ErrStaleCursor) {
		t.Fatalf("old revision cursor error = %v, want ErrStaleCursor", err)
	}
}

func TestServiceDetailAndSnapshotAreDefensive(t *testing.T) {
	snapshot := mustTestRegistry(t, validDefinition("provider-a", 1)).Snapshot()
	service, err := NewService(staticSnapshotSource(snapshot), ServiceOptions{CursorKeyring: testCursorKeyring})
	if err != nil {
		t.Fatal(err)
	}
	detail, err := service.Get(context.Background(), "provider-a")
	if err != nil {
		t.Fatal(err)
	}
	detail.Provider.Display.Name = "mutated"
	detail.Provider.Interfaces[0].Compiler.Config["path"] = "/mutated"
	detail.Provider.Discovery.Headers = map[string]string{"X-Discovery-Version": "mutated"}
	again, err := service.Get(context.Background(), "provider-a")
	if err != nil || again.Provider.Display.Name == "mutated" || again.Provider.Interfaces[0].Compiler.Config["path"] == "/mutated" ||
		again.Provider.Discovery.Headers["X-Discovery-Version"] == "mutated" {
		t.Fatalf("caller mutation escaped into service snapshot: %+v, %v", again, err)
	}
	if _, err := service.Get(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing provider error = %v, want ErrNotFound", err)
	}
	references := snapshot.IntegrationReferences()
	references[0].ProviderID = "mutated"
	if snapshot.IntegrationReferences()[0].ProviderID == "mutated" {
		t.Fatal("IntegrationReferences returned mutable snapshot storage")
	}
}

type recordingDiscoveryValidator struct {
	id     string
	called int
	plan   DiscoveryPlan
	err    error
}

func (validator *recordingDiscoveryValidator) AdapterID() string { return validator.id }
func (validator *recordingDiscoveryValidator) ValidateDiscovery(_ context.Context, plan DiscoveryPlan) error {
	validator.called++
	validator.plan = plan
	return validator.err
}

func TestPrepareDiscoveryValidatesGenericRequestWithoutExecutingNetwork(t *testing.T) {
	provider := validDefinition("provider-a", 1)
	provider.Origin = Origin{Mode: OriginUserSupplied, Label: "Base URL"}
	provider.Credential = Credential{Mode: CredentialNone}
	provider.Interfaces[0].Compiler = Compiler{
		AdapterID: "test.fields.v1", Config: map[string]any{"path": "/chat/completions"},
	}
	provider.ConnectionFields = []ConnectionField{
		{Name: "region", Label: "Region", Kind: FieldText, Required: true},
		{Name: "preview", Label: "Preview", Kind: FieldBoolean, Default: "false"},
		{Name: "shard", Label: "Shard", Kind: FieldInteger, Default: "2"},
		{Name: "tier", Label: "Tier", Kind: FieldSelect, Default: "standard", Options: []FieldOption{{Value: "standard", Label: "Standard"}, {Value: "fast", Label: "Fast"}}},
	}
	snapshot := mustTestRegistry(t, provider).Snapshot()
	validator := &recordingDiscoveryValidator{id: "openai.models.v1"}
	registry, err := NewDiscoveryRegistry([]DiscoveryRequestValidator{validator})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(staticSnapshotSource(snapshot), ServiceOptions{CursorKeyring: testCursorKeyring, DiscoveryPlugins: registry})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := service.PrepareDiscovery(context.Background(), "provider-a", DiscoverModelsRequest{
		NamespaceID:      "11111111-1111-4111-8111-111111111111",
		Origin:           "HTTPS://Catalog.Example.com:443/v1/",
		ConnectionFields: map[string]any{"region": "us-east", "shard": json.Number("3")},
		Search:           "reasoning", PageSize: 25, ProviderCursor: "opaque-cursor",
	})
	if err != nil {
		t.Fatal(err)
	}
	if validator.called != 1 || plan.NormalizedOrigin != "https://catalog.example.com/v1" ||
		plan.DiscoveryAdapterID != validator.id || plan.CredentialAdapterID != "" ||
		plan.ConnectionFields["preview"].Value != "false" || plan.ConnectionFields["shard"].Value != "3" ||
		plan.ConnectionFields["tier"].Value != "standard" {
		t.Fatalf("validated discovery plan = %+v, validator = %+v", plan, validator)
	}
	if _, err := service.PrepareDiscovery(context.Background(), "provider-a", DiscoverModelsRequest{
		NamespaceID: "11111111-1111-4111-8111-111111111111", Origin: "https://catalog.example.com",
		ConnectionFields: map[string]any{"region": "us-east", "unknown": "value"},
	}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unknown connection field error = %v, want ErrInvalidRequest", err)
	}
	if _, err := service.PrepareDiscovery(context.Background(), "provider-a", DiscoverModelsRequest{
		NamespaceID: "11111111-1111-4111-8111-111111111111", Origin: "https://catalog.example.com",
		ConnectionFields: map[string]any{"region": "us-east", "shard": 1.5},
	}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("floating integer field error = %v, want ErrInvalidRequest", err)
	}
}

func TestPrepareDiscoveryEnforcesCredentialOriginAndPluginBoundaries(t *testing.T) {
	snapshot := mustTestRegistry(t, validDefinition("provider-a", 1)).Snapshot()
	registry, _ := NewDiscoveryRegistry([]DiscoveryRequestValidator{
		&recordingDiscoveryValidator{id: "openai.models.v1"},
	})
	service, newServiceErr := NewService(staticSnapshotSource(snapshot), ServiceOptions{CursorKeyring: testCursorKeyring, DiscoveryPlugins: registry})
	if newServiceErr != nil {
		t.Fatal(newServiceErr)
	}
	base := DiscoverModelsRequest{
		NamespaceID:      "11111111-1111-4111-8111-111111111111",
		ConnectionFields: map[string]any{"region": "us-east"},
	}
	if _, err := service.PrepareDiscovery(context.Background(), "provider-a", base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("missing required credential error = %v", err)
	}
	base.CredentialID = "22222222-2222-4222-8222-222222222222"
	base.Origin = "https://override.example.com"
	if _, err := service.PrepareDiscovery(context.Background(), "provider-a", base); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("fixed origin override error = %v", err)
	}

	serviceWithoutPlugin, newServiceErr := NewService(staticSnapshotSource(snapshot), ServiceOptions{CursorKeyring: testCursorKeyring})
	if newServiceErr != nil {
		t.Fatal(newServiceErr)
	}
	base.Origin = ""
	if _, err := serviceWithoutPlugin.PrepareDiscovery(context.Background(), "provider-a", base); !errors.Is(err, ErrDiscoveryPluginUnavailable) {
		t.Fatalf("missing discovery plugin error = %v", err)
	}
}
