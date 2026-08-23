package routingmanagement

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const modelPatchCatalogRevision = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

type modelPatchStore struct {
	Store
	current Model
	updated routingsnapshot.Model
}

func (store *modelPatchStore) GetModel(context.Context, string, string) (Model, error) {
	return store.current, nil
}

func (store *modelPatchStore) NamespaceCurrency(context.Context, string) (string, error) {
	return "USD", nil
}

func (store *modelPatchStore) UpdateModel(
	_ context.Context, _ string, _ string, _ int64, model routingsnapshot.Model, _ MutationContext,
) (Model, RevisionReceipt, error) {
	store.updated = model
	return Model{ResourceIdentity: ResourceIdentity{ID: model.ID, Revision: 8}, Current: model},
		RevisionReceipt{ResourceRevision: 8}, nil
}

func TestPatchModelPreservesServerOwnedBackendConfiguration(t *testing.T) {
	backend := routingsnapshot.Backend{
		ID: "11111111-1111-4111-8111-111111111111", ProviderID: "private",
		WireFormat: "openai.chat.v1", Origin: "https://models.example.com/v1",
		ProviderModelID: "private/model", ProviderCredentialID: "22222222-2222-4222-8222-222222222222",
		Connection: routingsnapshot.BackendConnection{
			Path: "/chat/completions", Headers: map[string]string{"X-Provider-Version": "2026-08-01"},
		},
		Weight: "1",
	}
	store := &modelPatchStore{current: Model{
		ResourceIdentity: ResourceIdentity{NamespaceID: "33333333-3333-4333-8333-333333333333", ID: "model_one", Name: "Model One", Revision: 7},
		Current: routingsnapshot.Model{
			ID: "model_one", Revision: 3, CatalogRevision: modelPatchCatalogRevision, Name: "Model One",
			Aliases: []string{"model-one"}, Capabilities: []string{"chat"},
			Execution: routingsnapshot.ModelExecution{MaxRetries: 1, RequestTimeout: "30s", StreamTimeout: "2m"},
			Backends:  []routingsnapshot.Backend{backend},
		},
	}}
	service, err := NewService(ServiceOptions{
		Store:         store,
		CursorKeyring: testRoutingCursorKeyring(),
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, errors.New("backend compilation must not run")
			}),
			Registry: &providercatalog.Registry{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	execution := routingsnapshot.ModelExecution{MaxRetries: 4, RequestTimeout: "45s", StreamTimeout: "5m"}
	aliases := []string{}
	inputPrice, outputPrice := "0.25", "1.5"
	pricing := routingsnapshot.ModelPricing{
		InputCostPerMillionTokens: &inputPrice, OutputCostPerMillionTokens: &outputPrice,
	}
	updated, receipt, err := service.PatchModel(
		context.Background(), store.current.NamespaceID, store.current.ID, 7,
		ModelPatch{Aliases: &aliases, Execution: &execution, Pricing: &pricing}, MutationContext{RequestID: "request-one"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.ResourceRevision != 8 || updated.Current.Revision != 4 {
		t.Fatalf("revision result = %#v, receipt = %#v", updated, receipt)
	}
	if store.updated.CatalogRevision != modelPatchCatalogRevision ||
		!reflect.DeepEqual(store.updated.Backends, []routingsnapshot.Backend{backend}) {
		t.Fatalf("server-owned backend changed: %#v", store.updated)
	}
	if store.updated.Execution != execution || store.updated.Pricing.InputCostPerMillionTokens == nil ||
		*store.updated.Pricing.InputCostPerMillionTokens != inputPrice ||
		store.updated.Pricing.CacheReadCostPerMillionTokens == nil ||
		*store.updated.Pricing.CacheReadCostPerMillionTokens != inputPrice {
		t.Fatalf("normalized execution or pricing = %#v, %#v", store.updated.Execution, store.updated.Pricing)
	}
	if len(store.updated.Aliases) != 0 {
		t.Fatalf("explicitly cleared aliases = %#v", store.updated.Aliases)
	}
}

func TestPatchModelRejectsAnEmptyPatch(t *testing.T) {
	store := &modelPatchStore{current: Model{
		ResourceIdentity: ResourceIdentity{ID: "model_one", Revision: 1},
	}}
	service, err := NewService(ServiceOptions{
		Store:         store,
		CursorKeyring: testRoutingCursorKeyring(),
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, nil
			}),
			Registry: &providercatalog.Registry{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := service.PatchModel(
		context.Background(), "33333333-3333-4333-8333-333333333333", "model_one", 1,
		ModelPatch{}, MutationContext{},
	); !errors.Is(err, ErrInvalid) {
		t.Fatalf("empty patch error = %v, want ErrInvalid", err)
	}
}
