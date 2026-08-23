package routingmanagement

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type listServiceStore struct {
	Store
	models            ListResult[Model]
	recipes           ListResult[Recipe]
	entrypoints       ListResult[Entrypoint]
	snapshots         ListResult[SnapshotMetadata]
	lastQuery         ListQuery
	lastSnapshotQuery SnapshotListQuery
}

func (store *listServiceStore) ListModels(
	_ context.Context, _ string, query ListQuery,
) (ListResult[Model], error) {
	store.lastQuery = query
	return store.models, nil
}

func (store *listServiceStore) ListRecipes(
	_ context.Context, _ string, query ListQuery,
) (ListResult[Recipe], error) {
	store.lastQuery = query
	return store.recipes, nil
}

func (store *listServiceStore) ListEntrypoints(
	_ context.Context, _ string, query ListQuery,
) (ListResult[Entrypoint], error) {
	store.lastQuery = query
	return store.entrypoints, nil
}

func (store *listServiceStore) ListSnapshots(
	_ context.Context, _ string, query SnapshotListQuery,
) (ListResult[SnapshotMetadata], error) {
	store.lastSnapshotQuery = query
	return store.snapshots, nil
}

func newListService(t *testing.T, store Store, keyring securitykeyring.Symmetric) *Service {
	t.Helper()
	service, err := NewService(ServiceOptions{
		Store: store,
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, nil
			}),
			Registry: &providercatalog.Registry{},
		},
		CursorKeyring: keyring,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func TestRoutingListCursorBindsEveryVisibilityDimension(t *testing.T) {
	namespaceID := "11111111-1111-4111-8111-111111111111"
	createdAt := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	store := &listServiceStore{models: ListResult[Model]{
		Items:   []Model{{ResourceIdentity: ResourceIdentity{ID: "model_one", CreatedAt: createdAt}}},
		HasMore: true,
	}}
	service := newListService(t, store, testRoutingCursorKeyring())
	scope := accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true}
	first, err := service.ListModels(context.Background(), namespaceID, PageRequest{
		PageSize: 10, Search: "  MODEL_100%  ", Scope: scope,
	})
	if err != nil || first.NextCursor == "" || !first.HasMore {
		t.Fatalf("first page = %#v, %v", first, err)
	}
	if store.lastQuery.Search != "model_100%" || store.lastQuery.Limit != 10 {
		t.Fatalf("normalized repository query = %#v", store.lastQuery)
	}
	if _, err := service.ListModels(context.Background(), namespaceID, PageRequest{
		PageSize: 10, Search: "model_100%", Scope: scope, Cursor: first.NextCursor,
	}); err != nil || store.lastQuery.After == nil || store.lastQuery.After.ID != "model_one" {
		t.Fatalf("cursor continuation = %#v, %v", store.lastQuery.After, err)
	}

	otherScope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceModel: {"model_one"},
		},
	}
	requests := []PageRequest{
		{PageSize: 10, Search: "different", Scope: scope, Cursor: first.NextCursor},
		{PageSize: 10, Search: "model_100%", Status: StatusActive, Scope: scope, Cursor: first.NextCursor},
		{PageSize: 10, Search: "model_100%", Scope: otherScope, Cursor: first.NextCursor},
	}
	for index, request := range requests {
		if _, err := service.ListModels(context.Background(), namespaceID, request); !errors.Is(err, ErrInvalid) {
			t.Fatalf("binding mismatch %d error = %v, want ErrInvalid", index, err)
		}
	}
	if _, err := service.ListRecipes(context.Background(), namespaceID, PageRequest{
		PageSize: 10, Search: "model_100%", Scope: scope, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cross-resource cursor error = %v, want ErrInvalid", err)
	}
	replacement := "A"
	if strings.HasSuffix(first.NextCursor, replacement) {
		replacement = "B"
	}
	tampered := first.NextCursor[:len(first.NextCursor)-1] + replacement
	if _, err := service.ListModels(context.Background(), namespaceID, PageRequest{
		PageSize: 10, Search: "model_100%", Scope: scope, Cursor: tampered,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("tampered cursor error = %v, want ErrInvalid", err)
	}
}

func TestRoutingListCursorSupportsExplicitKeyRotation(t *testing.T) {
	oldKeys := securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{0x11}, 32)},
	}
	oldCodec, testRoutingListCursorSupportsExplicitKeyRotationErr := newRoutingCursorCodec(oldKeys)
	if testRoutingListCursorSupportsExplicitKeyRotationErr != nil {
		t.Fatal(testRoutingListCursorSupportsExplicitKeyRotationErr)
	}
	cursor, testRoutingListCursorSupportsExplicitKeyRotationErr := oldCodec.encode(routingCursorPayload{
		NamespaceID: "11111111-1111-4111-8111-111111111111", ResourceKind: routingResourceModel,
		ScopeDigest: strings.Repeat("a", 43), CreatedAt: time.Now().UTC(), ID: "model_one",
	})
	oldCodec.close()
	if testRoutingListCursorSupportsExplicitKeyRotationErr != nil {
		t.Fatal(testRoutingListCursorSupportsExplicitKeyRotationErr)
	}
	rotated, testRoutingListCursorSupportsExplicitKeyRotationErr := newRoutingCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v2", Keys: map[string][]byte{
			"v1": bytes.Repeat([]byte{0x11}, 32), "v2": bytes.Repeat([]byte{0x22}, 32),
		},
	})
	if testRoutingListCursorSupportsExplicitKeyRotationErr != nil {
		t.Fatal(testRoutingListCursorSupportsExplicitKeyRotationErr)
	}
	if _, err := rotated.decode(cursor); err != nil {
		t.Fatalf("retained cursor key rejected: %v", err)
	}
	rotated.close()
	retired, testRoutingListCursorSupportsExplicitKeyRotationErr := newRoutingCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v2", Keys: map[string][]byte{"v2": bytes.Repeat([]byte{0x22}, 32)},
	})
	if testRoutingListCursorSupportsExplicitKeyRotationErr != nil {
		t.Fatal(testRoutingListCursorSupportsExplicitKeyRotationErr)
	}
	defer retired.close()
	if _, err := retired.decode(cursor); !errors.Is(err, ErrInvalid) {
		t.Fatalf("retired cursor key error = %v, want ErrInvalid", err)
	}
}

func TestRoutingSnapshotCursorIsNamespaceBoundAndDescending(t *testing.T) {
	namespaceID := "11111111-1111-4111-8111-111111111111"
	store := &listServiceStore{snapshots: ListResult[SnapshotMetadata]{
		Items:   []SnapshotMetadata{{NamespaceID: namespaceID, RoutingRevision: 41}},
		HasMore: true,
	}}
	service := newListService(t, store, testRoutingCursorKeyring())
	first, err := service.ListSnapshots(context.Background(), namespaceID, SnapshotPageRequest{PageSize: 7})
	if err != nil || first.NextCursor == "" || !first.HasMore {
		t.Fatalf("first snapshot page = %#v, %v", first, err)
	}
	if store.lastSnapshotQuery.Limit != 7 || store.lastSnapshotQuery.BeforeRevision != nil {
		t.Fatalf("first repository query = %#v", store.lastSnapshotQuery)
	}
	if _, err := service.ListSnapshots(context.Background(), namespaceID, SnapshotPageRequest{
		PageSize: 7, Cursor: first.NextCursor,
	}); err != nil || store.lastSnapshotQuery.BeforeRevision == nil || *store.lastSnapshotQuery.BeforeRevision != 41 {
		t.Fatalf("snapshot continuation = %#v, %v", store.lastSnapshotQuery, err)
	}
	if _, err := service.ListSnapshots(context.Background(),
		"22222222-2222-4222-8222-222222222222",
		SnapshotPageRequest{PageSize: 7, Cursor: first.NextCursor},
	); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cross-Namespace cursor error = %v, want ErrInvalid", err)
	}
	if _, err := service.ListModels(context.Background(), namespaceID, PageRequest{
		PageSize: 7,
		Scope:    accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true},
		Cursor:   first.NextCursor,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("snapshot cursor accepted for Model list: %v", err)
	}
}

func TestRoutingListRejectsUnboundedOrControlSearch(t *testing.T) {
	service := newListService(t, &listServiceStore{}, testRoutingCursorKeyring())
	scope := accesscontrol.ResultScope{
		NamespaceID: "11111111-1111-4111-8111-111111111111", All: true,
	}
	for _, search := range []string{strings.Repeat("界", 201), "model\nsecret"} {
		if _, err := service.ListModels(context.Background(), string(scope.NamespaceID), PageRequest{
			Search: search, Scope: scope,
		}); !errors.Is(err, ErrInvalid) {
			t.Fatalf("search %q error = %v, want ErrInvalid", search, err)
		}
	}
}
