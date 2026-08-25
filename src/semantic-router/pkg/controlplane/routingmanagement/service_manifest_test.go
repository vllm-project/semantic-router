package routingmanagement

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type manifestServiceStore struct {
	Store
	previewNamespace string
	previewExpected  int64
	previewSnapshot  *routingsnapshot.Snapshot
	currentSnapshot  *routingsnapshot.Snapshot
	credentialIDs    map[string]string
	credentialNames  map[string]string
}

func (store *manifestServiceStore) ProviderCredentialIDsByName(
	_ context.Context, _ string, names []string,
) (map[string]string, error) {
	result := make(map[string]string, len(names))
	for _, name := range names {
		id, found := store.credentialIDs[name]
		if !found {
			return nil, ErrManifest
		}
		result[name] = id
	}
	return result, nil
}

func (store *manifestServiceStore) ProviderCredentialNamesByID(
	_ context.Context, _ string, ids []string,
) (map[string]string, error) {
	result := make(map[string]string, len(ids))
	for _, id := range ids {
		name, found := store.credentialNames[id]
		if !found {
			return nil, ErrPublication
		}
		result[id] = name
	}
	return result, nil
}

func (store *manifestServiceStore) PreviewManifest(
	_ context.Context, namespaceID string, expected int64, snapshot *routingsnapshot.Snapshot,
) (ManifestDiff, error) {
	store.previewNamespace = namespaceID
	store.previewExpected = expected
	store.previewSnapshot = snapshot
	return ManifestDiff{Models: ManifestResourceDiff{Create: []string{"Model One"}}}, nil
}

func (store *manifestServiceStore) CurrentManifest(
	context.Context, string,
) (*routingsnapshot.Snapshot, int64, error) {
	return store.currentSnapshot, 9, nil
}

type manifestCodecStub struct {
	snapshot *routingsnapshot.Snapshot
	document []byte
	encoded  *routingsnapshot.Snapshot
}

func (codec manifestCodecStub) Decode([]byte) (*routingsnapshot.Snapshot, error) {
	return codec.snapshot, nil
}

func (codec *manifestCodecStub) Encode(snapshot *routingsnapshot.Snapshot) ([]byte, error) {
	codec.encoded = snapshot
	return append([]byte(nil), codec.document...), nil
}

func TestManifestServiceKeepsConfigCodecOutsideRoutingDomain(t *testing.T) {
	namespaceID := "33333333-3333-4333-8333-333333333333"
	credentialID := "11111111-1111-4111-8111-111111111111"
	credentialName := "Readable provider credential"
	authoring := manifestCredentialSnapshot(t, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", credentialName)
	internal := manifestCredentialSnapshot(t, namespaceID, credentialID)
	store := &manifestServiceStore{
		currentSnapshot: internal,
		credentialIDs:   map[string]string{credentialName: credentialID},
		credentialNames: map[string]string{credentialID: credentialName},
	}
	codec := &manifestCodecStub{
		snapshot: authoring, document: []byte("version: v0.3\n"),
	}
	service, err := NewService(ServiceOptions{
		Store: store, ManifestCodec: codec, CursorKeyring: testRoutingCursorKeyring(),
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, errors.New("not used")
			}),
			Registry: &providercatalog.Registry{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	prepared, err := service.PrepareManifest(context.Background(), namespaceID, []byte("manifest"))
	if err != nil || !reflect.DeepEqual(prepared.CredentialIDs, []string{credentialID}) {
		t.Fatalf("prepared credentials = %#v, err = %v", prepared.CredentialIDs, err)
	}
	if prepared.Snapshot.NamespaceID != namespaceID ||
		prepared.Snapshot.Models[0].Backends[0].ProviderCredentialID != credentialID {
		t.Fatalf("prepared snapshot = %#v", prepared.Snapshot)
	}
	verified, err := routingsnapshot.Compile(prepared.Snapshot.Bundle)
	if err != nil || verified.Digest != prepared.Snapshot.Digest ||
		verified.SemanticDigest != prepared.Snapshot.SemanticDigest {
		t.Fatalf("prepared snapshot digest = %#v, %v", prepared.Snapshot, err)
	}
	result, err := service.ImportManifest(context.Background(), namespaceID, ManifestImportRequest{
		Prepared: prepared, DryRun: true, ExpectedRevision: 8,
	}, MutationContext{})
	if err != nil || !reflect.DeepEqual(result.Diff.Models.Create, []string{"Model One"}) {
		t.Fatalf("dry-run result = %#v, err = %v", result, err)
	}
	if store.previewNamespace != namespaceID || store.previewExpected != 8 ||
		store.previewSnapshot.NamespaceID != namespaceID {
		t.Fatalf("preview input = namespace %q, expected %d, snapshot %#v",
			store.previewNamespace, store.previewExpected, store.previewSnapshot)
	}
	document, revision, err := service.ExportCurrentManifest(context.Background(), namespaceID)
	if err != nil || revision != 9 || string(document) != "version: v0.3\n" {
		t.Fatalf("export = %q revision %d, err = %v", document, revision, err)
	}
	if codec.encoded == nil || codec.encoded.Models[0].Backends[0].ProviderCredentialID != credentialName {
		t.Fatalf("exported snapshot leaked generated credential identity: %#v", codec.encoded)
	}
}

func TestPrepareManifestRejectsCredentialNameOutsideNamespace(t *testing.T) {
	namespaceID := "33333333-3333-4333-8333-333333333333"
	source := manifestCredentialSnapshot(t, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "other-namespace")
	codec := &manifestCodecStub{snapshot: source}
	service, err := NewService(ServiceOptions{
		Store: &manifestServiceStore{}, ManifestCodec: codec,
		CursorKeyring: testRoutingCursorKeyring(),
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, errors.New("not used")
			}),
			Registry: &providercatalog.Registry{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	if _, err := service.PrepareManifest(context.Background(), namespaceID, []byte("manifest")); !errors.Is(err, ErrManifest) {
		t.Fatalf("PrepareManifest() error = %v", err)
	}
}

func TestImportManifestRejectsPreparedSnapshotMutation(t *testing.T) {
	namespaceID := "33333333-3333-4333-8333-333333333333"
	credentialID := "11111111-1111-4111-8111-111111111111"
	credentialName := "Readable provider credential"
	codec := &manifestCodecStub{
		snapshot: manifestCredentialSnapshot(
			t, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", credentialName,
		),
	}
	service, err := NewService(ServiceOptions{
		Store: &manifestServiceStore{
			credentialIDs: map[string]string{credentialName: credentialID},
		},
		ManifestCodec: codec, CursorKeyring: testRoutingCursorKeyring(),
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return nil, errors.New("not used")
			}),
			Registry: &providercatalog.Registry{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	prepared, err := service.PrepareManifest(context.Background(), namespaceID, []byte("manifest"))
	if err != nil {
		t.Fatal(err)
	}
	prepared.Snapshot.Bundle.Models[0].Backends[0].ProviderCredentialID =
		"22222222-2222-4222-8222-222222222222"
	if _, err := service.ImportManifest(
		context.Background(), namespaceID,
		ManifestImportRequest{Prepared: prepared, DryRun: true, ExpectedRevision: 1},
		MutationContext{},
	); !errors.Is(err, ErrManifest) {
		t.Fatalf("ImportManifest() error = %v", err)
	}
}

func manifestCredentialSnapshot(t *testing.T, namespaceID, credentialReference string) *routingsnapshot.Snapshot {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: namespaceID,
		Revision:    1,
		Models: []routingsnapshot.Model{{
			ID: "mdl_manifest", Revision: 1,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "remote/model",
			Backends: []routingsnapshot.Backend{{
				ID:         "11111111-2222-4333-8444-555555555555",
				ProviderID: "openai-compatible", WireFormat: llmprotocol.OpenAIChatV1,
				Origin: "https://models.example", ProviderModelID: "remote/model",
				ProviderCredentialID: credentialReference,
				Connection:           routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
