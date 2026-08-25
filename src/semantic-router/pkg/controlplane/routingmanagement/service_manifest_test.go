package routingmanagement

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type manifestServiceStore struct {
	Store
	previewNamespace string
	previewExpected  int64
	previewSnapshot  *routingsnapshot.Snapshot
	currentSnapshot  *routingsnapshot.Snapshot
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
	snapshot    *routingsnapshot.Snapshot
	credentials []string
	document    []byte
}

func (codec manifestCodecStub) Decode([]byte) (*routingsnapshot.Snapshot, error) {
	return codec.snapshot, nil
}

func (codec manifestCodecStub) Encode(*routingsnapshot.Snapshot) ([]byte, error) {
	return append([]byte(nil), codec.document...), nil
}

func (codec manifestCodecStub) CredentialIDs([]byte) ([]string, error) {
	return append([]string(nil), codec.credentials...), nil
}

func TestManifestServiceKeepsConfigCodecOutsideRoutingDomain(t *testing.T) {
	namespaceID := "33333333-3333-4333-8333-333333333333"
	snapshot := &routingsnapshot.Snapshot{}
	store := &manifestServiceStore{currentSnapshot: snapshot}
	codec := manifestCodecStub{
		snapshot: snapshot, credentials: []string{"11111111-1111-4111-8111-111111111111"},
		document: []byte("version: v0.3\n"),
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
	credentials, err := service.ManifestCredentialIDs([]byte("manifest"))
	if err != nil || !reflect.DeepEqual(credentials, codec.credentials) {
		t.Fatalf("credentials = %#v, err = %v", credentials, err)
	}
	result, err := service.ImportManifest(context.Background(), namespaceID, ManifestImportRequest{
		Document: []byte("manifest"), DryRun: true, ExpectedRevision: 8,
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
}
