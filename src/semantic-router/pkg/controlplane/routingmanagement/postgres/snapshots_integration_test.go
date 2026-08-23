package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRoutingSnapshotsListAndExportExactImmutableRevision(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	first := compileSnapshotFixture(t, namespaceID, 1, "provider-model-v1")
	second := compileSnapshotFixture(t, namespaceID, 2, "provider-model-v2")
	insertSnapshotFixture(t, ctx, db, first, routingmanagement.SnapshotStatusRetired)
	insertSnapshotFixture(t, ctx, db, second, routingmanagement.SnapshotStatusActive)

	page, err := store.ListSnapshots(ctx, namespaceID, routingmanagement.SnapshotListQuery{Limit: 1})
	if err != nil || !page.HasMore || len(page.Items) != 1 || page.Items[0].RoutingRevision != 2 ||
		page.Items[0].ContentDigest != "sha256:"+second.Digest || page.Items[0].MemberCount != 3 {
		t.Fatalf("first routing snapshot page = %#v, %v", page, err)
	}
	before := page.Items[0].RoutingRevision
	page, err = store.ListSnapshots(ctx, namespaceID, routingmanagement.SnapshotListQuery{
		Limit: 1, BeforeRevision: &before,
	})
	if err != nil || page.HasMore || len(page.Items) != 1 || page.Items[0].RoutingRevision != 1 {
		t.Fatalf("second routing snapshot page = %#v, %v", page, err)
	}

	detail, err := store.GetSnapshot(ctx, namespaceID, 2)
	if err != nil || detail.Export.Digest != second.Digest || detail.Export.Revision != 2 ||
		len(detail.Members) != 3 || detail.Metadata.MemberCount != 3 {
		t.Fatalf("routing snapshot detail = %#v, %v", detail, err)
	}
	if _, err := store.GetSnapshot(ctx, namespaceID, 99); !errors.Is(err, routingmanagement.ErrNotFound) {
		t.Fatalf("missing routing snapshot error = %v, want ErrNotFound", err)
	}
}

func TestRoutingSnapshotExportRejectsMemberDrift(t *testing.T) {
	db, namespaceID := routingIntegrationDatabase(t)
	store := newRoutingStore(t, db)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	snapshot := compileSnapshotFixture(t, namespaceID, 1, "provider-model")
	insertSnapshotFixture(t, ctx, db, snapshot, routingmanagement.SnapshotStatusActive)
	if _, err := db.ExecContext(ctx, `UPDATE routing_snapshot_members
SET resource_revision=resource_revision+1
WHERE namespace_id=$1 AND routing_revision=1 AND resource_type='model'`, namespaceID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetSnapshot(ctx, namespaceID, 1); !errors.Is(err, routingmanagement.ErrPublication) {
		t.Fatalf("drifted routing snapshot error = %v, want ErrPublication", err)
	}
}

func compileSnapshotFixture(
	t *testing.T,
	namespaceID string,
	revision int64,
	providerModelID string,
) *routingsnapshot.Snapshot {
	t.Helper()
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: namespaceID, Revision: revision, Currency: "USD",
		Models:      []routingsnapshot.Model{routingTestModel(1, providerModelID)},
		Recipes:     []routingsnapshot.Recipe{routingTestRecipe(1, "Simple")},
		Entrypoints: []routingsnapshot.Entrypoint{routingTestEntrypoint(1, 1, 1)},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func insertSnapshotFixture(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	snapshot *routingsnapshot.Snapshot,
	status routingmanagement.SnapshotStatus,
) {
	t.Helper()
	payload, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := hex.DecodeString(snapshot.Digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO routing_snapshots
  (namespace_id,routing_revision,content_digest,compiled_blob,status,activated_at)
VALUES ($1,$2,$3,$4,$5,CASE WHEN $5 IN ('active','retired') THEN clock_timestamp() END)`,
		snapshot.NamespaceID, snapshot.Revision, digest, payload, string(status)); err != nil {
		t.Fatal(err)
	}
	for _, member := range []routingmanagement.SnapshotMember{
		{ResourceType: "model", ResourceID: snapshot.Models[0].ID, ResourceRevision: snapshot.Models[0].Revision},
		{ResourceType: "recipe", ResourceID: snapshot.Recipes[0].ID, ResourceRevision: snapshot.Recipes[0].Revision},
		{ResourceType: "entrypoint", ResourceID: snapshot.Entrypoints[0].ID, ResourceRevision: snapshot.Entrypoints[0].Revision},
	} {
		if _, err := db.ExecContext(ctx, `INSERT INTO routing_snapshot_members
  (namespace_id,routing_revision,resource_type,resource_id,resource_revision)
VALUES ($1,$2,$3,$4,$5)`, snapshot.NamespaceID, snapshot.Revision,
			member.ResourceType, member.ResourceID, member.ResourceRevision); err != nil {
			t.Fatal(err)
		}
	}
}
