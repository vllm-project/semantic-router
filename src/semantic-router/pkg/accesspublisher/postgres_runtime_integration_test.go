package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestPostgresRuntimeDiscoversActiveNamespaceBeforeFirstPublication(t *testing.T) {
	database := postgresIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	activeNamespace := uuid.NewString()
	activePartition := "partition-" + uuid.NewString()
	disabledNamespace := uuid.NewString()
	if _, err := database.ExecContext(ctx, `INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch)
VALUES ($1, $2, $3, 'USD', 'active', 1, 1),
       ($4, $5, $6, 'USD', 'disabled', 1, 1)`,
		activeNamespace, "active-"+activeNamespace, activePartition,
		disabledNamespace, "disabled-"+disabledNamespace, "partition-"+uuid.NewString()); err != nil {
		t.Fatal(err)
	}

	store, err := NewPostgresStore(database, PostgresStoreOptions{Projector: "routing-publication-unpublished-it"})
	if err != nil {
		t.Fatal(err)
	}
	references, err := store.ListPublicationNamespaces(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(references) != 1 || references[0].NamespaceID != activeNamespace ||
		references[0].QuotaPartition != activePartition {
		t.Fatalf("publication namespaces before first publication = %+v", references)
	}
	heads, err := store.ReadPublicationHeads(ctx, references[0])
	if err != nil {
		t.Fatal(err)
	}
	if heads.Active != nil || heads.Candidate != nil {
		t.Fatalf("unpublished namespace heads = %+v", heads)
	}
	readiness, err := store.Readiness(ctx, activeNamespace, activePartition)
	if err != nil {
		t.Fatal(err)
	}
	if readiness.Ready || readiness.Reason != "publication_gate_unpublished" {
		t.Fatalf("unpublished namespace readiness = %+v", readiness)
	}
}

func TestPostgresRuntimePublicationSurvivesRestartAndRequiresEveryReplica(t *testing.T) {
	database := postgresIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	namespaceID := uuid.NewString()
	partition := "partition-" + uuid.NewString()
	const epoch = uint64(7)
	if _, err := database.ExecContext(ctx, `INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch)
VALUES ($1, $2, $3, 'USD', 'active', 1, $4)`,
		namespaceID, "namespace-"+namespaceID, partition, int64(epoch)); err != nil {
		t.Fatal(err)
	}

	store, err := NewPostgresStore(database, PostgresStoreOptions{
		Projector: "routing-publication-runtime-it", ReplicaLease: 30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, replicaID := range []string{"replica-a", "replica-b"} {
		if _, registerErr := store.RegisterFleetReplica(ctx, replicaID); registerErr != nil {
			t.Fatal(registerErr)
		}
	}

	first := publishPostgresRoutingRevision(t, ctx, database, store, namespaceID, partition, 1, epoch)
	second := publishPostgresRoutingRevision(t, ctx, database, store, namespaceID, partition, 2, epoch)

	readiness, err := store.Readiness(ctx, namespaceID, partition)
	if err != nil {
		t.Fatal(err)
	}
	if !readiness.Ready || readiness.DesiredRevision != 2 || readiness.AppliedRevision != 2 ||
		readiness.RoutingGate != second.ID {
		t.Fatalf("PostgreSQL publication readiness = %+v", readiness)
	}

	restarted, err := NewPostgresStore(database, PostgresStoreOptions{
		Projector: "routing-publication-runtime-it", ReplicaLease: 30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	references, err := restarted.ListPublicationNamespaces(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(references) != 1 || references[0].NamespaceID != namespaceID || references[0].QuotaPartition != partition {
		t.Fatalf("publication namespaces after restart = %+v", references)
	}
	heads, err := restarted.ReadPublicationHeads(ctx, references[0])
	if err != nil {
		t.Fatal(err)
	}
	if heads.Active == nil || !heads.Active.Activated() || heads.Active.PublicationID != second.ID || heads.Candidate != nil {
		t.Fatalf("publication heads after restart = %+v", heads)
	}
	loaded, err := restarted.LoadRoutingPublication(ctx, *heads.Active)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Identity.PublicationID != second.ID || loaded.Snapshot.Digest != second.Routing.Snapshot.Digest {
		t.Fatalf("loaded publication after restart = %+v", loaded.Identity)
	}
	for _, replicaID := range []string{"replica-a", "replica-b"} {
		if _, err := restarted.RegisterReplica(ctx, namespaceID, partition, ReplicaRegistration{
			ReplicaID: replicaID, RuntimeEpoch: epoch,
			AccessPublication: second.ID, RoutingPublication: second.ID,
		}); err != nil {
			t.Fatalf("register restarted %s: %v", replicaID, err)
		}
	}

	if _, err := restarted.Prepare(ctx, first); !errors.Is(err, ErrSuperseded) {
		t.Fatalf("older revision Prepare() = %v, want ErrSuperseded", err)
	}
}

func publishPostgresRoutingRevision(
	t testing.TB,
	ctx context.Context,
	database *sql.DB,
	store *PostgresStore,
	namespaceID, partition string,
	revision, epoch uint64,
) Publication {
	t.Helper()
	insertRevision(t, ctx, database, namespaceID, fixturePostgresBigint(revision), fixturePostgresBigint(epoch))
	insertOutbox(t, ctx, database, uuid.NewString(), namespaceID, fixturePostgresBigint(revision))
	batch, err := store.ClaimLatest(ctx, "runtime-publication-worker", 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	publication := postgresPublication(t, namespaceID, partition, revision, epoch)
	if stagedErr := store.RecordStaged(ctx, batch, publication); stagedErr != nil {
		t.Fatal(stagedErr)
	}
	plan, err := store.Prepare(ctx, publication)
	if err != nil {
		t.Fatal(err)
	}
	if stageErr := store.Stage(ctx, plan); stageErr != nil {
		t.Fatal(stageErr)
	}
	if validationErr := store.ValidateStaged(ctx, plan); validationErr != nil {
		t.Fatal(validationErr)
	}
	status, err := store.RoutingAcknowledgements(ctx, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(status.Required) != 2 || len(status.Missing) != 2 {
		t.Fatalf("initial routing acknowledgements = %+v", status)
	}
	for _, replicaID := range []string{"replica-a", "replica-b"} {
		if err := store.AcknowledgeRouting(
			ctx, namespaceID, partition, replicaID, publication.ID, publication.Digest,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.WithRevisionFence(ctx, batch, func(fenced context.Context) error {
		status, err := store.RoutingAcknowledgements(fenced, plan)
		if err != nil {
			return err
		}
		if !status.Complete() {
			return ErrAcknowledgements
		}
		return store.Activate(fenced, plan)
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkApplied(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAppliedBarriers(ctx, plan); err != nil {
		t.Fatal(err)
	}
	return publication
}
