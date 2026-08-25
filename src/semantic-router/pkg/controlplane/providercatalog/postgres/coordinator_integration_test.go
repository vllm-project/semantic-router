package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroups(t *testing.T) {
	db := providerCatalogIntegrationDatabase(t)
	coordinator, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := New(db, integrationRegistry(t))
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil {
		t.Fatal(testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	first := testSnapshot(t, "provider-one")
	second := testSnapshot(t, "provider-two")
	if _, err := coordinator.Stage(ctx, StageRequest{
		Snapshot: first, ExpectedGeneration: 1,
	}); err == nil {
		t.Fatal("Stage() accepted an implicit empty rollout-group gate")
	}
	if _, err := coordinator.ActiveSnapshot(ctx); !errors.Is(err, ErrNoActiveSnapshot) {
		t.Fatalf("ActiveSnapshot() before activation = %v", err)
	}
	staged, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.Stage(ctx, StageRequest{
		Snapshot: first, ExpectedGeneration: 1, RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	})
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil {
		t.Fatal(testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	if staged.DesiredRevision != first.Revision() || staged.ActiveRevision != "" || staged.Generation != 2 {
		t.Fatalf("staged state = %+v", staged)
	}
	desired, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.DesiredSnapshot(ctx)
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil || desired.Revision() != first.Revision() {
		t.Fatalf("DesiredSnapshot() = %v, %v", desired, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	retried, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.Stage(ctx, StageRequest{
		Snapshot: first, ExpectedGeneration: 1, RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	})
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil || retried.Generation != staged.Generation {
		t.Fatalf("idempotent Stage() = %+v, %v", retried, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	if _, err := coordinator.Stage(ctx, StageRequest{
		Snapshot: second, ExpectedGeneration: 1, RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	}); !errors.Is(err, providercatalog.ErrPublicationConflict) {
		t.Fatalf("stale Stage() = %v", err)
	}
	var secondRows int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM provider_catalog_revisions WHERE revision = $1`, second.Revision()).Scan(&secondRows); err != nil {
		t.Fatal(err)
	}
	if secondRows != 0 {
		t.Fatal("CAS-rejected stage leaked its immutable snapshot")
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(second.Revision(), "router-a", AckCompatible)); !errors.Is(err, ErrStaleRevision) {
		t.Fatalf("stale acknowledgement = %v", err)
	}

	active := assertStableRolloutActivation(t, ctx, coordinator, first, staged)
	assertSecondCatalogStage(t, ctx, db, coordinator, first, second, active)
}

func assertStableRolloutActivation(
	t *testing.T,
	ctx context.Context,
	coordinator *Coordinator,
	first *providercatalog.Snapshot,
	staged State,
) State {
	t.Helper()
	var testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr error
	_, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr = coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	blocked := activationBlockers(t, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	if !reflect.DeepEqual(blocked.Missing, []providercatalog.RolloutGroup{dataRolloutGroup()}) {
		t.Fatalf("missing blockers = %+v", blocked)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgementForGroup(first.Revision(), "management-a", controlRolloutGroup(), AckCompatible)); err != nil {
		t.Fatal(err)
	}
	_, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr = coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	blocked = activationBlockers(t, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	if !reflect.DeepEqual(blocked.Missing, []providercatalog.RolloutGroup{dataRolloutGroup()}) {
		t.Fatalf("an observed instance from a non-required rollout group satisfied the gate: %+v", blocked)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(first.Revision(), "router-b", AckIncompatible)); err != nil {
		t.Fatal(err)
	}
	_, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr = coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	blocked = activationBlockers(t, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	if len(blocked.Incompatible) != 1 || blocked.Incompatible[0].ReplicaID != "router-b" ||
		len(blocked.Missing) != 0 {
		t.Fatalf("incompatible blockers = %+v", blocked)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(first.Revision(), "router-a", AckCompatible)); err != nil {
		t.Fatal(err)
	}
	_, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr = coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	blocked = activationBlockers(t, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	if len(blocked.Incompatible) != 1 || blocked.Incompatible[0].ReplicaID != "router-b" {
		t.Fatalf("a compatible peer hid an incompatible live replica: %+v", blocked)
	}
	divergent := acknowledgement(first.Revision(), "router-b", AckCompatible)
	divergent.CapabilityDigest = bytes.Repeat([]byte{0x6b}, 32)
	if _, err := coordinator.Acknowledge(ctx, divergent); err != nil {
		t.Fatal(err)
	}
	_, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr = coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	blocked = activationBlockers(t, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	if !reflect.DeepEqual(blocked.Divergent, []providercatalog.RolloutGroup{dataRolloutGroup()}) {
		t.Fatalf("mixed live rollout capabilities = %+v", blocked)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(first.Revision(), "router-b", AckCompatible)); err != nil {
		t.Fatal(err)
	}
	active, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.Activate(ctx, ActivateRequest{Revision: first.Revision(), ExpectedGeneration: staged.Generation})
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil || active.ActiveRevision != first.Revision() {
		t.Fatalf("Activate() = %+v, %v", active, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	if _, err := coordinator.Activate(ctx, ActivateRequest{
		Revision: first.Revision(), ExpectedGeneration: staged.Generation + 1,
	}); !errors.Is(err, providercatalog.ErrPublicationConflict) {
		t.Fatalf("idempotent Activate() with the wrong generation = %v", err)
	}
	loaded, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.ActiveSnapshot(ctx)
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil || loaded.Revision() != first.Revision() || !reflect.DeepEqual(loaded.List(), first.List()) {
		t.Fatalf("ActiveSnapshot() = %v, %v", loaded, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}

	return active
}

func assertSecondCatalogStage(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	coordinator *Coordinator,
	first *providercatalog.Snapshot,
	second *providercatalog.Snapshot,
	active State,
) {
	t.Helper()
	stagedSecond, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := coordinator.Stage(ctx, StageRequest{
		Snapshot: second, ExpectedDesiredRevision: first.Revision(), ExpectedGeneration: active.Generation,
		RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	})
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil {
		t.Fatal(testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	restarted, _ := New(db, integrationRegistry(t))
	stillActive, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr := restarted.ActiveSnapshot(ctx)
	if testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr != nil || stillActive.Revision() != first.Revision() || stagedSecond.DesiredRevision != second.Revision() {
		t.Fatalf("startup source fell back or promoted desired implicitly: active=%v staged=%+v err=%v", stillActive, stagedSecond, testCoordinatorStagesAcknowledgesAndActivatesStableRolloutGroupsErr)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(first.Revision(), "router-a", AckCompatible)); err != nil {
		t.Fatalf("active revision acknowledgement while another revision is staged: %v", err)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(second.Revision(), "router-a", AckCompatible)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Activate(ctx, ActivateRequest{
		Revision: second.Revision(), ExpectedGeneration: stagedSecond.Generation,
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosed(t *testing.T) {
	db := providerCatalogIntegrationDatabase(t)
	coordinator, _ := New(db, integrationRegistry(t))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	snapshot := testSnapshot(t, "provider-expiry")
	staged, testCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosedErr := coordinator.Stage(ctx, StageRequest{
		Snapshot: snapshot, ExpectedGeneration: 1, RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	})
	if testCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosedErr != nil {
		t.Fatal(testCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosedErr)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(snapshot.Revision(), "router-a", AckCompatible)); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `UPDATE provider_catalog_replica_acks
SET acknowledged_at = clock_timestamp() - interval '2 minutes',
    lease_expires_at = clock_timestamp() - interval '1 minute'
WHERE revision = $1 AND replica_id = 'router-a'`, snapshot.Revision()); err != nil {
		t.Fatal(err)
	}
	_, testCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosedErr = coordinator.Activate(ctx, ActivateRequest{Revision: snapshot.Revision(), ExpectedGeneration: staged.Generation})
	blocked := activationBlockers(t, testCoordinatorExpiredLeaseAndMissingInstalledAdapterFailClosedErr)
	if !reflect.DeepEqual(blocked.Expired, []providercatalog.RolloutGroup{dataRolloutGroup()}) {
		t.Fatalf("expired blockers = %+v", blocked)
	}
	if _, err := db.ExecContext(ctx, `UPDATE provider_catalog_replica_acks
SET acknowledged_at = clock_timestamp() + interval '1 minute',
    lease_expires_at = clock_timestamp() + interval '2 minutes'
WHERE revision = $1 AND replica_id = 'router-a'`, snapshot.Revision()); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Activate(ctx, ActivateRequest{
		Revision: snapshot.Revision(), ExpectedGeneration: staged.Generation,
	}); !errors.Is(err, ErrCorruptState) {
		t.Fatalf("Activate() over a future acknowledgement = %v", err)
	}
	if _, err := coordinator.Acknowledge(ctx, acknowledgement(snapshot.Revision(), "router-a", AckCompatible)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Activate(ctx, ActivateRequest{Revision: snapshot.Revision(), ExpectedGeneration: staged.Generation}); err != nil {
		t.Fatal(err)
	}
	incompatibleReader, _ := New(db, missingRegistry(t))
	if _, err := incompatibleReader.ActiveSnapshot(ctx); !errors.Is(err, ErrCorruptSnapshot) {
		t.Fatalf("ActiveSnapshot() without required installed adapters = %v", err)
	}
}

func TestCoordinatorConcurrentStageCASHasOneWinner(t *testing.T) {
	db := providerCatalogIntegrationDatabase(t)
	coordinator, _ := New(db, integrationRegistry(t))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	snapshots := []*providercatalog.Snapshot{
		testSnapshot(t, "provider-race-one"),
		testSnapshot(t, "provider-race-two"),
	}
	type outcome struct {
		state State
		err   error
	}
	results := make(chan outcome, len(snapshots))
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index, snapshot := range snapshots {
		wait.Add(1)
		go func(candidate *providercatalog.Snapshot, replicaID string) {
			defer wait.Done()
			<-start
			state, err := coordinator.Stage(ctx, StageRequest{
				Snapshot: candidate, ExpectedGeneration: 1,
				RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
			})
			results <- outcome{state: state, err: err}
		}(snapshot, fmt.Sprintf("router-%d", index))
	}
	close(start)
	wait.Wait()
	close(results)
	successes, conflicts := 0, 0
	for result := range results {
		switch {
		case result.err == nil:
			successes++
		case errors.Is(result.err, providercatalog.ErrPublicationConflict):
			conflicts++
		default:
			t.Fatalf("concurrent Stage() error = %v", result.err)
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("concurrent Stage() outcomes: success=%d conflict=%d", successes, conflicts)
	}
	var stored int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM provider_catalog_revisions`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored != 1 {
		t.Fatalf("CAS loser leaked immutable revision: stored=%d", stored)
	}
}

func TestCoordinatorImmutableConflictAndReadCorruptionFailClosed(t *testing.T) {
	db := providerCatalogIntegrationDatabase(t)
	coordinator, _ := New(db, integrationRegistry(t))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	snapshot := testSnapshot(t, "provider-immutable")
	corrupt := []byte(`{"schema":"corrupt"}`)
	digest := sha256.Sum256(corrupt)
	if _, err := db.ExecContext(ctx, `INSERT INTO provider_catalog_revisions
  (revision, snapshot_bytes, snapshot_digest, integration_references, catalog,
   required_wire_formats, required_credential_adapters, required_discovery_adapters)
VALUES ($1, $2, $3, '[]'::jsonb, '{"providers":[]}'::jsonb,
  '[]'::jsonb, '[]'::jsonb, '[]'::jsonb)`, snapshot.Revision(), corrupt, digest[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Stage(ctx, StageRequest{
		Snapshot: snapshot, ExpectedGeneration: 1, RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	}); !errors.Is(err, providercatalog.ErrPublicationConflict) {
		t.Fatalf("Stage() over conflicting immutable revision = %v", err)
	}
	if _, err := db.ExecContext(ctx, `UPDATE provider_catalog_revisions SET snapshot_bytes = 'changed'
WHERE revision = $1`, snapshot.Revision()); err == nil {
		t.Fatal("immutable provider catalog revision accepted an UPDATE")
	}
	if _, err := db.ExecContext(ctx, `UPDATE provider_catalog_state
SET desired_revision = $1, active_revision = $1, generation = 2`, snapshot.Revision()); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.ActiveSnapshot(ctx); !errors.Is(err, ErrCorruptSnapshot) {
		t.Fatalf("ActiveSnapshot() over corrupt bytes = %v", err)
	}
}

func acknowledgement(revision, replicaID string, status AckStatus) AcknowledgeRequest {
	return acknowledgementForGroup(revision, replicaID, dataRolloutGroup(), status)
}

func acknowledgementForGroup(
	revision string,
	replicaID string,
	group providercatalog.RolloutGroup,
	status AckStatus,
) AcknowledgeRequest {
	reason := ""
	if status == AckIncompatible {
		reason = "required wire format is unavailable"
	}
	return AcknowledgeRequest{
		Revision: revision, ReplicaID: replicaID, RolloutGroup: group,
		CapabilityDigest: bytes.Repeat([]byte{0x5a}, 32),
		Status:           status, Reason: reason, Lease: time.Minute,
	}
}

func dataRolloutGroup() providercatalog.RolloutGroup {
	return providercatalog.RolloutGroup{Plane: providercatalog.CapabilityPlaneData, ID: "router"}
}

func controlRolloutGroup() providercatalog.RolloutGroup {
	return providercatalog.RolloutGroup{Plane: providercatalog.CapabilityPlaneControl, ID: "management"}
}

func activationBlockers(t testing.TB, err error) providercatalog.ActivationBlockers {
	t.Helper()
	var blocked *providercatalog.ActivationBlockedError
	if !errors.As(err, &blocked) {
		t.Fatalf("activation error = %v, want ActivationBlockedError", err)
	}
	return blocked.Blockers
}

func testSnapshot(t testing.TB, providerID string) *providercatalog.Snapshot {
	t.Helper()
	definition := integrationDefinition(providerID, "openai.chat.v1")
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{"openai.chat.v1"}, CredentialAdapterIDs: []string{"bearer"},
		DiscoveryAdapterIDs: []string{"openai.models.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry.Snapshot()
}

func integrationRegistry(t testing.TB) *providercatalog.Registry {
	t.Helper()
	definition := integrationDefinition("provider-registry", "openai.chat.v1")
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{"openai.chat.v1"}, CredentialAdapterIDs: []string{"bearer"},
		DiscoveryAdapterIDs: []string{"openai.models.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func missingRegistry(t testing.TB) *providercatalog.Registry {
	t.Helper()
	definition := integrationDefinition("provider-other", "other.protocol.v1")
	definition.Credential = providercatalog.Credential{Mode: providercatalog.CredentialNone}
	definition.Discovery = nil
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{"other.protocol.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func integrationDefinition(providerID, protocolID string) providercatalog.Definition {
	return providercatalog.Definition{
		ID: providerID, Order: 1,
		Display: providercatalog.Display{
			Name: providerID, Description: "A declarative provider.", Category: "Model APIs",
			Icon: providercatalog.Icon{Source: "lobe", Value: "provider", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "chat", Label: "Chat Completions", Default: true,
			WireFormat: llmprotocol.WireFormat(protocolID),
			Compiler: providercatalog.Compiler{
				AdapterID: providercatalog.StaticBackendCompilerID, Config: map[string]any{"path": "/chat/completions"},
			},
		}},
		Credential:   providercatalog.Credential{Mode: providercatalog.CredentialRequired, AdapterID: "bearer", Label: "API key"},
		Origin:       providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com/v1"},
		Discovery:    &providercatalog.Discovery{AdapterID: "openai.models.v1", Path: "/models"},
		Capabilities: []string{"streaming", "tools"},
	}
}

func providerCatalogIntegrationDatabase(t *testing.T) *sql.DB {
	t.Helper()
	dsn := os.Getenv("PROVIDERCATALOG_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("PROVIDERCATALOG_POSTGRES_DSN is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, providerCatalogIntegrationDatabaseErr := sql.Open("postgres", dsn)
	if providerCatalogIntegrationDatabaseErr != nil {
		t.Fatal(providerCatalogIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schema := "provider_catalog_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedDSN, providerCatalogIntegrationDatabaseErr := catalogDSNWithSearchPath(dsn, schema)
	if providerCatalogIntegrationDatabaseErr != nil {
		t.Fatal(providerCatalogIntegrationDatabaseErr)
	}
	db, providerCatalogIntegrationDatabaseErr := sql.Open("postgres", scopedDSN)
	if providerCatalogIntegrationDatabaseErr != nil {
		t.Fatal(providerCatalogIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return db
}

func catalogDSNWithSearchPath(dsn, schema string) (string, error) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		return "", fmt.Errorf("PostgreSQL DSN must be a URL")
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
