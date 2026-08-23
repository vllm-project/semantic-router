package postgres

import (
	"context"
	"database/sql"
	"errors"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKey(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL Management command test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	first, second := isolatedCommandDatabases(t, ctx, dsn)
	if _, err := first.ExecContext(ctx, `CREATE TABLE management_idempotency (
  scope_kind TEXT NOT NULL CHECK (scope_kind IN ('cluster','namespace')),
  namespace_id UUID,
  principal_id UUID NOT NULL,
  endpoint TEXT NOT NULL,
  hmac_version TEXT NOT NULL,
  idempotency_key_digest BYTEA NOT NULL,
  request_digest BYTEA NOT NULL,
  operation_id UUID,
  resource_type TEXT,
  resource_id TEXT,
  resource_revision BIGINT,
  desired_revision BIGINT,
  response_status INTEGER NOT NULL,
	secret_response_ciphertext BYTEA,
	secret_response_nonce BYTEA,
	response_kek_version TEXT,
	secret_response_expires_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK ((scope_kind = 'cluster' AND namespace_id IS NULL) OR
         (scope_kind = 'namespace' AND namespace_id IS NOT NULL))
)`); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ExecContext(ctx, `CREATE UNIQUE INDEX management_idempotency_cluster_identity_uq
  ON management_idempotency (principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'cluster';
CREATE UNIQUE INDEX management_idempotency_namespace_identity_uq
  ON management_idempotency (namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'namespace'`); err != nil {
		t.Fatal(err)
	}

	command, _ := testCommand(t, []byte(`{"secret":"never-persist"}`), "cross-replica-0123456789")
	firstTx, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := first.BeginTx(ctx, nil)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	if _, replayed, err := Lock(ctx, firstTx, command); err != nil || replayed {
		t.Fatalf("first replica Lock() replayed=%t err=%v", replayed, err)
	}

	type outcome struct {
		result   managementcommand.StoredResult
		replayed bool
		err      error
	}
	started := make(chan struct{})
	finished := make(chan outcome, 1)
	go func() {
		tx, beginErr := second.BeginTx(ctx, nil)
		if beginErr != nil {
			finished <- outcome{err: beginErr}
			return
		}
		close(started)
		result, replayed, lockErr := Lock(ctx, tx, command)
		if lockErr == nil {
			lockErr = tx.Commit()
		} else {
			_ = tx.Rollback()
		}
		finished <- outcome{result: result, replayed: replayed, err: lockErr}
	}()
	<-started
	if err := CompleteResource(ctx, firstTx, command, managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: testResourceID,
		ResourceRevision: 1, ResponseStatus: 201,
	}); err != nil {
		_ = firstTx.Rollback()
		t.Fatal(err)
	}
	if err := firstTx.Commit(); err != nil {
		t.Fatal(err)
	}
	secondOutcome := <-finished
	if secondOutcome.err != nil || !secondOutcome.replayed || secondOutcome.result.Resource == nil ||
		secondOutcome.result.Resource.ResourceRevision != 1 {
		t.Fatalf("second replica outcome = %#v", secondOutcome)
	}

	codec, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("k", 32))},
	})
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	conflict, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := codec.Bind(
		managementcommand.NamespaceCommandScope(testNamespaceID), testPrincipalID, command.Endpoint,
		"cross-replica-0123456789", []byte(`{"secret":"different"}`),
		time.Now().UTC(), time.Now().UTC().Add(time.Hour),
	)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	conflictTx, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := second.BeginTx(ctx, nil)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	if _, _, err := Lock(ctx, conflictTx, conflict); !errors.Is(err, managementcommand.ErrConflict) {
		_ = conflictTx.Rollback()
		t.Fatalf("conflicting command error = %v", err)
	}
	_ = conflictTx.Rollback()

	rollbackCommand, _ := testCommand(t, []byte(`{"name":"rollback"}`), "rollback-key-0123456789")
	rolledBack, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := first.BeginTx(ctx, nil)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	if _, replayed, err := Lock(ctx, rolledBack, rollbackCommand); err != nil || replayed {
		t.Fatalf("rollback winner Lock() replayed=%t err=%v", replayed, err)
	}
	if err := rolledBack.Rollback(); err != nil {
		t.Fatal(err)
	}
	retry, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := second.BeginTx(ctx, nil)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	if _, replayed, err := Lock(ctx, retry, rollbackCommand); err != nil || replayed {
		_ = retry.Rollback()
		t.Fatalf("post-rollback Lock() replayed=%t err=%v", replayed, err)
	}
	_ = retry.Rollback()

	keys := map[string][]byte{
		"command-v1": []byte(strings.Repeat("o", 32)),
		"command-v2": []byte(strings.Repeat("n", 32)),
	}
	oldCodec, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := managementcommand.NewCodec(securitykeyring.Symmetric{ActiveVersion: "command-v1", Keys: keys})
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	newCodec, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := managementcommand.NewCodec(securitykeyring.Symmetric{ActiveVersion: "command-v2", Keys: keys})
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	rotationNow := time.Now().UTC()
	bindRotated := func(codec *managementcommand.Codec) managementcommand.Command {
		command, bindErr := codec.Bind(
			managementcommand.NamespaceCommandScope(testNamespaceID), testPrincipalID, "/management/v1/provider-credentials",
			"cross-version-0123456789", []byte(`{"name":"rotated"}`),
			rotationNow, rotationNow.Add(time.Hour),
		)
		if bindErr != nil {
			t.Fatal(bindErr)
		}
		return command
	}
	oldCommand, newCommand := bindRotated(oldCodec), bindRotated(newCodec)
	if oldCommand.AdvisoryLockKey() != newCommand.AdvisoryLockKey() {
		t.Fatal("cross-version commands did not share a stable advisory identity")
	}
	oldTx, testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr := first.BeginTx(ctx, nil)
	if testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr != nil {
		t.Fatal(testCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKeyErr)
	}
	if _, replayed, err := Lock(ctx, oldTx, oldCommand); err != nil || replayed {
		t.Fatalf("old-version winner Lock() replayed=%t err=%v", replayed, err)
	}
	rotatedFinished := make(chan outcome, 1)
	go func() {
		tx, beginErr := second.BeginTx(ctx, nil)
		if beginErr != nil {
			rotatedFinished <- outcome{err: beginErr}
			return
		}
		result, replayed, lockErr := Lock(ctx, tx, newCommand)
		if lockErr == nil {
			lockErr = tx.Commit()
		} else {
			_ = tx.Rollback()
		}
		rotatedFinished <- outcome{result: result, replayed: replayed, err: lockErr}
	}()
	if err := CompleteResource(ctx, oldTx, oldCommand, managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: testResourceID,
		ResourceRevision: 2, ResponseStatus: 201,
	}); err != nil {
		_ = oldTx.Rollback()
		t.Fatal(err)
	}
	if err := oldTx.Commit(); err != nil {
		t.Fatal(err)
	}
	rotatedOutcome := <-rotatedFinished
	if rotatedOutcome.err != nil || !rotatedOutcome.replayed || rotatedOutcome.result.Resource == nil ||
		rotatedOutcome.result.Resource.ResourceRevision != 2 {
		t.Fatalf("cross-version outcome = %#v", rotatedOutcome)
	}

	if err := ValidateReferencedHMACVersions(ctx, first, newCodec); err != nil {
		t.Fatalf("retained-version readiness = %v", err)
	}
	if _, err := first.ExecContext(ctx, `INSERT INTO management_idempotency
  (scope_kind, namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest,
   request_digest, resource_type, resource_id, resource_revision, response_status, expires_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
		string(managementcommand.ScopeNamespace), testNamespaceID, testPrincipalID, "/management/v1/missing-version", "retired-v0",
		[]byte(strings.Repeat("d", 32)), []byte(strings.Repeat("r", 32)), "provider_credential",
		testResourceID, 3, 201, time.Now().UTC().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReferencedHMACVersions(ctx, first, newCodec); !errors.Is(err, managementcommand.ErrHMACVersionUnavailable) {
		t.Fatalf("missing-version readiness = %v", err)
	}
	if _, err := first.ExecContext(ctx, `UPDATE management_idempotency
SET expires_at = clock_timestamp() - interval '1 second' WHERE hmac_version = 'retired-v0'`); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReferencedHMACVersions(ctx, first, newCodec); err != nil {
		t.Fatalf("post-retention readiness = %v", err)
	}

	var keyDigest, requestDigest string
	if err := first.QueryRowContext(ctx, `SELECT encode(idempotency_key_digest, 'hex'), encode(request_digest, 'hex')
FROM management_idempotency LIMIT 1`).Scan(&keyDigest, &requestDigest); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(keyDigest, "cross-replica") || strings.Contains(requestDigest, "never-persist") {
		t.Fatal("stored command exposed raw idempotency or request material")
	}
}

func isolatedCommandDatabases(t *testing.T, ctx context.Context, dsn string) (*sql.DB, *sql.DB) {
	t.Helper()
	admin, isolatedCommandDatabasesErr := sql.Open("postgres", dsn)
	if isolatedCommandDatabasesErr != nil {
		t.Fatal(isolatedCommandDatabasesErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_command_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scoped, isolatedCommandDatabasesErr := commandDSNWithSearchPath(dsn, schema)
	if isolatedCommandDatabasesErr != nil {
		t.Fatal(isolatedCommandDatabasesErr)
	}
	open := func() *sql.DB {
		db, openErr := sql.Open("postgres", scoped)
		if openErr != nil {
			t.Fatal(openErr)
		}
		t.Cleanup(func() { _ = db.Close() })
		return db
	}
	return open(), open()
}

func commandDSNWithSearchPath(dsn, schema string) (string, error) {
	if !strings.Contains(dsn, "://") {
		return dsn + " search_path=" + schema, nil
	}
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
