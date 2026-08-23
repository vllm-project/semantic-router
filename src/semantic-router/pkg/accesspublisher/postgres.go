package accesspublisher

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/lib/pq"
)

type PostgresStoreOptions struct {
	Projector string
}

type PostgresStore struct {
	db        *sql.DB
	projector string
}

func NewPostgresStore(db *sql.DB, options PostgresStoreOptions) (*PostgresStore, error) {
	if db == nil {
		return nil, fmt.Errorf("PostgreSQL publication database is required")
	}
	if err := validateWorker(options.Projector); err != nil {
		return nil, fmt.Errorf("projector: %w", err)
	}
	return &PostgresStore{db: db, projector: options.Projector}, nil
}

func (s *PostgresStore) ClaimLatest(ctx context.Context, workerID string, lease time.Duration) (OutboxBatch, error) {
	if err := validateWorker(workerID); err != nil {
		return OutboxBatch{}, err
	}
	if lease < time.Second || lease > time.Hour {
		return OutboxBatch{}, fmt.Errorf("publication lease must be between one second and one hour")
	}
	tx, claimLatestErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if claimLatestErr != nil {
		return OutboxBatch{}, fmt.Errorf("begin outbox claim: %w", claimLatestErr)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.ExecContext(ctx, `UPDATE policy_outbox
SET state = 'pending', locked_by = NULL, locked_at = NULL, available_at = clock_timestamp()
WHERE state = 'processing' AND locked_at < clock_timestamp() - ($1 * interval '1 millisecond')`, lease.Milliseconds()); err != nil {
		return OutboxBatch{}, fmt.Errorf("reclaim expired outbox leases: %w", err)
	}
	var seedID, namespaceID string
	claimLatestErr = tx.QueryRowContext(ctx, `SELECT o.id, o.namespace_id
FROM policy_outbox o
WHERE ((o.state = 'pending' AND o.available_at <= clock_timestamp())
       OR (o.state = 'processing' AND o.locked_by = $1))
  AND NOT EXISTS (
    SELECT 1 FROM policy_outbox busy
    WHERE busy.namespace_id = o.namespace_id AND busy.state = 'processing'
      AND busy.locked_by <> $1
  )
ORDER BY o.created_at, o.id
FOR UPDATE OF o SKIP LOCKED
LIMIT 1`, workerID).Scan(&seedID, &namespaceID)
	if errors.Is(claimLatestErr, sql.ErrNoRows) {
		return OutboxBatch{}, ErrNoWork
	}
	if claimLatestErr != nil {
		return OutboxBatch{}, fmt.Errorf("select outbox namespace: %w", claimLatestErr)
	}
	_ = seedID

	var runtimeEpoch int64
	var partition string
	if err := tx.QueryRowContext(ctx, `SELECT runtime_epoch, quota_partition_id
FROM access_namespaces WHERE id = $1 FOR UPDATE`, namespaceID).Scan(&runtimeEpoch, &partition); err != nil {
		return OutboxBatch{}, fmt.Errorf("lock publication namespace: %w", err)
	}
	var desiredRevision int64
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(desired_revision), 0)
FROM policy_outbox
WHERE namespace_id = $1 AND (state = 'pending' OR (state = 'processing' AND locked_by = $2))`,
		namespaceID, workerID,
	).Scan(&desiredRevision); err != nil {
		return OutboxBatch{}, fmt.Errorf("read namespace desired revision: %w", err)
	}
	if desiredRevision <= 0 || runtimeEpoch <= 0 {
		return OutboxBatch{}, fmt.Errorf("namespace desired revision or runtime epoch is invalid")
	}
	rows, claimLatestErr := tx.QueryContext(ctx, `UPDATE policy_outbox
SET state = 'processing', attempt_count = CASE WHEN locked_by = $2 THEN attempt_count ELSE attempt_count + 1 END,
    locked_by = $2, locked_at = clock_timestamp(), last_error = NULL
WHERE namespace_id = $1 AND desired_revision <= $3
  AND (state IN ('pending','failed') OR (state = 'processing' AND locked_by = $2))
RETURNING id`, namespaceID, workerID, desiredRevision)
	if claimLatestErr != nil {
		return OutboxBatch{}, fmt.Errorf("claim namespace outbox rows: %w", claimLatestErr)
	}
	rowIDs := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return OutboxBatch{}, errors.Join(
				fmt.Errorf("scan claimed outbox row: %w", err),
				rows.Close(),
			)
		}
		rowIDs = append(rowIDs, id)
	}
	if err := rows.Close(); err != nil {
		return OutboxBatch{}, fmt.Errorf("close claimed outbox rows: %w", err)
	}
	if len(rowIDs) == 0 {
		return OutboxBatch{}, ErrNoWork
	}
	sort.Strings(rowIDs)
	var claimedAt time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&claimedAt); err != nil {
		return OutboxBatch{}, fmt.Errorf("read claim timestamp: %w", err)
	}
	batch := OutboxBatch{
		NamespaceID: namespaceID, DesiredRevision: uint64(desiredRevision), RuntimeEpoch: uint64(runtimeEpoch),
		QuotaPartition: partition, RowIDs: rowIDs, WorkerID: workerID, ClaimedAt: claimedAt.UTC(),
	}
	if err := batch.Validate(); err != nil {
		return OutboxBatch{}, err
	}
	if err := tx.Commit(); err != nil {
		return OutboxBatch{}, fmt.Errorf("commit outbox claim: %w", err)
	}
	return batch, nil
}

func (s *PostgresStore) RecordStaged(ctx context.Context, batch OutboxBatch, publication Publication) error {
	if err := validateBatchPublication(batch, publication); err != nil {
		return err
	}
	payload, err := json.Marshal(publication.Routing.Snapshot)
	if err != nil {
		return fmt.Errorf("encode routing snapshot: %w", err)
	}
	digest, err := hex.DecodeString(publication.Routing.Snapshot.Digest)
	if err != nil || len(digest) != 32 {
		return fmt.Errorf("routing snapshot digest is invalid")
	}
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return fmt.Errorf("begin routing snapshot staging: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	var storedDigest, storedPayload []byte
	err = tx.QueryRowContext(ctx, `INSERT INTO routing_snapshots
  (namespace_id, routing_revision, content_digest, compiled_blob, status)
VALUES ($1, $2, $3, $4, 'staged')
ON CONFLICT (namespace_id, routing_revision) DO UPDATE
SET compiled_blob = CASE WHEN routing_snapshots.content_digest = EXCLUDED.content_digest
                         THEN routing_snapshots.compiled_blob ELSE routing_snapshots.compiled_blob END
RETURNING content_digest, compiled_blob`, batch.NamespaceID, int64(batch.DesiredRevision), digest, payload).Scan(
		&storedDigest, &storedPayload,
	)
	if err != nil {
		return fmt.Errorf("stage PostgreSQL routing snapshot: %w", err)
	}
	if !equalBytes(storedDigest, digest) || !equalBytes(storedPayload, payload) {
		return fmt.Errorf("%w: routing revision already has another immutable snapshot", ErrConflict)
	}
	if err := insertRoutingMembers(ctx, tx, publication); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit routing snapshot staging: %w", err)
	}
	return nil
}

func insertRoutingMembers(ctx context.Context, tx *sql.Tx, publication Publication) error {
	for _, model := range publication.Routing.Snapshot.Models {
		if err := insertRoutingMember(ctx, tx, publication, "model", model.ID, model.Revision); err != nil {
			return err
		}
	}
	for _, recipe := range publication.Routing.Snapshot.Recipes {
		if err := insertRoutingMember(ctx, tx, publication, "recipe", recipe.ID, recipe.Revision); err != nil {
			return err
		}
	}
	for _, entrypoint := range publication.Routing.Snapshot.Entrypoints {
		if err := insertRoutingMember(ctx, tx, publication, "entrypoint", entrypoint.ID, entrypoint.Revision); err != nil {
			return err
		}
	}
	return nil
}

func insertRoutingMember(
	ctx context.Context,
	tx *sql.Tx,
	publication Publication,
	kind, resourceID string,
	revision int64,
) error {
	var storedRevision int64
	err := tx.QueryRowContext(ctx, `INSERT INTO routing_snapshot_members
  (namespace_id, routing_revision, resource_type, resource_id, resource_revision)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (namespace_id, routing_revision, resource_type, resource_id) DO UPDATE
SET resource_revision = routing_snapshot_members.resource_revision
RETURNING resource_revision`, publication.NamespaceID, int64(publication.DesiredRevision), kind, resourceID, revision).Scan(
		&storedRevision,
	)
	if err != nil {
		return fmt.Errorf("stage routing snapshot member %s/%s: %w", kind, resourceID, err)
	}
	if storedRevision != revision {
		return fmt.Errorf("%w: routing snapshot member %s/%s revision changed", ErrConflict, kind, resourceID)
	}
	return nil
}

func (s *PostgresStore) Release(ctx context.Context, batch OutboxBatch, cause error, delay time.Duration) error {
	if err := batch.Validate(); err != nil {
		return err
	}
	if delay < 0 || delay > time.Hour {
		return fmt.Errorf("outbox retry delay is invalid")
	}
	result, err := s.db.ExecContext(ctx, `UPDATE policy_outbox
SET state = 'pending', available_at = clock_timestamp() + ($3 * interval '1 millisecond'),
    locked_by = NULL, locked_at = NULL, last_error = $4
WHERE id = ANY($1) AND state = 'processing' AND locked_by = $2`,
		pq.Array(batch.RowIDs), batch.WorkerID, delay.Milliseconds(), safeFailureCode(cause))
	if err != nil {
		return fmt.Errorf("release outbox claim: %w", err)
	}
	return requireRowsAffected(result, len(batch.RowIDs), "release outbox claim")
}

func (s *PostgresStore) Fail(ctx context.Context, batch OutboxBatch, cause error) error {
	if err := batch.Validate(); err != nil {
		return err
	}
	tx, failErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if failErr != nil {
		return fmt.Errorf("begin outbox failure: %w", failErr)
	}
	defer func() { _ = tx.Rollback() }()
	code := safeFailureCode(cause)
	result, failErr := tx.ExecContext(ctx, `UPDATE policy_outbox
SET state = 'failed', locked_by = NULL, locked_at = NULL, last_error = $3
WHERE id = ANY($1) AND state = 'processing' AND locked_by = $2`, pq.Array(batch.RowIDs), batch.WorkerID, code)
	if failErr != nil {
		return fmt.Errorf("fail outbox claim: %w", failErr)
	}
	if err := requireRowsAffected(result, len(batch.RowIDs), "fail outbox claim"); err != nil {
		return err
	}
	_, failErr = tx.ExecContext(ctx, `INSERT INTO projector_watermarks
  (projector, namespace_id, desired_revision, applied_revision, runtime_epoch, last_error)
VALUES ($1, $2, $3, 0, $4, $5)
ON CONFLICT (projector, namespace_id) DO UPDATE
SET desired_revision = GREATEST(projector_watermarks.desired_revision, EXCLUDED.desired_revision),
    last_error = EXCLUDED.last_error, updated_at = clock_timestamp()`,
		s.projector, batch.NamespaceID, int64(batch.DesiredRevision), int64(batch.RuntimeEpoch), code)
	if failErr != nil {
		return fmt.Errorf("record projector failure: %w", failErr)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE management_operations
SET state='failed', item_errors=jsonb_build_array(jsonb_build_object(
      'code','publication_failed','reason','Routing publication failed.')),
    updated_at=clock_timestamp()
WHERE namespace_id=$1 AND desired_revision=$2 AND state IN ('pending','running')`,
		batch.NamespaceID, int64(batch.DesiredRevision)); err != nil {
		return fmt.Errorf("fail publication operations: %w", err)
	}
	return tx.Commit()
}

func (s *PostgresStore) WithRevisionFence(
	ctx context.Context,
	batch OutboxBatch,
	activate func(context.Context) error,
) error {
	if err := batch.Validate(); err != nil {
		return err
	}
	if activate == nil {
		return fmt.Errorf("activation callback is required")
	}
	tx, withRevisionFenceErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if withRevisionFenceErr != nil {
		return fmt.Errorf("begin publication revision fence: %w", withRevisionFenceErr)
	}
	defer func() { _ = tx.Rollback() }()
	var epoch int64
	var partition string
	if err := tx.QueryRowContext(ctx, `SELECT runtime_epoch, quota_partition_id
FROM access_namespaces WHERE id = $1 FOR UPDATE`, batch.NamespaceID).Scan(&epoch, &partition); err != nil {
		return fmt.Errorf("lock publication revision fence: %w", err)
	}
	if uint64(epoch) != batch.RuntimeEpoch || partition != batch.QuotaPartition {
		return ErrEpochMismatch
	}
	var latest int64
	if err := tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(revision), 0) FROM policy_revisions WHERE namespace_id = $1`, batch.NamespaceID,
	).Scan(&latest); err != nil {
		return fmt.Errorf("verify publication desired revision: %w", err)
	}
	if latest != int64(batch.DesiredRevision) {
		return ErrSuperseded
	}
	var owned int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox
WHERE id = ANY($1) AND state = 'processing' AND locked_by = $2`, pq.Array(batch.RowIDs), batch.WorkerID).Scan(&owned); err != nil {
		return fmt.Errorf("verify publication claim ownership: %w", err)
	}
	if owned != len(batch.RowIDs) {
		return ErrConflict
	}
	if err := activate(ctx); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE routing_snapshots SET status = 'retired'
WHERE namespace_id = $1 AND status = 'active' AND routing_revision <> $2`, batch.NamespaceID, latest); err != nil {
		return fmt.Errorf("retire prior routing snapshot: %w", err)
	}
	result, withRevisionFenceErr := tx.ExecContext(ctx, `UPDATE routing_snapshots
SET status = 'active', activated_at = COALESCE(activated_at, clock_timestamp()), failure_reason = NULL
WHERE namespace_id = $1 AND routing_revision = $2 AND status IN ('staged','active')`, batch.NamespaceID, latest)
	if withRevisionFenceErr != nil {
		return fmt.Errorf("activate PostgreSQL routing snapshot: %w", withRevisionFenceErr)
	}
	if err := requireRowsAffected(result, 1, "activate PostgreSQL routing snapshot"); err != nil {
		return err
	}
	result, withRevisionFenceErr = tx.ExecContext(ctx, `UPDATE policy_outbox
SET state = 'applied', applied_at = COALESCE(applied_at, clock_timestamp()),
    locked_by = NULL, locked_at = NULL, last_error = NULL
WHERE id = ANY($1) AND state = 'processing' AND locked_by = $2`, pq.Array(batch.RowIDs), batch.WorkerID)
	if withRevisionFenceErr != nil {
		return fmt.Errorf("apply outbox rows: %w", withRevisionFenceErr)
	}
	if err := requireRowsAffected(result, len(batch.RowIDs), "apply outbox rows"); err != nil {
		return err
	}
	var remaining int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox
WHERE namespace_id = $1 AND desired_revision <= $2 AND state <> 'applied'`, batch.NamespaceID, latest).Scan(&remaining); err != nil {
		return fmt.Errorf("verify contiguous publication watermark: %w", err)
	}
	if remaining != 0 {
		return fmt.Errorf("%w: lower outbox revision is not applied", ErrConflict)
	}
	_, withRevisionFenceErr = tx.ExecContext(ctx, `INSERT INTO projector_watermarks
  (projector, namespace_id, desired_revision, applied_revision, runtime_epoch, last_error)
VALUES ($1, $2, $3, $3, $4, NULL)
ON CONFLICT (projector, namespace_id) DO UPDATE
SET desired_revision = EXCLUDED.desired_revision, applied_revision = EXCLUDED.applied_revision,
    runtime_epoch = EXCLUDED.runtime_epoch, last_error = NULL, updated_at = clock_timestamp()
WHERE projector_watermarks.applied_revision <= EXCLUDED.applied_revision
  AND projector_watermarks.runtime_epoch = EXCLUDED.runtime_epoch`,
		s.projector, batch.NamespaceID, latest, epoch)
	if withRevisionFenceErr != nil {
		return fmt.Errorf("advance projector watermark: %w", withRevisionFenceErr)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE management_operations
SET state='succeeded', progress_completed=progress_total,
    publication_revision=$2, applied_revision=$2, updated_at=clock_timestamp()
WHERE namespace_id=$1 AND desired_revision <= $2 AND state IN ('pending','running')`,
		batch.NamespaceID, latest); err != nil {
		return fmt.Errorf("complete publication operations: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit publication revision fence: %w", err)
	}
	return nil
}

func (s *PostgresStore) Applied(ctx context.Context, namespaceID string) (AppliedState, error) {
	var state AppliedState
	state.NamespaceID = namespaceID
	var epoch, desired, applied int64
	err := s.db.QueryRowContext(ctx, `SELECT w.runtime_epoch, w.desired_revision, w.applied_revision, n.quota_partition_id
FROM projector_watermarks w JOIN access_namespaces n ON n.id = w.namespace_id
WHERE w.projector = $1 AND w.namespace_id = $2`, s.projector, namespaceID).Scan(
		&epoch, &desired, &applied, &state.QuotaPartition,
	)
	if err == sql.ErrNoRows {
		return AppliedState{}, ErrNoWork
	}
	if err != nil {
		return AppliedState{}, fmt.Errorf("read projector watermark: %w", err)
	}
	if epoch <= 0 || desired < 0 || applied < 0 {
		return AppliedState{}, fmt.Errorf("projector watermark is invalid")
	}
	state.RuntimeEpoch, state.DesiredRevision = uint64(epoch), uint64(applied)
	var digest []byte
	err = s.db.QueryRowContext(ctx, `SELECT content_digest FROM routing_snapshots
WHERE namespace_id = $1 AND routing_revision = $2 AND status = 'active'`, namespaceID, applied).Scan(&digest)
	if err != nil && err != sql.ErrNoRows {
		return AppliedState{}, fmt.Errorf("read active routing snapshot: %w", err)
	}
	if err == nil {
		state.RoutingDigest = hex.EncodeToString(digest)
	}
	return state, nil
}

func validateBatchPublication(batch OutboxBatch, publication Publication) error {
	if err := batch.Validate(); err != nil {
		return err
	}
	if err := publication.Validate(); err != nil {
		return err
	}
	if batch.NamespaceID != publication.NamespaceID || batch.QuotaPartition != publication.QuotaPartition ||
		batch.DesiredRevision != publication.DesiredRevision || batch.RuntimeEpoch != publication.RuntimeEpoch {
		return fmt.Errorf("outbox batch and publication identity disagree")
	}
	return nil
}

func requireRowsAffected(result sql.Result, expected int, operation string) error {
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("%s row count: %w", operation, err)
	}
	if affected != int64(expected) {
		return fmt.Errorf("%w: %s affected %d rows, want %d", ErrConflict, operation, affected, expected)
	}
	return nil
}

func validateWorker(value string) error {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 {
		return fmt.Errorf("worker identifier is required and bounded")
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return fmt.Errorf("worker identifier contains a control character")
		}
	}
	return nil
}

func safeFailureCode(err error) string {
	switch {
	case err == nil:
		return "publication_retry"
	case errors.Is(err, ErrSuperseded):
		return "desired_revision_superseded"
	case errors.Is(err, ErrEpochMismatch):
		return "runtime_epoch_mismatch"
	case errors.Is(err, ErrAcknowledgements):
		return "replica_acknowledgements_incomplete"
	case errors.Is(err, ErrStagedCorrupt):
		return "staged_projection_invalid"
	case errors.Is(err, ErrConflict):
		return "publication_cas_conflict"
	default:
		return "publication_failed"
	}
}
