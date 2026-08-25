package accesspublisher

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/lib/pq"
)

const postgresPublicationChannel = "vllm_sr_routing_publication"

type postgresRevisionFenceKey struct{}

type postgresExecutor interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func (s *PostgresStore) executor(ctx context.Context) postgresExecutor {
	if tx, ok := ctx.Value(postgresRevisionFenceKey{}).(*sql.Tx); ok && tx != nil {
		return tx
	}
	return s.db
}

// Close releases only the optional LISTEN connection. The process composition
// owns and closes the shared database pool separately.
func (s *PostgresStore) Close() error {
	if s == nil || s.listener == nil {
		return nil
	}
	return s.listener.Close()
}

// PublicationNotifications is an acceleration signal. Callers must retain
// their periodic reconciliation because PostgreSQL notifications are not a
// durable queue and may be lost while a replica is disconnected.
func (s *PostgresStore) PublicationNotifications(ctx context.Context) <-chan struct{} {
	wake := make(chan struct{}, 1)
	if s == nil || s.listener == nil {
		return nil
	}
	go func() {
		defer close(wake)
		for {
			select {
			case <-ctx.Done():
				return
			case notification, ok := <-s.listener.Notify:
				if !ok {
					return
				}
				if notification == nil {
					_ = s.listener.Ping()
					continue
				}
				select {
				case wake <- struct{}{}:
				default:
				}
			}
		}
	}()
	return wake
}

func notifyPostgresPublication(ctx context.Context, executor postgresExecutor, namespaceID string) error {
	if _, err := executor.ExecContext(
		ctx,
		`SELECT pg_notify($1, $2)`,
		postgresPublicationChannel,
		namespaceID,
	); err != nil {
		return fmt.Errorf("notify routing publication replicas: %w", err)
	}
	return nil
}

// Prepare records one immutable PostgreSQL publication candidate. In the
// routing-only topology PostgreSQL owns both desired state and the publication
// gate, so no access barrier or hot-store pointer is synthesized.
func (s *PostgresStore) Prepare(ctx context.Context, publication Publication) (PublicationPlan, error) {
	if err := publication.Validate(); err != nil {
		return PublicationPlan{}, err
	}
	if err := verifyPublication(publication); err != nil {
		return PublicationPlan{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	payload, err := json.Marshal(publication)
	if err != nil {
		return PublicationPlan{}, fmt.Errorf("encode PostgreSQL routing publication: %w", err)
	}
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return PublicationPlan{}, fmt.Errorf("begin PostgreSQL publication prepare: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	activeID, err := preparePostgresCandidate(ctx, tx, publication, payload)
	if err != nil {
		return PublicationPlan{}, err
	}
	previous, err := loadPreviousPostgresManifest(ctx, tx, publication.NamespaceID, activeID)
	if err != nil {
		return PublicationPlan{}, err
	}
	if err := notifyPostgresPublication(ctx, tx, publication.NamespaceID); err != nil {
		return PublicationPlan{}, err
	}
	if err := tx.Commit(); err != nil {
		return PublicationPlan{}, fmt.Errorf("commit PostgreSQL publication prepare: %w", err)
	}
	prior := ""
	if activeID.Valid {
		prior = activeID.String
	}
	return PublicationPlan{
		Publication:      publication,
		Previous:         previous,
		PriorAccessGate:  prior,
		PriorRoutingGate: prior,
	}, nil
}

func preparePostgresCandidate(
	ctx context.Context,
	tx *sql.Tx,
	publication Publication,
	payload []byte,
) (sql.NullString, error) {
	desiredRevision, err := postgresBigint(publication.DesiredRevision, "desired revision")
	if err != nil {
		return sql.NullString{}, err
	}
	runtimeEpoch, err := postgresBigint(publication.RuntimeEpoch, "runtime epoch")
	if err != nil {
		return sql.NullString{}, err
	}
	activeID, err := lockPostgresPublicationHead(ctx, tx, publication)
	if err != nil {
		return sql.NullString{}, err
	}
	result, err := tx.ExecContext(ctx, `INSERT INTO routing_publications
  (namespace_id, desired_revision, publication_id, quota_partition_id, runtime_epoch,
   publication_digest, manifest_digest, routing_digest, state, restrictive, publication_blob)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'prepared',FALSE,$9)
ON CONFLICT (namespace_id, desired_revision) DO NOTHING`,
		publication.NamespaceID, desiredRevision, publication.ID,
		publication.QuotaPartition, runtimeEpoch, publication.Digest,
		publication.Manifest.Digest, publication.Routing.Digest, payload,
	)
	if err != nil {
		return sql.NullString{}, fmt.Errorf("record PostgreSQL publication candidate: %w", err)
	}
	inserted, _ := result.RowsAffected()
	if inserted == 0 {
		var storedID, storedDigest string
		var storedPayload []byte
		if readErr := tx.QueryRowContext(ctx, `SELECT publication_id, publication_digest, publication_blob
	FROM routing_publications WHERE namespace_id=$1 AND desired_revision=$2`,
			publication.NamespaceID, desiredRevision,
		).Scan(&storedID, &storedDigest, &storedPayload); readErr != nil {
			return sql.NullString{}, fmt.Errorf("read existing PostgreSQL publication candidate: %w", readErr)
		}
		if storedID != publication.ID || storedDigest != publication.Digest || !reflect.DeepEqual(storedPayload, payload) {
			return sql.NullString{}, fmt.Errorf("%w: PostgreSQL publication candidate changed", ErrConflict)
		}
	}
	result, err = tx.ExecContext(ctx, `UPDATE routing_publication_heads
SET candidate_publication_id=$2, candidate_revision=$3, updated_at=clock_timestamp()
WHERE namespace_id=$1
  AND (active_revision IS NULL OR active_revision <= $3)
  AND (candidate_revision IS NULL OR candidate_revision <= $3)`,
		publication.NamespaceID, publication.ID, desiredRevision)
	if err != nil {
		return sql.NullString{}, fmt.Errorf("advance PostgreSQL publication candidate: %w", err)
	}
	if err := requireRowsAffected(result, 1, "advance PostgreSQL publication candidate"); err != nil {
		return sql.NullString{}, err
	}
	return activeID, nil
}

func lockPostgresPublicationHead(
	ctx context.Context,
	tx *sql.Tx,
	publication Publication,
) (sql.NullString, error) {
	runtimeEpoch, err := postgresBigint(publication.RuntimeEpoch, "runtime epoch")
	if err != nil {
		return sql.NullString{}, err
	}
	desiredRevision, err := postgresBigint(publication.DesiredRevision, "desired revision")
	if err != nil {
		return sql.NullString{}, err
	}
	var epoch int64
	var partition string
	if err := tx.QueryRowContext(ctx, `SELECT runtime_epoch, quota_partition_id
FROM access_namespaces WHERE id=$1 FOR UPDATE`, publication.NamespaceID).Scan(&epoch, &partition); err != nil {
		return sql.NullString{}, fmt.Errorf("lock PostgreSQL publication namespace: %w", err)
	}
	if epoch != runtimeEpoch || partition != publication.QuotaPartition {
		return sql.NullString{}, ErrEpochMismatch
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO routing_publication_heads
  (namespace_id, quota_partition_id) VALUES ($1,$2)
ON CONFLICT (namespace_id) DO NOTHING`, publication.NamespaceID, publication.QuotaPartition); err != nil {
		return sql.NullString{}, fmt.Errorf("initialize PostgreSQL publication head: %w", err)
	}
	var activeID, candidateID sql.NullString
	var activeRevision, candidateRevision sql.NullInt64
	if err := tx.QueryRowContext(ctx, `SELECT active_publication_id, active_revision,
       candidate_publication_id, candidate_revision
FROM routing_publication_heads WHERE namespace_id=$1 FOR UPDATE`, publication.NamespaceID).Scan(
		&activeID, &activeRevision, &candidateID, &candidateRevision,
	); err != nil {
		return sql.NullString{}, fmt.Errorf("lock PostgreSQL publication head: %w", err)
	}
	if activeRevision.Valid && activeRevision.Int64 > desiredRevision {
		return sql.NullString{}, ErrSuperseded
	}
	if activeRevision.Valid && activeRevision.Int64 == desiredRevision && activeID.String != publication.ID {
		return sql.NullString{}, ErrConflict
	}
	if candidateRevision.Valid && candidateRevision.Int64 > desiredRevision {
		return sql.NullString{}, ErrSuperseded
	}
	return activeID, nil
}

func loadPreviousPostgresManifest(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	activeID sql.NullString,
) (*Manifest, error) {
	if !activeID.Valid || activeID.String == "" {
		return nil, nil
	}
	var activePayload []byte
	if err := tx.QueryRowContext(ctx, `SELECT publication_blob FROM routing_publications
WHERE namespace_id=$1 AND publication_id=$2`, namespaceID, activeID.String).Scan(&activePayload); err != nil {
		return nil, fmt.Errorf("read active PostgreSQL publication: %w", err)
	}
	var active Publication
	if err := decodeStrict(activePayload, &active); err != nil || verifyPublication(active) != nil {
		return nil, fmt.Errorf("%w: active PostgreSQL publication is invalid", ErrStagedCorrupt)
	}
	manifest := active.Manifest
	return &manifest, nil
}

// InstallBarriers is intentionally empty for a routing-only publication. The
// native access topology uses RedisStore, which remains the deny-barrier and
// global-counter authority.
func (s *PostgresStore) InstallBarriers(_ context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	if plan.Restrictive() {
		return fmt.Errorf("PostgreSQL routing publication cannot install access barriers")
	}
	return nil
}

func (s *PostgresStore) Stage(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return fmt.Errorf("begin PostgreSQL publication stage: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	result, err := tx.ExecContext(ctx, `UPDATE routing_publications
SET state=CASE WHEN state='prepared' THEN 'staged' ELSE state END,
    updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND publication_digest=$3
  AND state IN ('prepared','staged','validated')`, p.NamespaceID, p.ID, p.Digest)
	if err != nil {
		return fmt.Errorf("stage PostgreSQL publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	if err := s.replaceRequiredReplicas(ctx, tx, p); err != nil {
		return err
	}
	if err := notifyPostgresPublication(ctx, tx, p.NamespaceID); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit PostgreSQL publication stage: %w", err)
	}
	return nil
}

func (s *PostgresStore) ValidateStaged(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	if err := verifyPublication(plan.Publication); err != nil {
		return fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	p := plan.Publication
	var payload []byte
	var state string
	if err := s.db.QueryRowContext(ctx, `SELECT publication_blob, state FROM routing_publications
WHERE namespace_id=$1 AND publication_id=$2 AND publication_digest=$3`,
		p.NamespaceID, p.ID, p.Digest).Scan(&payload, &state); err != nil {
		return fmt.Errorf("read staged PostgreSQL publication: %w", err)
	}
	var stored Publication
	if err := decodeStrict(payload, &stored); err != nil || verifyPublication(stored) != nil ||
		stored.ID != p.ID || stored.Digest != p.Digest || stored.Routing.Digest != p.Routing.Digest {
		return fmt.Errorf("%w: staged PostgreSQL publication changed", ErrStagedCorrupt)
	}
	if state != "staged" && state != "validated" {
		return ErrConflict
	}
	result, err := s.db.ExecContext(ctx, `UPDATE routing_publications
SET state='validated', updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND state IN ('staged','validated')`, p.NamespaceID, p.ID)
	if err != nil {
		return fmt.Errorf("validate PostgreSQL publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	if err := notifyPostgresPublication(ctx, s.db, p.NamespaceID); err != nil {
		return err
	}
	return nil
}

func (s *PostgresStore) replaceRequiredReplicas(
	ctx context.Context,
	executor postgresExecutor,
	publication Publication,
) error {
	rows, err := executor.QueryContext(ctx, `SELECT replica_id FROM routing_fleet_replicas
WHERE lease_expires_at > clock_timestamp()
UNION
SELECT replica_id FROM routing_replica_leases
WHERE namespace_id=$1 AND lease_expires_at > clock_timestamp()
ORDER BY replica_id`, publication.NamespaceID)
	if err != nil {
		return fmt.Errorf("read live PostgreSQL routing replicas: %w", err)
	}
	var replicas []string
	for rows.Next() {
		var replica string
		if err := rows.Scan(&replica); err != nil {
			_ = rows.Close()
			return fmt.Errorf("scan live PostgreSQL routing replica: %w", err)
		}
		replicas = append(replicas, replica)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close live PostgreSQL routing replicas: %w", err)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate live PostgreSQL routing replicas: %w", err)
	}
	if len(replicas) == 0 {
		return ErrAcknowledgements
	}
	if _, err := executor.ExecContext(ctx, `DELETE FROM routing_publication_required_replicas
WHERE namespace_id=$1 AND publication_id=$2 AND NOT (replica_id = ANY($3))`,
		publication.NamespaceID, publication.ID, pq.Array(replicas)); err != nil {
		return fmt.Errorf("expire PostgreSQL required replicas: %w", err)
	}
	for _, replica := range replicas {
		if _, err := executor.ExecContext(ctx, `INSERT INTO routing_publication_required_replicas
  (namespace_id, publication_id, replica_id) VALUES ($1,$2,$3)
ON CONFLICT DO NOTHING`, publication.NamespaceID, publication.ID, replica); err != nil {
			return fmt.Errorf("record PostgreSQL required replica: %w", err)
		}
	}
	return nil
}

func (s *PostgresStore) BarrierAcknowledgements(_ context.Context, plan PublicationPlan) (AckStatus, error) {
	if err := validatePlan(plan); err != nil {
		return AckStatus{}, err
	}
	return AckStatus{}, nil
}

func (s *PostgresStore) RoutingAcknowledgements(ctx context.Context, plan PublicationPlan) (AckStatus, error) {
	if err := validatePlan(plan); err != nil {
		return AckStatus{}, err
	}
	executor := s.executor(ctx)
	if err := s.replaceRequiredReplicas(ctx, executor, plan.Publication); err != nil {
		return AckStatus{}, err
	}
	rows, err := executor.QueryContext(ctx, `SELECT required.replica_id,
       (ack.replica_id IS NOT NULL) AS acknowledged
FROM routing_publication_required_replicas required
LEFT JOIN routing_publication_acknowledgements ack
  ON ack.namespace_id=required.namespace_id
 AND ack.publication_id=required.publication_id
 AND ack.replica_id=required.replica_id
 AND ack.kind='routing'
WHERE required.namespace_id=$1 AND required.publication_id=$2
ORDER BY required.replica_id`, plan.Publication.NamespaceID, plan.Publication.ID)
	if err != nil {
		return AckStatus{}, fmt.Errorf("read PostgreSQL routing acknowledgements: %w", err)
	}
	defer rows.Close()
	status := AckStatus{}
	for rows.Next() {
		var replica string
		var acknowledged bool
		if err := rows.Scan(&replica, &acknowledged); err != nil {
			return AckStatus{}, fmt.Errorf("scan PostgreSQL routing acknowledgement: %w", err)
		}
		status.Required = append(status.Required, replica)
		if !acknowledged {
			status.Missing = append(status.Missing, replica)
		}
	}
	if err := rows.Err(); err != nil {
		return AckStatus{}, fmt.Errorf("iterate PostgreSQL routing acknowledgements: %w", err)
	}
	return status, nil
}

func (s *PostgresStore) Activate(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	executor := s.executor(ctx)
	p := plan.Publication
	desiredRevision, conversionErr := postgresBigint(p.DesiredRevision, "desired revision")
	if conversionErr != nil {
		return conversionErr
	}
	var missing int
	if queryErr := executor.QueryRowContext(ctx, `SELECT count(*)
FROM routing_publication_required_replicas required
LEFT JOIN routing_publication_acknowledgements ack
  ON ack.namespace_id=required.namespace_id
 AND ack.publication_id=required.publication_id
 AND ack.replica_id=required.replica_id
 AND ack.kind='routing'
WHERE required.namespace_id=$1 AND required.publication_id=$2
	  AND ack.replica_id IS NULL`, p.NamespaceID, p.ID).Scan(&missing); queryErr != nil {
		return fmt.Errorf("verify PostgreSQL routing acknowledgements: %w", queryErr)
	}
	if missing != 0 {
		return ErrAcknowledgements
	}
	if _, retireErr := executor.ExecContext(
		ctx,
		`UPDATE routing_publications SET state='finalized', updated_at=clock_timestamp()
	WHERE namespace_id=$1 AND state IN ('active','applied') AND publication_id<>$2`,
		p.NamespaceID, p.ID,
	); retireErr != nil {
		return fmt.Errorf("retire prior PostgreSQL publication: %w", retireErr)
	}
	result, err := executor.ExecContext(ctx, `UPDATE routing_publications
SET state='active', activated_at=COALESCE(activated_at,clock_timestamp()), updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND publication_digest=$3 AND state IN ('validated','active')`,
		p.NamespaceID, p.ID, p.Digest)
	if err != nil {
		return fmt.Errorf("activate PostgreSQL publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	result, err = executor.ExecContext(ctx, `UPDATE routing_publication_heads
SET active_publication_id=$2, active_revision=$3,
    candidate_publication_id=NULL, candidate_revision=NULL, updated_at=clock_timestamp()
WHERE namespace_id=$1 AND candidate_publication_id=$2 AND candidate_revision=$3
  AND (active_revision IS NULL OR active_revision <= $3)`,
		p.NamespaceID, p.ID, desiredRevision)
	if err != nil {
		return fmt.Errorf("advance PostgreSQL active publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	return notifyPostgresPublication(ctx, executor, p.NamespaceID)
}

func (s *PostgresStore) Compact(_ context.Context, plan PublicationPlan, batchSize int) (bool, error) {
	if err := validatePlan(plan); err != nil {
		return false, err
	}
	if batchSize < 1 || batchSize > 1000 {
		return false, fmt.Errorf("compaction batch size must be between 1 and 1000")
	}
	return true, nil
}

func (s *PostgresStore) MarkApplied(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	result, err := s.db.ExecContext(ctx, `UPDATE routing_publications
SET state='applied', updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND state IN ('active','applied')`, p.NamespaceID, p.ID)
	if err != nil {
		return fmt.Errorf("mark PostgreSQL publication applied: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	return notifyPostgresPublication(ctx, s.db, p.NamespaceID)
}

func (s *PostgresStore) ClearAppliedBarriers(ctx context.Context, plan PublicationPlan) error {
	if err := validatePlan(plan); err != nil {
		return err
	}
	p := plan.Publication
	result, err := s.db.ExecContext(ctx, `UPDATE routing_publications
SET state='finalized', updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND state IN ('applied','finalized')`, p.NamespaceID, p.ID)
	if err != nil {
		return fmt.Errorf("finalize PostgreSQL publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrConflict
	}
	return notifyPostgresPublication(ctx, s.db, p.NamespaceID)
}

func (s *PostgresStore) ReconcileApplied(ctx context.Context, applied AppliedState) error {
	if strings.TrimSpace(applied.NamespaceID) == "" || applied.DesiredRevision == 0 || applied.RuntimeEpoch == 0 {
		return fmt.Errorf("applied namespace, epoch, and revision are required")
	}
	desiredRevision, err := postgresBigint(applied.DesiredRevision, "desired revision")
	if err != nil {
		return err
	}
	runtimeEpoch, err := postgresBigint(applied.RuntimeEpoch, "runtime epoch")
	if err != nil {
		return err
	}
	var publicationID, routingDigest, state string
	var epoch int64
	err = s.db.QueryRowContext(ctx, `SELECT publication_id, routing_digest, runtime_epoch, state
	FROM routing_publications
	WHERE namespace_id=$1 AND desired_revision=$2`, applied.NamespaceID, desiredRevision).Scan(
		&publicationID, &routingDigest, &epoch, &state,
	)
	if err != nil {
		return fmt.Errorf("read applied PostgreSQL publication: %w", err)
	}
	if epoch != runtimeEpoch || (applied.RoutingDigest != "" && routingDigest != applied.RoutingDigest) {
		return ErrConflict
	}
	var activeID string
	if headReadError := s.db.QueryRowContext(ctx, `SELECT active_publication_id FROM routing_publication_heads
	WHERE namespace_id=$1`, applied.NamespaceID).Scan(&activeID); headReadError != nil {
		return fmt.Errorf("read active PostgreSQL publication head: %w", headReadError)
	}
	if activeID != publicationID {
		return ErrConflict
	}
	if state == string(PublicationStateFinalized) {
		return nil
	}
	result, err := s.db.ExecContext(ctx, `UPDATE routing_publications SET state='finalized', updated_at=clock_timestamp()
WHERE namespace_id=$1 AND publication_id=$2 AND state IN ('active','applied')`, applied.NamespaceID, publicationID)
	if err != nil {
		return fmt.Errorf("reconcile applied PostgreSQL publication: %w", err)
	}
	if err := requireRowsAffected(result, 1, "reconcile applied PostgreSQL publication"); err != nil {
		return err
	}
	return notifyPostgresPublication(ctx, s.db, applied.NamespaceID)
}

func (s *PostgresStore) Readiness(ctx context.Context, namespaceID, partition string) (Readiness, error) {
	var readiness Readiness
	var runtimeEpoch int64
	var activeID sql.NullString
	var activeRevision sql.NullInt64
	if err := s.db.QueryRowContext(ctx, `SELECT n.runtime_epoch, h.active_publication_id, h.active_revision
FROM access_namespaces n
LEFT JOIN routing_publication_heads h ON h.namespace_id=n.id
WHERE n.id=$1 AND n.quota_partition_id=$2`, namespaceID, partition).Scan(
		&runtimeEpoch, &activeID, &activeRevision,
	); err != nil {
		return Readiness{}, fmt.Errorf("read PostgreSQL publication readiness: %w", err)
	}
	if runtimeEpoch <= 0 {
		return Readiness{}, fmt.Errorf("%w: PostgreSQL runtime epoch is invalid", ErrStagedCorrupt)
	}
	runtimeEpochValue, err := databasePositiveUint64(runtimeEpoch, "runtime epoch")
	if err != nil {
		return Readiness{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	readiness.RuntimeEpoch = runtimeEpochValue
	if activeID.Valid {
		readiness.AccessGate = activeID.String
		readiness.RoutingGate = activeID.String
	}
	if activeRevision.Valid {
		desiredRevision, conversionErr := databasePositiveUint64(activeRevision.Int64, "active revision")
		if conversionErr != nil {
			return Readiness{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, conversionErr)
		}
		readiness.DesiredRevision = desiredRevision
	}
	var applied int64
	err = s.db.QueryRowContext(ctx, `SELECT applied_revision FROM projector_watermarks
WHERE projector=$1 AND namespace_id=$2`, s.projector, namespaceID).Scan(&applied)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return Readiness{}, fmt.Errorf("read PostgreSQL publication watermark: %w", err)
	}
	if applied > 0 {
		appliedRevision, conversionErr := databasePositiveUint64(applied, "applied revision")
		if conversionErr != nil {
			return Readiness{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, conversionErr)
		}
		readiness.AppliedRevision = appliedRevision
	}
	if readiness.DesiredRevision >= readiness.AppliedRevision {
		readiness.ProjectorLag = readiness.DesiredRevision - readiness.AppliedRevision
	}
	switch {
	case readiness.RuntimeEpoch == 0:
		readiness.Reason = "runtime_epoch_unpublished"
	case readiness.RoutingGate == "":
		readiness.Reason = "publication_gate_unpublished"
	case readiness.AppliedRevision != readiness.DesiredRevision:
		readiness.Reason = "applied_revision_lagging"
	default:
		readiness.Ready = true
		readiness.Reason = "ready"
	}
	return readiness, nil
}

func (s *PostgresStore) RegisterFleetReplica(ctx context.Context, replicaID string) (time.Time, error) {
	if strings.TrimSpace(replicaID) == "" || len(replicaID) > 256 || strings.ContainsRune(replicaID, 0) {
		return time.Time{}, fmt.Errorf("replica id is required and must not exceed 256 bytes")
	}
	var expiry time.Time
	err := s.db.QueryRowContext(ctx, `INSERT INTO routing_fleet_replicas (replica_id, lease_expires_at)
VALUES ($1, clock_timestamp() + ($2 * interval '1 millisecond'))
ON CONFLICT (replica_id) DO UPDATE
SET lease_expires_at=EXCLUDED.lease_expires_at, updated_at=clock_timestamp()
RETURNING lease_expires_at`, replicaID, s.replicaLease.Milliseconds()).Scan(&expiry)
	if err != nil {
		return time.Time{}, fmt.Errorf("register PostgreSQL fleet replica: %w", err)
	}
	return expiry.UTC(), nil
}

func (s *PostgresStore) ListPublicationNamespaces(ctx context.Context) ([]NamespacePublication, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, quota_partition_id
FROM access_namespaces
WHERE status='active'
ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("list PostgreSQL publication namespaces: %w", err)
	}
	defer rows.Close()
	var references []NamespacePublication
	for rows.Next() {
		var reference NamespacePublication
		if err := rows.Scan(&reference.NamespaceID, &reference.QuotaPartition); err != nil {
			return nil, fmt.Errorf("scan PostgreSQL publication namespace: %w", err)
		}
		if err := reference.Validate(); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
		}
		references = append(references, reference)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate PostgreSQL publication namespaces: %w", err)
	}
	return references, nil
}
