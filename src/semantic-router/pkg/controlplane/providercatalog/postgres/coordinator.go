package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

const (
	minimumReplicaLease = time.Second
	maximumReplicaLease = 5 * time.Minute
)

type Coordinator struct {
	db       *sql.DB
	registry *providercatalog.Registry
}

var _ providercatalog.SnapshotSource = (*Coordinator)(nil)

func New(db *sql.DB, registry *providercatalog.Registry) (*Coordinator, error) {
	if db == nil {
		return nil, fmt.Errorf("provider catalog PostgreSQL database is required")
	}
	if registry == nil {
		return nil, fmt.Errorf("provider integration registry is required")
	}
	return &Coordinator{db: db, registry: registry}, nil
}

// Stage is the only integration-to-desired publication operation. It atomically
// persists the immutable snapshot, compares the caller's desired-state token,
// installs an explicit stable rollout-group gate, and advances desired revision.
func (c *Coordinator) Stage(ctx context.Context, request StageRequest) (State, error) {
	if request.ExpectedGeneration == 0 || request.ExpectedGeneration > math.MaxInt64 {
		return State{}, fmt.Errorf("expected generation must fit a positive PostgreSQL BIGINT")
	}
	if request.ExpectedDesiredRevision != "" && !validRevision(request.ExpectedDesiredRevision) {
		return State{}, fmt.Errorf("expected desired revision is invalid")
	}
	required, stageErr := normalizeRolloutGroups(request.RequiredRolloutGroups)
	if stageErr != nil {
		return State{}, stageErr
	}
	compiled, stageErr := compilePersistedSnapshot(request.Snapshot, c.registry)
	if stageErr != nil {
		return State{}, stageErr
	}
	tx, stageErr := c.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if stageErr != nil {
		return State{}, fmt.Errorf("begin provider catalog stage: %w", stageErr)
	}
	defer func() { _ = tx.Rollback() }()
	if err := insertImmutableSnapshot(ctx, tx, compiled, c.registry); err != nil {
		return State{}, err
	}
	current, stageErr := readState(ctx, tx, ` FOR UPDATE`)
	if stageErr != nil {
		return State{}, stageErr
	}
	currentRequired, stageErr := readRequiredRolloutGroups(ctx, tx, current.Generation, current.DesiredRevision)
	if stageErr != nil {
		return State{}, stageErr
	}
	if current.DesiredRevision != "" && len(currentRequired) == 0 {
		return State{}, fmt.Errorf("%w: desired revision has no required rollout groups", ErrCorruptState)
	}
	if current.DesiredRevision == compiled.revision && reflect.DeepEqual(currentRequired, required) {
		if err := tx.Commit(); err != nil {
			return State{}, fmt.Errorf("commit idempotent provider catalog stage: %w", err)
		}
		return current, nil
	}
	if current.Generation != request.ExpectedGeneration ||
		current.DesiredRevision != request.ExpectedDesiredRevision {
		return State{}, providercatalog.ErrPublicationConflict
	}
	if current.Generation >= math.MaxInt64 {
		return State{}, fmt.Errorf("provider catalog generation is exhausted")
	}
	nextGeneration := current.Generation + 1
	var updated State
	var desired, active sql.NullString
	stageErr = tx.QueryRowContext(ctx, `UPDATE provider_catalog_state
SET desired_revision = $1, generation = $2, updated_at = clock_timestamp()
WHERE singleton = TRUE AND generation = $3
  AND desired_revision IS NOT DISTINCT FROM NULLIF($4, '')
RETURNING desired_revision, active_revision, generation, updated_at`,
		compiled.revision, int64(nextGeneration), int64(request.ExpectedGeneration), request.ExpectedDesiredRevision,
	).Scan(&desired, &active, &updated.Generation, &updated.UpdatedAt)
	if errors.Is(stageErr, sql.ErrNoRows) {
		return State{}, providercatalog.ErrPublicationConflict
	}
	if stageErr != nil {
		return State{}, fmt.Errorf("advance provider catalog desired revision: %w", stageErr)
	}
	updated.DesiredRevision, updated.ActiveRevision = desired.String, active.String
	for _, group := range required {
		if _, err := tx.ExecContext(ctx, `INSERT INTO provider_catalog_required_rollout_groups
  (generation, revision, plane, rollout_group) VALUES ($1, $2, $3, $4)`,
			int64(nextGeneration), compiled.revision, group.Plane, group.ID); err != nil {
			return State{}, fmt.Errorf("persist required Provider Catalog rollout group %s: %w", group.Key(), err)
		}
	}
	if err := tx.Commit(); err != nil {
		return State{}, fmt.Errorf("commit provider catalog stage: %w", err)
	}
	return updated, nil
}

func (c *Coordinator) Acknowledge(ctx context.Context, request AcknowledgeRequest) (ReplicaAcknowledgement, error) {
	if !validRevision(request.Revision) {
		return ReplicaAcknowledgement{}, fmt.Errorf("provider catalog acknowledgement revision is invalid")
	}
	if err := validateReplicaID(request.ReplicaID); err != nil {
		return ReplicaAcknowledgement{}, err
	}
	if err := request.RolloutGroup.Validate(); err != nil {
		return ReplicaAcknowledgement{}, err
	}
	if len(request.CapabilityDigest) != 32 {
		return ReplicaAcknowledgement{}, fmt.Errorf("capability digest must be exactly 32 bytes")
	}
	if request.Lease < minimumReplicaLease || request.Lease > maximumReplicaLease {
		return ReplicaAcknowledgement{}, fmt.Errorf("replica lease must be between one second and five minutes")
	}
	switch request.Status {
	case AckCompatible:
		if request.Reason != "" {
			return ReplicaAcknowledgement{}, fmt.Errorf("compatible acknowledgement cannot include a reason")
		}
	case AckIncompatible:
		if request.Reason == "" || request.Reason != strings.TrimSpace(request.Reason) || len(request.Reason) > 1024 {
			return ReplicaAcknowledgement{}, fmt.Errorf("incompatible acknowledgement requires a canonical bounded reason")
		}
	default:
		return ReplicaAcknowledgement{}, fmt.Errorf("acknowledgement status is invalid")
	}
	tx, err := c.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		return ReplicaAcknowledgement{}, fmt.Errorf("begin provider catalog acknowledgement: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	state, err := readState(ctx, tx, ` FOR SHARE`)
	if err != nil {
		return ReplicaAcknowledgement{}, err
	}
	if request.Revision != state.DesiredRevision && request.Revision != state.ActiveRevision {
		return ReplicaAcknowledgement{}, ErrStaleRevision
	}
	var result ReplicaAcknowledgement
	err = tx.QueryRowContext(ctx, `WITH observed AS (SELECT clock_timestamp() AS now)
INSERT INTO provider_catalog_replica_acks
  (revision, plane, rollout_group, replica_id, capability_digest, status, reason, acknowledged_at, lease_expires_at)
SELECT $1, $2, $3, $4, $5, $6, $7, observed.now,
       observed.now + ($8 * interval '1 millisecond')
FROM observed
ON CONFLICT (revision, plane, rollout_group, replica_id) DO UPDATE
SET capability_digest = EXCLUDED.capability_digest, status = EXCLUDED.status,
    reason = EXCLUDED.reason, acknowledged_at = EXCLUDED.acknowledged_at,
    lease_expires_at = EXCLUDED.lease_expires_at
RETURNING revision, plane, rollout_group, replica_id, capability_digest, status, reason, acknowledged_at, lease_expires_at`,
		request.Revision, request.RolloutGroup.Plane, request.RolloutGroup.ID, request.ReplicaID,
		request.CapabilityDigest, request.Status, request.Reason, request.Lease.Milliseconds(),
	).Scan(&result.Revision, &result.RolloutGroup.Plane, &result.RolloutGroup.ID, &result.ReplicaID,
		&result.CapabilityDigest, &result.Status, &result.Reason, &result.AcknowledgedAt, &result.LeaseExpiresAt)
	if err != nil {
		return ReplicaAcknowledgement{}, fmt.Errorf("persist provider catalog acknowledgement: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return ReplicaAcknowledgement{}, fmt.Errorf("commit provider catalog acknowledgement: %w", err)
	}
	return result, nil
}

func (c *Coordinator) Activate(ctx context.Context, request ActivateRequest) (State, error) {
	if !validRevision(request.Revision) || request.ExpectedGeneration == 0 || request.ExpectedGeneration > math.MaxInt64 {
		return State{}, fmt.Errorf("activation revision and expected generation are required")
	}
	tx, activateErr := c.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if activateErr != nil {
		return State{}, fmt.Errorf("begin provider catalog activation: %w", activateErr)
	}
	defer func() { _ = tx.Rollback() }()
	state, activateErr := readState(ctx, tx, ` FOR UPDATE`)
	if activateErr != nil {
		return State{}, activateErr
	}
	if state.DesiredRevision != request.Revision || state.Generation != request.ExpectedGeneration {
		return State{}, providercatalog.ErrPublicationConflict
	}
	if state.ActiveRevision == request.Revision {
		if err := tx.Commit(); err != nil {
			return State{}, fmt.Errorf("commit idempotent provider catalog activation: %w", err)
		}
		return state, nil
	}
	required, activateErr := readRequiredRolloutGroups(ctx, tx, state.Generation, state.DesiredRevision)
	if activateErr != nil {
		return State{}, activateErr
	}
	if len(required) == 0 {
		return State{}, fmt.Errorf("%w: staged revision has no required rollout groups", ErrCorruptState)
	}
	acknowledgements, observedAt, activateErr := readAcknowledgementsForActivation(ctx, tx, request.Revision, required)
	if activateErr != nil {
		return State{}, activateErr
	}
	blockers := classifyBlockers(required, acknowledgements, observedAt)
	if !blockers.Empty() {
		return State{}, &providercatalog.ActivationBlockedError{Revision: request.Revision, Blockers: blockers}
	}
	var desired, active sql.NullString
	activateErr = tx.QueryRowContext(ctx, `UPDATE provider_catalog_state
SET active_revision = $1, updated_at = clock_timestamp()
WHERE singleton = TRUE AND desired_revision = $1 AND generation = $2
RETURNING desired_revision, active_revision, generation, updated_at`,
		request.Revision, int64(request.ExpectedGeneration),
	).Scan(&desired, &active, &state.Generation, &state.UpdatedAt)
	if errors.Is(activateErr, sql.ErrNoRows) {
		return State{}, providercatalog.ErrPublicationConflict
	}
	if activateErr != nil {
		return State{}, fmt.Errorf("activate provider catalog revision: %w", activateErr)
	}
	state.DesiredRevision, state.ActiveRevision = desired.String, active.String
	if err := tx.Commit(); err != nil {
		return State{}, fmt.Errorf("commit provider catalog activation: %w", err)
	}
	return state, nil
}

func (c *Coordinator) State(ctx context.Context) (State, error) {
	return readState(ctx, c.db, "")
}

func (c *Coordinator) ActiveSnapshot(ctx context.Context) (*providercatalog.Snapshot, error) {
	return c.snapshotAtState(ctx, true)
}

func (c *Coordinator) DesiredSnapshot(ctx context.Context) (*providercatalog.Snapshot, error) {
	return c.snapshotAtState(ctx, false)
}

func (c *Coordinator) snapshotAtState(ctx context.Context, active bool) (*providercatalog.Snapshot, error) {
	tx, err := c.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return nil, fmt.Errorf("begin provider catalog snapshot read: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	state, err := readState(ctx, tx, "")
	if err != nil {
		return nil, err
	}
	revision := state.DesiredRevision
	missing := ErrNoDesiredSnapshot
	if active {
		revision = state.ActiveRevision
		missing = ErrNoActiveSnapshot
	}
	if revision == "" {
		return nil, missing
	}
	stored, err := readPersistedSnapshot(ctx, tx, revision)
	if err != nil {
		return nil, err
	}
	snapshot, err := restorePersistedSnapshot(stored, c.registry)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit provider catalog snapshot read: %w", err)
	}
	return snapshot, nil
}

func readState(ctx context.Context, query rowQuerier, lockClause string) (State, error) {
	statement := `SELECT desired_revision, active_revision, generation, updated_at
FROM provider_catalog_state WHERE singleton = TRUE`
	if lockClause != "" {
		statement += lockClause
	}
	var desired, active sql.NullString
	var generation int64
	var state State
	err := query.QueryRowContext(ctx, statement).Scan(&desired, &active, &generation, &state.UpdatedAt)
	if err == sql.ErrNoRows {
		return State{}, fmt.Errorf("%w: provider catalog singleton state is absent", ErrCorruptState)
	}
	if err != nil {
		return State{}, fmt.Errorf("read provider catalog state: %w", err)
	}
	if generation <= 0 || (desired.Valid && !validRevision(desired.String)) ||
		(active.Valid && !validRevision(active.String)) {
		return State{}, fmt.Errorf("%w: provider catalog state is invalid", ErrCorruptState)
	}
	state.DesiredRevision, state.ActiveRevision, state.Generation = desired.String, active.String, uint64(generation)
	return state, nil
}

func readRequiredRolloutGroups(
	ctx context.Context,
	query rowQuerier,
	generation uint64,
	revision string,
) (_ []providercatalog.RolloutGroup, returnErr error) {
	if revision == "" {
		return nil, nil
	}
	rows, err := queryRows(ctx, query, `SELECT plane, rollout_group
FROM provider_catalog_required_rollout_groups
WHERE generation = $1 AND revision = $2 ORDER BY plane, rollout_group`, int64(generation), revision)
	if err != nil {
		return nil, fmt.Errorf("read required Provider Catalog rollout groups: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]providercatalog.RolloutGroup, 0)
	for rows.Next() {
		var group providercatalog.RolloutGroup
		if err := rows.Scan(&group.Plane, &group.ID); err != nil {
			return nil, err
		}
		if err := group.Validate(); err != nil {
			return nil, fmt.Errorf("%w: invalid required rollout group", ErrCorruptState)
		}
		result = append(result, group)
	}
	return result, rows.Err()
}

type rowsQuerier interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

func queryRows(ctx context.Context, query rowQuerier, statement string, arguments ...any) (*sql.Rows, error) {
	rowsQuery, ok := query.(rowsQuerier)
	if !ok {
		return nil, fmt.Errorf("query does not support rows")
	}
	return rowsQuery.QueryContext(ctx, statement, arguments...)
}

func readAcknowledgementsForActivation(
	ctx context.Context,
	tx *sql.Tx,
	revision string,
	required []providercatalog.RolloutGroup,
) (_ map[string][]ReplicaAcknowledgement, _ time.Time, returnErr error) {
	var observedAt time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&observedAt); err != nil {
		return nil, time.Time{}, fmt.Errorf("read provider catalog activation time: %w", err)
	}
	rows, err := tx.QueryContext(ctx, `SELECT revision, plane, rollout_group, replica_id, capability_digest, status, reason,
acknowledged_at, lease_expires_at FROM provider_catalog_replica_acks
WHERE revision = $1 ORDER BY plane, rollout_group, replica_id FOR SHARE`, revision)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("read provider catalog activation acknowledgements: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	requiredKeys := make(map[string]struct{}, len(required))
	for _, group := range required {
		requiredKeys[group.Key()] = struct{}{}
	}
	result := make(map[string][]ReplicaAcknowledgement, len(required))
	for rows.Next() {
		var acknowledgement ReplicaAcknowledgement
		if err := rows.Scan(&acknowledgement.Revision, &acknowledgement.RolloutGroup.Plane,
			&acknowledgement.RolloutGroup.ID, &acknowledgement.ReplicaID,
			&acknowledgement.CapabilityDigest, &acknowledgement.Status, &acknowledgement.Reason,
			&acknowledgement.AcknowledgedAt, &acknowledgement.LeaseExpiresAt); err != nil {
			return nil, time.Time{}, err
		}
		if err := validateStoredAcknowledgement(acknowledgement, revision, observedAt); err != nil {
			return nil, time.Time{}, err
		}
		if _, needed := requiredKeys[acknowledgement.RolloutGroup.Key()]; needed {
			key := acknowledgement.RolloutGroup.Key()
			result[key] = append(result[key], acknowledgement)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, time.Time{}, err
	}
	return result, observedAt.UTC(), nil
}

func validateStoredAcknowledgement(
	acknowledgement ReplicaAcknowledgement,
	revision string,
	observedAt time.Time,
) error {
	lease := acknowledgement.LeaseExpiresAt.Sub(acknowledgement.AcknowledgedAt)
	if acknowledgement.Revision != revision || len(acknowledgement.CapabilityDigest) != 32 ||
		validateReplicaID(acknowledgement.ReplicaID) != nil ||
		acknowledgement.RolloutGroup.Validate() != nil ||
		acknowledgement.AcknowledgedAt.After(observedAt) ||
		lease < minimumReplicaLease || lease > maximumReplicaLease {
		return fmt.Errorf("%w: replica acknowledgement is invalid", ErrCorruptState)
	}
	switch acknowledgement.Status {
	case AckCompatible:
		if acknowledgement.Reason != "" {
			return fmt.Errorf("%w: compatible replica acknowledgement has a reason", ErrCorruptState)
		}
	case AckIncompatible:
		if acknowledgement.Reason == "" || acknowledgement.Reason != strings.TrimSpace(acknowledgement.Reason) {
			return fmt.Errorf("%w: incompatible replica acknowledgement has no canonical reason", ErrCorruptState)
		}
	default:
		return fmt.Errorf("%w: replica acknowledgement status is invalid", ErrCorruptState)
	}
	return nil
}

func classifyBlockers(
	required []providercatalog.RolloutGroup,
	acknowledgements map[string][]ReplicaAcknowledgement,
	observedAt time.Time,
) providercatalog.ActivationBlockers {
	var blockers providercatalog.ActivationBlockers
	for _, group := range required {
		groupAcknowledgements := acknowledgements[group.Key()]
		if len(groupAcknowledgements) == 0 {
			blockers.Missing = append(blockers.Missing, group)
			continue
		}
		compatibleLive, expired := false, false
		compatibleDigests := make(map[string]struct{})
		incompatibleCount := 0
		for _, acknowledgement := range groupAcknowledgements {
			if !acknowledgement.LeaseExpiresAt.After(observedAt) {
				expired = true
				continue
			}
			if acknowledgement.Status == AckIncompatible {
				incompatibleCount++
				blockers.Incompatible = append(blockers.Incompatible, providercatalog.ReplicaBlocker{
					RolloutGroup: group, ReplicaID: acknowledgement.ReplicaID, Reason: acknowledgement.Reason,
				})
				continue
			}
			compatibleLive = true
			compatibleDigests[string(acknowledgement.CapabilityDigest)] = struct{}{}
		}
		if len(compatibleDigests) > 1 {
			blockers.Divergent = append(blockers.Divergent, group)
		}
		if incompatibleCount == 0 && !compatibleLive {
			if expired {
				blockers.Expired = append(blockers.Expired, group)
			} else {
				blockers.Missing = append(blockers.Missing, group)
			}
		}
	}
	sort.Slice(blockers.Missing, func(i, j int) bool { return blockers.Missing[i].Key() < blockers.Missing[j].Key() })
	sort.Slice(blockers.Expired, func(i, j int) bool { return blockers.Expired[i].Key() < blockers.Expired[j].Key() })
	sort.Slice(blockers.Divergent, func(i, j int) bool { return blockers.Divergent[i].Key() < blockers.Divergent[j].Key() })
	sort.Slice(blockers.Incompatible, func(i, j int) bool {
		left := blockers.Incompatible[i].RolloutGroup.Key() + "/" + blockers.Incompatible[i].ReplicaID
		right := blockers.Incompatible[j].RolloutGroup.Key() + "/" + blockers.Incompatible[j].ReplicaID
		return left < right
	})
	return blockers
}
