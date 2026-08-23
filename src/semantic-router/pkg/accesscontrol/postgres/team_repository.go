package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getTeamQuery = `SELECT id, namespace_id, name, description, status,
       revision, created_at, updated_at, deleted_at
FROM access_teams
WHERE namespace_id = $1 AND id = $2`
	insertTeamQuery = `INSERT INTO access_teams
  (id, namespace_id, name, description, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, 1, $6, $7)
RETURNING id, namespace_id, name, description, status,
          revision, created_at, updated_at, deleted_at`
	updateTeamQuery = `UPDATE access_teams
SET name = $4, description = $5, status = $6,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING id, namespace_id, name, description, status,
          revision, created_at, updated_at, deleted_at`
	softDeleteTeamQuery = `UPDATE access_teams
SET status = 'disabled', deleted_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING id, namespace_id, name, description, status,
          revision, created_at, updated_at, deleted_at`
)

func (s *Store) GetTeam(ctx context.Context, namespaceID accesscontrol.NamespaceID, id accesscontrol.TeamID) (TeamRecord, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return TeamRecord{}, err
	}
	record, err := scanTeam(s.db.QueryRowContext(ctx, getTeamQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return TeamRecord{}, ErrNotFound
	}
	if err != nil {
		return TeamRecord{}, fmt.Errorf("get team: %w", err)
	}
	return record, nil
}

func (s *Store) CreateTeam(
	ctx context.Context,
	record TeamRecord,
	meta MutationMeta,
) (MutationResult[TeamRecord], error) {
	if err := validateTeamRecord(record); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	if record.Revision != 1 {
		return MutationResult[TeamRecord]{}, fmt.Errorf("new team revision must be 1")
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[TeamRecord], error) {
		team := record.Team
		if _, err := tx.ExecContext(ctx, insertSubjectQuery,
			team.NamespaceID, team.ID, accesscontrol.SubjectKindTeam, team.CreatedAt); err != nil {
			return MutationResult[TeamRecord]{}, fmt.Errorf("insert team subject: %w", err)
		}
		created, err := scanTeam(tx.QueryRowContext(ctx, insertTeamQuery,
			team.ID, team.NamespaceID, team.Name, record.Description, team.Status,
			team.CreatedAt, team.UpdatedAt))
		if err != nil {
			return MutationResult[TeamRecord]{}, fmt.Errorf("insert team: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, team.NamespaceID, outboxMutation{
			AggregateType: "team", AggregateID: string(team.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
		}, meta)
		if err != nil {
			return MutationResult[TeamRecord]{}, err
		}
		return MutationResult[TeamRecord]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateTeam(
	ctx context.Context,
	record TeamRecord,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[TeamRecord], error) {
	if err := validateTeamRecord(record); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	if record.Revision != expected {
		return MutationResult[TeamRecord]{}, fmt.Errorf("team record revision must match expected revision")
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[TeamRecord], error) {
		team := record.Team
		updated, err := scanTeam(tx.QueryRowContext(ctx, updateTeamQuery,
			team.NamespaceID, team.ID, expectedRevision, team.Name, record.Description, team.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[TeamRecord]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[TeamRecord]{}, fmt.Errorf("update team: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, team.NamespaceID, outboxMutation{
			AggregateType: "team", AggregateID: string(team.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if err != nil {
			return MutationResult[TeamRecord]{}, err
		}
		return MutationResult[TeamRecord]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) SoftDeleteTeam(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.TeamID,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[TeamRecord], error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[TeamRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[TeamRecord], error) {
		deleted, err := scanTeam(tx.QueryRowContext(ctx, softDeleteTeamQuery, namespaceID, id, expectedRevision))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[TeamRecord]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[TeamRecord]{}, fmt.Errorf("soft-delete team: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "team", AggregateID: string(id),
			AggregateRevision: deleted.Revision, Operation: outboxDeleted,
		}, meta)
		if err != nil {
			return MutationResult[TeamRecord]{}, err
		}
		return MutationResult[TeamRecord]{Value: deleted, Receipt: receipt}, nil
	})
}

func validateTeamRecord(record TeamRecord) error {
	if err := record.Team.Validate(); err != nil {
		return err
	}
	if record.DeletedAt != nil {
		return fmt.Errorf("deleted_at is managed only by SoftDeleteTeam")
	}
	return validateIdentityIDs(record.Team.NamespaceID, string(record.Team.ID))
}

func scanTeam(scanner rowScanner) (TeamRecord, error) {
	var record TeamRecord
	var revision int64
	var deletedAt sql.NullTime
	if err := scanner.Scan(
		&record.Team.ID, &record.Team.NamespaceID, &record.Team.Name, &record.Description,
		&record.Team.Status, &revision, &record.Team.CreatedAt, &record.Team.UpdatedAt, &deletedAt,
	); err != nil {
		return TeamRecord{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return TeamRecord{}, err
	}
	record.Revision = parsedRevision
	if deletedAt.Valid {
		value := deletedAt.Time
		record.DeletedAt = &value
		record.Team.Status = accesscontrol.TeamStatusDisabled
	}
	return record, nil
}
