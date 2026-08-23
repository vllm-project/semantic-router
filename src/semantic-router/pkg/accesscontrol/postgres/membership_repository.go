package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getMembershipQuery = `SELECT namespace_id, team_id, user_id, role, status,
       revision, created_at, updated_at
FROM access_team_memberships
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3`
	insertMembershipQuery = `INSERT INTO access_team_memberships
  (namespace_id, team_id, user_id, role, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, 1, $6, $7)
RETURNING namespace_id, team_id, user_id, role, status,
          revision, created_at, updated_at`
	updateMembershipQuery = `UPDATE access_team_memberships
SET role = $5, status = $6, revision = revision + 1,
    updated_at = clock_timestamp()
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3 AND revision = $4
RETURNING namespace_id, team_id, user_id, role, status,
          revision, created_at, updated_at`
)

func (s *Store) GetMembership(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	teamID accesscontrol.TeamID,
	userID accesscontrol.UserID,
) (MembershipRecord, error) {
	if err := validateMembershipIDs(namespaceID, teamID, userID); err != nil {
		return MembershipRecord{}, err
	}
	record, err := scanMembership(s.db.QueryRowContext(ctx, getMembershipQuery, namespaceID, teamID, userID))
	if errors.Is(err, sql.ErrNoRows) {
		return MembershipRecord{}, ErrNotFound
	}
	if err != nil {
		return MembershipRecord{}, fmt.Errorf("get team membership: %w", err)
	}
	return record, nil
}

func (s *Store) CreateMembership(
	ctx context.Context,
	membership accesscontrol.TeamMembership,
	meta MutationMeta,
) (MutationResult[MembershipRecord], error) {
	if err := validateMembership(membership); err != nil {
		return MutationResult[MembershipRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[MembershipRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[MembershipRecord], error) {
		created, err := scanMembership(tx.QueryRowContext(ctx, insertMembershipQuery,
			membership.NamespaceID, membership.TeamID, membership.UserID,
			membership.Role, membership.Status, membership.CreatedAt, membership.UpdatedAt))
		if err != nil {
			return MutationResult[MembershipRecord]{}, fmt.Errorf("insert team membership: %w", err)
		}
		receipt, err := appendMembershipOutbox(ctx, tx, created, outboxCreated, meta)
		if err != nil {
			return MutationResult[MembershipRecord]{}, err
		}
		return MutationResult[MembershipRecord]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateMembership(
	ctx context.Context,
	membership accesscontrol.TeamMembership,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[MembershipRecord], error) {
	if err := validateMembership(membership); err != nil {
		return MutationResult[MembershipRecord]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[MembershipRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[MembershipRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[MembershipRecord], error) {
		updated, err := scanMembership(tx.QueryRowContext(ctx, updateMembershipQuery,
			membership.NamespaceID, membership.TeamID, membership.UserID,
			expectedRevision, membership.Role, membership.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[MembershipRecord]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[MembershipRecord]{}, fmt.Errorf("update team membership: %w", err)
		}
		receipt, err := appendMembershipOutbox(ctx, tx, updated, outboxUpdated, meta)
		if err != nil {
			return MutationResult[MembershipRecord]{}, err
		}
		return MutationResult[MembershipRecord]{Value: updated, Receipt: receipt}, nil
	})
}

func appendMembershipOutbox(
	ctx context.Context,
	tx *sql.Tx,
	record MembershipRecord,
	operation outboxOperation,
	meta MutationMeta,
) (MutationReceipt, error) {
	membership := record.Membership
	return appendMutationRecords(ctx, tx, membership.NamespaceID, outboxMutation{
		AggregateType:     "team_membership",
		AggregateID:       membershipEventAggregateID(membership),
		AggregateRevision: record.Revision,
		Operation:         operation,
		References: map[string]string{
			"namespaceId": string(membership.NamespaceID),
			"teamId":      string(membership.TeamID),
			"userId":      string(membership.UserID),
			"resourceRef": membershipResourceReference(membership),
		},
	}, meta)
}

func validateMembership(membership accesscontrol.TeamMembership) error {
	if err := membership.Validate(); err != nil {
		return err
	}
	return validateMembershipIDs(membership.NamespaceID, membership.TeamID, membership.UserID)
}

func validateMembershipIDs(namespaceID accesscontrol.NamespaceID, teamID accesscontrol.TeamID, userID accesscontrol.UserID) error {
	if err := validateIdentityIDs(namespaceID, string(teamID)); err != nil {
		return err
	}
	return validateUUID("user id", string(userID))
}

func scanMembership(scanner rowScanner) (MembershipRecord, error) {
	var record MembershipRecord
	var revision int64
	if err := scanner.Scan(
		&record.Membership.NamespaceID, &record.Membership.TeamID, &record.Membership.UserID,
		&record.Membership.Role, &record.Membership.Status, &revision,
		&record.Membership.CreatedAt, &record.Membership.UpdatedAt,
	); err != nil {
		return MembershipRecord{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return MembershipRecord{}, err
	}
	record.Revision = parsedRevision
	return record, nil
}
