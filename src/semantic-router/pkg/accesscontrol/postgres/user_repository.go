package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getUserQuery = `SELECT id, namespace_id, email, display_name, status,
       revision, created_at, updated_at, deleted_at
FROM access_users
WHERE namespace_id = $1 AND id = $2`
	insertSubjectQuery = `INSERT INTO access_subjects (namespace_id, id, kind, created_at)
VALUES ($1, $2, $3, $4)`
	insertUserQuery = `INSERT INTO access_users
  (id, namespace_id, email, display_name, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, 1, $6, $7)
RETURNING id, namespace_id, email, display_name, status,
          revision, created_at, updated_at, deleted_at`
	updateUserQuery = `UPDATE access_users
SET email = $4, display_name = $5, status = $6,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING id, namespace_id, email, display_name, status,
          revision, created_at, updated_at, deleted_at`
	softDeleteUserQuery = `UPDATE access_users
SET status = 'disabled', deleted_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING id, namespace_id, email, display_name, status,
          revision, created_at, updated_at, deleted_at`
)

func (s *Store) GetUser(ctx context.Context, namespaceID accesscontrol.NamespaceID, id accesscontrol.UserID) (UserRecord, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return UserRecord{}, err
	}
	record, err := scanUser(s.db.QueryRowContext(ctx, getUserQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return UserRecord{}, ErrNotFound
	}
	if err != nil {
		return UserRecord{}, fmt.Errorf("get user: %w", err)
	}
	return record, nil
}

func (s *Store) CreateUser(
	ctx context.Context,
	user accesscontrol.User,
	meta MutationMeta,
) (MutationResult[UserRecord], error) {
	if err := validatePersistableUser(user); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[UserRecord], error) {
		if _, err := tx.ExecContext(ctx, insertSubjectQuery,
			user.NamespaceID, user.ID, accesscontrol.SubjectKindUser, user.CreatedAt); err != nil {
			return MutationResult[UserRecord]{}, fmt.Errorf("insert user subject: %w", err)
		}
		created, err := scanUser(tx.QueryRowContext(ctx, insertUserQuery,
			user.ID, user.NamespaceID, user.Email, user.DisplayName, user.Status,
			user.CreatedAt, user.UpdatedAt))
		if err != nil {
			return MutationResult[UserRecord]{}, fmt.Errorf("insert user: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, user.NamespaceID, outboxMutation{
			AggregateType: "user", AggregateID: string(user.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
		}, meta)
		if err != nil {
			return MutationResult[UserRecord]{}, err
		}
		return MutationResult[UserRecord]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateUser(
	ctx context.Context,
	user accesscontrol.User,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[UserRecord], error) {
	if err := validatePersistableUser(user); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[UserRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[UserRecord], error) {
		updated, err := scanUser(tx.QueryRowContext(ctx, updateUserQuery,
			user.NamespaceID, user.ID, expectedRevision, user.Email, user.DisplayName, user.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[UserRecord]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[UserRecord]{}, fmt.Errorf("update user: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, user.NamespaceID, outboxMutation{
			AggregateType: "user", AggregateID: string(user.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if err != nil {
			return MutationResult[UserRecord]{}, err
		}
		return MutationResult[UserRecord]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) SoftDeleteUser(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.UserID,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[UserRecord], error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[UserRecord]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[UserRecord]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[UserRecord], error) {
		deleted, err := scanUser(tx.QueryRowContext(ctx, softDeleteUserQuery, namespaceID, id, expectedRevision))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[UserRecord]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[UserRecord]{}, fmt.Errorf("soft-delete user: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "user", AggregateID: string(id),
			AggregateRevision: deleted.Revision, Operation: outboxDeleted,
		}, meta)
		if err != nil {
			return MutationResult[UserRecord]{}, err
		}
		return MutationResult[UserRecord]{Value: deleted, Receipt: receipt}, nil
	})
}

func validatePersistableUser(user accesscontrol.User) error {
	if err := user.Validate(); err != nil {
		return err
	}
	if user.Status == accesscontrol.UserStatusDeleted {
		return fmt.Errorf("deleted user status is derived from deleted_at and cannot be written directly")
	}
	return validateIdentityIDs(user.NamespaceID, string(user.ID))
}

func validateIdentityIDs(namespaceID accesscontrol.NamespaceID, id string) error {
	if err := validateUUID("namespace id", string(namespaceID)); err != nil {
		return err
	}
	return validateUUID("resource id", id)
}

func scanUser(scanner rowScanner) (UserRecord, error) {
	var record UserRecord
	var storedStatus accesscontrol.UserStatus
	var revision int64
	var deletedAt sql.NullTime
	if err := scanner.Scan(
		&record.User.ID, &record.User.NamespaceID, &record.User.Email, &record.User.DisplayName,
		&storedStatus, &revision, &record.User.CreatedAt, &record.User.UpdatedAt, &deletedAt,
	); err != nil {
		return UserRecord{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return UserRecord{}, err
	}
	record.User.Status = storedStatus
	record.Revision = parsedRevision
	if deletedAt.Valid {
		value := deletedAt.Time
		record.DeletedAt = &value
		record.User.Status = accesscontrol.UserStatusDeleted
	}
	return record, nil
}
