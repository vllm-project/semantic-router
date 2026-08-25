package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	// #nosec G101 -- this is a database column list and contains no API-key value.
	apiKeyColumns = `id, namespace_id, name, owner_user_id, owner_team_id,
       context_team_id, status, expires_at, policy_epoch, delegation_epoch,
       revision, last_used_at, created_at, updated_at, deleted_at`
	getAPIKeyQuery = `SELECT ` + apiKeyColumns + `
FROM access_api_keys
WHERE namespace_id = $1 AND id = $2`
	insertAPIKeyQuery = `INSERT INTO access_api_keys
  (id, namespace_id, name, owner_user_id, owner_team_id, context_team_id,
   status, expires_at, policy_epoch, delegation_epoch, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING ` + apiKeyColumns
	updateAPIKeyQuery = `UPDATE access_api_keys
SET name = $4, owner_user_id = $5, owner_team_id = $6, context_team_id = $7,
    status = $8, expires_at = $9, policy_epoch = $10, delegation_epoch = $11,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + apiKeyColumns
	advanceAPIKeyRevisionQuery = `UPDATE access_api_keys
SET revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + apiKeyColumns
	softDeleteAPIKeyQuery = `UPDATE access_api_keys
SET status = 'disabled', deleted_at = clock_timestamp(),
    policy_epoch = policy_epoch + 1, delegation_epoch = delegation_epoch + 1,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + apiKeyColumns
	revokeAllCredentialsQuery = `UPDATE access_api_key_credentials
SET status = 'revoked', revoked_at = COALESCE(revoked_at, clock_timestamp()),
    secret_ciphertext = NULL, ciphertext_nonce = NULL, kek_version = NULL
WHERE namespace_id = $1 AND api_key_id = $2 AND status IN ('active', 'retiring')`
)

func (s *Store) GetAPIKey(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.APIKeyID,
) (accesscontrol.APIKey, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return accesscontrol.APIKey{}, err
	}
	key, err := scanAPIKey(s.db.QueryRowContext(ctx, getAPIKeyQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return accesscontrol.APIKey{}, ErrNotFound
	}
	if err != nil {
		return accesscontrol.APIKey{}, fmt.Errorf("get API key: %w", err)
	}
	return key, nil
}

func (s *Store) CreateAPIKey(
	ctx context.Context,
	key accesscontrol.APIKey,
	credential accesscontrol.CredentialVersion,
	meta MutationMeta,
) (MutationResult[accesscontrol.APIKey], error) {
	if err := validateNewAPIKey(key, credential); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.APIKey], error) {
		if _, err := tx.ExecContext(ctx, insertSubjectQuery,
			key.NamespaceID, key.ID, accesscontrol.SubjectKindAPIKey, key.CreatedAt); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("insert API-key subject: %w", err)
		}
		ownerUser, ownerTeam := apiKeyOwnerValues(key.Owner)
		created, createAPIKeyErr := scanAPIKey(tx.QueryRowContext(ctx, insertAPIKeyQuery,
			key.ID, key.NamespaceID, key.Name, ownerUser, ownerTeam, nullableTeamID(key.ContextTeamID),
			key.Status, key.ExpiresAt, key.PolicyEpoch, key.DelegationEpoch, key.Revision,
			key.CreatedAt, key.UpdatedAt))
		if createAPIKeyErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("insert API key: %w", createAPIKeyErr)
		}
		if err := insertCredential(ctx, tx, key.NamespaceID, credential); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, err
		}
		receipt, createAPIKeyErr := appendMutationRecords(ctx, tx, key.NamespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(key.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
			References: map[string]string{"credentialId": string(credential.ID)},
		}, meta)
		if createAPIKeyErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, createAPIKeyErr
		}
		return MutationResult[accesscontrol.APIKey]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateAPIKey(
	ctx context.Context,
	key accesscontrol.APIKey,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[accesscontrol.APIKey], error) {
	if err := validateUpdatedAPIKey(key, expected); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.APIKey], error) {
		ownerUser, ownerTeam := apiKeyOwnerValues(key.Owner)
		updated, err := scanAPIKey(tx.QueryRowContext(ctx, updateAPIKeyQuery,
			key.NamespaceID, key.ID, expectedRevision, key.Name, ownerUser, ownerTeam,
			nullableTeamID(key.ContextTeamID), key.Status, key.ExpiresAt,
			key.PolicyEpoch, key.DelegationEpoch))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.APIKey]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("update API key: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, key.NamespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(key.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.APIKey]{}, err
		}
		return MutationResult[accesscontrol.APIKey]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) SoftDeleteAPIKey(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.APIKeyID,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[accesscontrol.APIKey], error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.APIKey], error) {
		deleted, softDeleteAPIKeyErr := scanAPIKey(tx.QueryRowContext(ctx, softDeleteAPIKeyQuery, namespaceID, id, expectedRevision))
		if errors.Is(softDeleteAPIKeyErr, sql.ErrNoRows) {
			return MutationResult[accesscontrol.APIKey]{}, ErrRevisionConflict
		}
		if softDeleteAPIKeyErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("soft-delete API key: %w", softDeleteAPIKeyErr)
		}
		if _, err := tx.ExecContext(ctx, revokeAllCredentialsQuery, namespaceID, id); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("revoke deleted key credentials: %w", err)
		}
		receipt, softDeleteAPIKeyErr := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(id),
			AggregateRevision: deleted.Revision, Operation: outboxDeleted,
		}, meta)
		if softDeleteAPIKeyErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, softDeleteAPIKeyErr
		}
		return MutationResult[accesscontrol.APIKey]{Value: deleted, Receipt: receipt}, nil
	})
}

func validateNewAPIKey(key accesscontrol.APIKey, credential accesscontrol.CredentialVersion) error {
	if err := validateUpdatedAPIKey(key, key.Revision); err != nil {
		return err
	}
	if key.Revision != 1 {
		return fmt.Errorf("new API key revision must be 1")
	}
	if credential.APIKeyID != key.ID || credential.Status != accesscontrol.CredentialStatusActive {
		return fmt.Errorf("initial credential must be active and belong to the logical key")
	}
	return validateCredentialForWrite(key.NamespaceID, credential)
}

func validateUpdatedAPIKey(key accesscontrol.APIKey, expected accesscontrol.Revision) error {
	if err := key.Validate(); err != nil {
		return err
	}
	if key.Status == accesscontrol.APIKeyStatusDeleted || key.DeletedAt != nil {
		return fmt.Errorf("deleted lifecycle is managed only by SoftDeleteAPIKey")
	}
	if key.Revision != expected {
		return fmt.Errorf("API-key revision must match expected revision")
	}
	if err := validateIdentityIDs(key.NamespaceID, string(key.ID)); err != nil {
		return err
	}
	if err := validateUUID("owner id", string(key.Owner.ID)); err != nil {
		return err
	}
	if key.ContextTeamID != "" {
		return validateUUID("context team id", string(key.ContextTeamID))
	}
	return nil
}

func apiKeyOwnerValues(owner accesscontrol.SubjectRef) (any, any) {
	if owner.Kind == accesscontrol.SubjectKindUser {
		return string(owner.ID), nil
	}
	return nil, string(owner.ID)
}

func nullableTeamID(id accesscontrol.TeamID) any {
	if id == "" {
		return nil
	}
	return string(id)
}

func scanAPIKey(scanner rowScanner) (accesscontrol.APIKey, error) {
	var key accesscontrol.APIKey
	var ownerUser, ownerTeam, contextTeam sql.NullString
	var storedStatus accesscontrol.APIKeyStatus
	var expiresAt, lastUsedAt, deletedAt sql.NullTime
	var policyEpoch, delegationEpoch, revision int64
	if err := scanner.Scan(
		&key.ID, &key.NamespaceID, &key.Name, &ownerUser, &ownerTeam, &contextTeam,
		&storedStatus, &expiresAt, &policyEpoch, &delegationEpoch, &revision, &lastUsedAt,
		&key.CreatedAt, &key.UpdatedAt, &deletedAt,
	); err != nil {
		return accesscontrol.APIKey{}, err
	}
	if err := populateAPIKeyIdentity(&key, ownerUser, ownerTeam, contextTeam); err != nil {
		return accesscontrol.APIKey{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.APIKey{}, err
	}
	if policyEpoch <= 0 || delegationEpoch <= 0 {
		return accesscontrol.APIKey{}, fmt.Errorf("database returned invalid API-key epoch")
	}
	key.Status = storedStatus
	key.PolicyEpoch = uint64(policyEpoch)
	key.DelegationEpoch = uint64(delegationEpoch)
	key.Revision = parsedRevision
	key.ExpiresAt = nullTimePointer(expiresAt)
	key.LastUsedAt = nullTimePointer(lastUsedAt)
	key.DeletedAt = nullTimePointer(deletedAt)
	if key.DeletedAt != nil {
		key.Status = accesscontrol.APIKeyStatusDeleted
	}
	return key, nil
}

func populateAPIKeyIdentity(key *accesscontrol.APIKey, ownerUser, ownerTeam, contextTeam sql.NullString) error {
	switch {
	case ownerUser.Valid && !ownerTeam.Valid:
		key.Owner = accesscontrol.SubjectRef{
			NamespaceID: key.NamespaceID, ID: accesscontrol.SubjectID(ownerUser.String), Kind: accesscontrol.SubjectKindUser,
		}
	case ownerTeam.Valid && !ownerUser.Valid:
		key.Owner = accesscontrol.SubjectRef{
			NamespaceID: key.NamespaceID, ID: accesscontrol.SubjectID(ownerTeam.String), Kind: accesscontrol.SubjectKindTeam,
		}
	default:
		return fmt.Errorf("database returned invalid API-key ownership")
	}
	if contextTeam.Valid {
		key.ContextTeamID = accesscontrol.TeamID(contextTeam.String)
	}
	return nil
}

func nullTimePointer(value sql.NullTime) *time.Time {
	if !value.Valid {
		return nil
	}
	copy := value.Time
	return &copy
}
