package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	managementListAPIKeysQuery = `SELECT ` + apiKeyColumns + `
FROM access_api_keys
WHERE namespace_id = $1
  AND ($2 = '' OR CASE WHEN deleted_at IS NULL THEN status ELSE 'deleted' END = $2)
  AND ($3 = '' OR ($3 = 'user' AND owner_user_id = $4::uuid)
                OR ($3 = 'team' AND owner_team_id = $4::uuid))
  AND ($5 OR id = ANY($6::uuid[]) OR owner_user_id = ANY($7::uuid[])
          OR owner_team_id = ANY($8::uuid[]))
  AND ($9::timestamptz IS NULL OR created_at < $9 OR (created_at = $9 AND id > $10::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $11`
	managementSearchAPIKeysQuery = `SELECT ` + apiKeyColumns + `
FROM access_api_keys
WHERE namespace_id = $1
  AND ($2 = '' OR CASE WHEN deleted_at IS NULL THEN status ELSE 'deleted' END = $2)
  AND ($3 = '' OR ($3 = 'user' AND owner_user_id = $4::uuid)
                OR ($3 = 'team' AND owner_team_id = $4::uuid))
  AND ($5 OR id = ANY($6::uuid[]) OR owner_user_id = ANY($7::uuid[])
          OR owner_team_id = ANY($8::uuid[]))
  AND (lower(name) LIKE $9 ESCAPE E'\\' OR id::text LIKE $9 ESCAPE E'\\')
  AND ($10::timestamptz IS NULL OR created_at < $10 OR (created_at = $10 AND id > $11::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $12`
	managementListCredentialsQuery = `SELECT ` + credentialColumns + `
FROM access_api_key_credentials
WHERE namespace_id = $1 AND api_key_id = $2
  AND ($3 = '' OR CASE
       WHEN status <> 'revoked' AND expires_at IS NOT NULL AND expires_at <= clock_timestamp()
         THEN 'expired' ELSE status END = $3)
  AND ($4::timestamptz IS NULL OR created_at < $4 OR (created_at = $4 AND id > $5::uuid))
ORDER BY created_at DESC, id ASC
LIMIT $6`
	managementGetCredentialQuery = `SELECT ` + credentialColumns + `
FROM access_api_key_credentials
WHERE namespace_id = $1 AND api_key_id = $2 AND id = $3`
	managementGetActiveCredentialQuery = `SELECT ` + credentialColumns + `
FROM access_api_key_credentials
WHERE namespace_id = $1 AND api_key_id = $2 AND status = 'active'
  AND not_before <= clock_timestamp()
  AND (expires_at IS NULL OR expires_at > clock_timestamp())`
	managementLockUserRelationshipQuery = `SELECT id, namespace_id, email, display_name, status,
       created_at, updated_at
FROM access_users
WHERE namespace_id = $1 AND id = $2 AND deleted_at IS NULL
FOR KEY SHARE`
	managementLockTeamRelationshipQuery = `SELECT id, namespace_id, name, status, created_at, updated_at
FROM access_teams
WHERE namespace_id = $1 AND id = $2 AND deleted_at IS NULL
FOR KEY SHARE`
	managementLockMembershipRelationshipQuery = `SELECT namespace_id, team_id, user_id, role, status,
       created_at, updated_at
FROM access_team_memberships
WHERE namespace_id = $1 AND team_id = $2 AND user_id = $3
FOR KEY SHARE`
	managementUpdateAPIKeyQuery = `UPDATE access_api_keys
SET name = $4, owner_user_id = $5, owner_team_id = $6, context_team_id = $7,
    status = $8, expires_at = $9, policy_epoch = $10, delegation_epoch = $11,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + apiKeyColumns
	managementAdvanceAPIKeyRevisionQuery = `UPDATE access_api_keys
SET revision = revision + 1, updated_at = $4
WHERE namespace_id = $1 AND id = $2 AND revision = $3 AND deleted_at IS NULL
RETURNING ` + apiKeyColumns
	managementRevokePreviousCredentialQuery = `UPDATE access_api_key_credentials
SET status = 'revoked', revoked_at = $4,
    secret_ciphertext = NULL, ciphertext_nonce = NULL, kek_version = NULL
WHERE namespace_id = $1 AND api_key_id = $2 AND id = $3 AND status = 'active'`
	// #nosec G101 -- this constant is a parameterized query and contains no credential value.
	managementRevealCredentialQuery = `SELECT c.id, c.namespace_id, c.api_key_id, c.kid,
       c.secret_hmac, c.pepper_version, c.secret_ciphertext, c.ciphertext_nonce,
       c.kek_version, c.status, c.not_before, c.expires_at, c.revoked_at, c.created_at
FROM access_api_key_credentials c
JOIN access_api_keys k ON k.namespace_id = c.namespace_id AND k.id = c.api_key_id
WHERE c.namespace_id = $1 AND c.api_key_id = $2 AND c.id = $3
  AND k.deleted_at IS NULL AND k.status = 'active'
  AND (k.expires_at IS NULL OR k.expires_at > clock_timestamp())
  AND c.status IN ('active','retiring')
  AND c.not_before <= clock_timestamp()
  AND (c.expires_at IS NULL OR c.expires_at > clock_timestamp())
  AND c.secret_ciphertext IS NOT NULL`
	managementLockRevealCredentialQuery = managementRevealCredentialQuery + ` FOR UPDATE OF c, k`
	// #nosec G101 -- this constant is a parameterized count query and contains no credential value.
	managementCountUsableCredentialsQuery = `SELECT count(*)
FROM access_api_key_credentials
WHERE namespace_id = $1 AND api_key_id = $2
  AND status IN ('active','retiring')
  AND not_before <= clock_timestamp()
  AND (expires_at IS NULL OR expires_at > clock_timestamp())`
)

type apiKeyRepositoryAdapter struct{ store *Store }

func NewAPIKeyManagementRepository(store *Store) (apikeymanagement.Repository, error) {
	if store == nil || store.db == nil {
		return nil, apikeymanagement.ErrUnavailable
	}
	return &apiKeyRepositoryAdapter{store: store}, nil
}

func (adapter *apiKeyRepositoryAdapter) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	if adapter == nil || adapter.store == nil || codec == nil {
		return apikeymanagement.ErrUnavailable
	}
	if err := adapter.store.db.PingContext(ctx); err != nil {
		return err
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, adapter.store.db, codec)
}

func (adapter *apiKeyRepositoryAdapter) ReplaySecret(ctx context.Context, command managementcommand.Command) (apikeymanagement.StoredSecret, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, adapter.store.db, command)
	if err != nil || !found {
		return apikeymanagement.StoredSecret{}, found, err
	}
	return storedAPIKeySecret(stored)
}

func (adapter *apiKeyRepositoryAdapter) ReplayMutation(ctx context.Context, command managementcommand.Command) (apikeymanagement.MutationResult, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, adapter.store.db, command)
	if err != nil || !found {
		return apikeymanagement.MutationResult{}, found, err
	}
	result, err := storedAPIKeyMutation(stored)
	if err != nil {
		return apikeymanagement.MutationResult{}, true, err
	}
	key, err := adapter.GetKey(ctx, command.Scope.NamespaceID, result.ResourceID)
	if err != nil {
		return apikeymanagement.MutationResult{}, true, err
	}
	return apikeymanagement.MutationResult{
		Key: key, HTTPStatus: result.ResponseStatus, Replayed: true,
	}, true, nil
}

func (adapter *apiKeyRepositoryAdapter) GetKey(ctx context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	key, err := adapter.store.GetAPIKey(ctx, accesscontrol.NamespaceID(namespaceID), accesscontrol.APIKeyID(keyID))
	return key, mapAPIKeyReadError(err, "get API key")
}

func (adapter *apiKeyRepositoryAdapter) ListKeys(ctx context.Context, query apikeymanagement.KeyQuery) (apikeymanagement.RepositoryPage[accesscontrol.APIKey], error) {
	if query.Limit < 1 || query.Limit > 200 {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, apikeymanagement.ErrInvalidRequest
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, apikeymanagement.ErrInvalidRequest
	}
	normalizedSearch, err := managementsearch.Normalize(query.Search)
	if err != nil || normalizedSearch != query.Search {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, apikeymanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.APIKeyIDs) == 0 &&
		len(query.Scope.UserIDs) == 0 && len(query.Scope.TeamIDs) == 0 {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{Items: []accesscontrol.APIKey{}}, nil
	}
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	var ownerID any
	if query.OwnerID != "" {
		ownerID = query.OwnerID
	}
	var rows *sql.Rows
	if query.Search == "" {
		rows, err = adapter.store.db.QueryContext(ctx, managementListAPIKeysQuery, query.NamespaceID,
			query.Status, query.OwnerKind, ownerID, query.Scope.All, pq.Array(query.Scope.APIKeyIDs),
			pq.Array(query.Scope.UserIDs), pq.Array(query.Scope.TeamIDs), afterTime, afterID, query.Limit+1)
	} else {
		rows, err = adapter.store.db.QueryContext(ctx, managementSearchAPIKeysQuery, query.NamespaceID,
			query.Status, query.OwnerKind, ownerID, query.Scope.All, pq.Array(query.Scope.APIKeyIDs),
			pq.Array(query.Scope.UserIDs), pq.Array(query.Scope.TeamIDs),
			managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit+1)
	}
	if err != nil {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, fmt.Errorf("list API keys: %w", err)
	}
	defer rows.Close()
	items := make([]accesscontrol.APIKey, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanAPIKey(rows)
		if err != nil {
			return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, fmt.Errorf("scan API key: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{}, err
	}
	hasMore := len(items) > query.Limit
	if hasMore {
		items = items[:query.Limit]
	}
	return apikeymanagement.RepositoryPage[accesscontrol.APIKey]{Items: items, HasMore: hasMore}, nil
}

func (adapter *apiKeyRepositoryAdapter) CreateKey(ctx context.Context, mutation apikeymanagement.CreateMutation) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(mutation.Actor, "api_key.create", "Create API key.", nil)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	if len(mutation.AccessBindings) > 12 {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		return createAPIKeyInTransaction(ctx, tx, mutation, meta)
	})
}

func (adapter *apiKeyRepositoryAdapter) UpdateKey(ctx context.Context, key accesscontrol.APIKey, expected uint64, actor apikeymanagement.Actor, action string) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(actor, action, "Update API key.", nil)
	if err != nil || expected == 0 {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		if err := validateAPIKeyRelationshipsTx(ctx, tx, key); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		ownerUser, ownerTeam := apiKeyOwnerValues(key.Owner)
		updated, err := scanAPIKey(tx.QueryRowContext(ctx, managementUpdateAPIKeyQuery,
			key.NamespaceID, key.ID, expected, key.Name, ownerUser, ownerTeam, nullableTeamID(key.ContextTeamID),
			key.Status, key.ExpiresAt, key.PolicyEpoch, key.DelegationEpoch))
		if err != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCAS(err, "update API key")
		}
		if _, err := appendMutationRecords(ctx, tx, key.NamespaceID, outboxMutation{
			AggregateType: "api_key",
			AggregateID:   string(key.ID), AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 200}, nil
	})
}

func (adapter *apiKeyRepositoryAdapter) UpdateKeyAction(ctx context.Context, mutation apikeymanagement.UpdateMutation) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(mutation.Actor, mutation.Action, mutation.Reason, nil)
	if err != nil || mutation.ExpectedRevision == 0 {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		stored, replayed, updateKeyActionErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if updateKeyActionErr != nil {
			return apikeymanagement.MutationResult{}, updateKeyActionErr
		}
		if replayed {
			result, err := storedAPIKeyMutation(stored)
			if err != nil || result.ResourceID != string(mutation.Key.ID) {
				if err != nil {
					return apikeymanagement.MutationResult{}, err
				}
				return apikeymanagement.MutationResult{}, apikeymanagement.ErrUnavailable
			}
			key, err := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery, mutation.Key.NamespaceID, result.ResourceID))
			if err != nil {
				return apikeymanagement.MutationResult{}, mapAPIKeyReadError(err, "read replayed API key")
			}
			return apikeymanagement.MutationResult{
				Key: key, HTTPStatus: result.ResponseStatus, Replayed: true,
			}, nil
		}
		if err := validateAPIKeyRelationshipsTx(ctx, tx, mutation.Key); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		ownerUser, ownerTeam := apiKeyOwnerValues(mutation.Key.Owner)
		updated, updateKeyActionErr := scanAPIKey(tx.QueryRowContext(ctx, managementUpdateAPIKeyQuery,
			mutation.Key.NamespaceID, mutation.Key.ID, mutation.ExpectedRevision, mutation.Key.Name,
			ownerUser, ownerTeam, nullableTeamID(mutation.Key.ContextTeamID), mutation.Key.Status,
			mutation.Key.ExpiresAt, mutation.Key.PolicyEpoch, mutation.Key.DelegationEpoch))
		if updateKeyActionErr != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCAS(updateKeyActionErr, "update API key")
		}
		if _, err := appendMutationRecords(ctx, tx, mutation.Key.NamespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(mutation.Key.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		if err := commandpostgres.CompleteResource(ctx, tx, mutation.Command, managementcommand.ResourceResult{
			ResourceType: "api_key", ResourceID: string(updated.ID),
			ResourceRevision: uint64(updated.Revision), ResponseStatus: 200,
		}); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 200}, nil
	})
}

func (adapter *apiKeyRepositoryAdapter) DeleteKey(ctx context.Context, namespaceID, keyID string, expected uint64, actor apikeymanagement.Actor) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(actor, "api_key.delete", "Delete API key.", nil)
	if err != nil || expected == 0 {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		deleted, err := scanAPIKey(tx.QueryRowContext(ctx, softDeleteAPIKeyQuery, namespaceID, keyID, expected))
		if err != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCAS(err, "delete API key")
		}
		if _, err := tx.ExecContext(ctx, revokeAllCredentialsQuery, namespaceID, keyID); err != nil {
			return apikeymanagement.MutationResult{}, fmt.Errorf("revoke deleted API-key credentials: %w", err)
		}
		if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespaceID), outboxMutation{
			AggregateType: "api_key", AggregateID: keyID, AggregateRevision: deleted.Revision, Operation: outboxDeleted,
		}, meta); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		return apikeymanagement.MutationResult{Key: deleted, HTTPStatus: 204}, nil
	})
}

func (adapter *apiKeyRepositoryAdapter) ListCredentials(ctx context.Context, query apikeymanagement.CredentialQuery) (apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot], error) {
	if query.Limit < 1 || query.Limit > 200 {
		return apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot]{}, apikeymanagement.ErrInvalidRequest
	}
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	rows, err := adapter.store.db.QueryContext(ctx, managementListCredentialsQuery, query.NamespaceID, query.KeyID,
		query.Status, afterTime, afterID, query.Limit+1)
	if err != nil {
		return apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot]{}, fmt.Errorf("list credentials: %w", err)
	}
	defer rows.Close()
	items := make([]apikeymanagement.RevealSnapshot, 0, query.Limit+1)
	for rows.Next() {
		record, err := scanCredential(rows)
		if err != nil {
			return apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot]{}, err
		}
		items = append(items, apikeymanagement.RevealSnapshot{NamespaceID: string(record.NamespaceID), Credential: record.Credential})
	}
	if err := rows.Err(); err != nil {
		return apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot]{}, err
	}
	hasMore := len(items) > query.Limit
	if hasMore {
		items = items[:query.Limit]
	}
	return apikeymanagement.RepositoryPage[apikeymanagement.RevealSnapshot]{Items: items, HasMore: hasMore}, nil
}

func (adapter *apiKeyRepositoryAdapter) GetCredential(ctx context.Context, namespaceID, keyID, credentialID string) (apikeymanagement.RevealSnapshot, error) {
	record, err := scanCredential(adapter.store.db.QueryRowContext(ctx, managementGetCredentialQuery,
		namespaceID, keyID, credentialID))
	if err != nil {
		return apikeymanagement.RevealSnapshot{}, mapCredentialUnavailable(err)
	}
	return apikeymanagement.RevealSnapshot{NamespaceID: namespaceID, Credential: record.Credential}, nil
}

func (adapter *apiKeyRepositoryAdapter) GetActiveCredential(ctx context.Context, namespaceID, keyID string) (apikeymanagement.RevealSnapshot, error) {
	record, err := scanCredential(adapter.store.db.QueryRowContext(ctx, managementGetActiveCredentialQuery,
		namespaceID, keyID))
	if err != nil {
		return apikeymanagement.RevealSnapshot{}, mapCredentialUnavailable(err)
	}
	return apikeymanagement.RevealSnapshot{NamespaceID: namespaceID, Credential: record.Credential}, nil
}

func (adapter *apiKeyRepositoryAdapter) RotateCredential(ctx context.Context, mutation apikeymanagement.RotateMutation) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(mutation.Actor, "api_key.credential.rotate", "Rotate API-key credential.", nil)
	if err != nil {
		return apikeymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		if replay, found, err := lockAPIKeySecretCommand(ctx, tx, mutation.Command); err != nil || found {
			if err != nil {
				return apikeymanagement.MutationResult{}, err
			}
			key, err := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery, mutation.NamespaceID, replay.Result.ResourceID))
			if err != nil {
				return apikeymanagement.MutationResult{}, mapAPIKeyReadError(err, "read replayed API key")
			}
			return apikeymanagement.MutationResult{Key: key, HTTPStatus: replay.Result.ResponseStatus, Replayed: true, Stored: &replay}, nil
		}
		updated, err := scanAPIKey(tx.QueryRowContext(ctx, managementAdvanceAPIKeyRevisionQuery,
			mutation.NamespaceID, mutation.KeyID, mutation.ExpectedRevision, mutation.Credential.CreatedAt))
		if err != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCAS(err, "rotate API-key credential")
		}
		var result sql.Result
		if mutation.RetireAt == nil {
			result, err = tx.ExecContext(ctx, managementRevokePreviousCredentialQuery,
				mutation.NamespaceID, mutation.KeyID, mutation.PreviousCredentialID, mutation.Credential.CreatedAt)
		} else {
			result, err = tx.ExecContext(ctx, retireCredentialQuery,
				mutation.NamespaceID, mutation.KeyID, mutation.PreviousCredentialID, *mutation.RetireAt)
		}
		if err != nil {
			return apikeymanagement.MutationResult{}, fmt.Errorf("retire prior credential: %w", err)
		}
		if err := requireOneRow(result, apikeymanagement.ErrCredentialUnavailable); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		if err := insertCredential(ctx, tx, accesscontrol.NamespaceID(mutation.NamespaceID), mutation.Credential); err != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCreateError(err, "insert rotated credential")
		}
		if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(mutation.NamespaceID), outboxMutation{
			AggregateType: "api_key", AggregateID: mutation.KeyID, AggregateRevision: updated.Revision,
			Operation: outboxCredentialRotated, References: map[string]string{
				"credentialId":        string(mutation.Credential.ID),
				"retiredCredentialId": mutation.PreviousCredentialID,
			},
		}, meta); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command,
			managementcommand.ResourceResult{
				ResourceType: "api_key", ResourceID: mutation.KeyID,
				ResourceRevision: uint64(updated.Revision), ResponseStatus: 200,
			},
			managementcommand.SecretResponse{
				Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
				KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
			}); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 200}, nil
	})
}

func (adapter *apiKeyRepositoryAdapter) RevokeCredential(ctx context.Context, namespaceID, keyID, credentialID string, expected uint64, actor apikeymanagement.Actor) (apikeymanagement.MutationResult, error) {
	meta, err := apiKeyMutationMeta(actor, "api_key.credential.revoke", "Delete API-key credential.", nil)
	if err != nil || expected == 0 {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	expectedRevision, err := postgresInt64(expected, "expected API-key revision")
	if err != nil {
		return apikeymanagement.MutationResult{}, apikeymanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (apikeymanagement.MutationResult, error) {
		key, revokeCredentialErr := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery+" FOR UPDATE", namespaceID, keyID))
		if revokeCredentialErr != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyReadError(revokeCredentialErr, "lock API key")
		}
		if uint64(key.Revision) != expected {
			return apikeymanagement.MutationResult{}, apikeymanagement.ErrRevisionConflict
		}
		if key.Status == accesscontrol.APIKeyStatusActive {
			var count int
			if err := tx.QueryRowContext(ctx, managementCountUsableCredentialsQuery, namespaceID, keyID).Scan(&count); err != nil {
				return apikeymanagement.MutationResult{}, err
			}
			if count <= 1 {
				return apikeymanagement.MutationResult{}, apikeymanagement.ErrLastActiveCredential
			}
		}
		updated, revokeCredentialErr := advanceAPIKeyRevision(
			ctx, tx, accesscontrol.NamespaceID(namespaceID), accesscontrol.APIKeyID(keyID), expectedRevision,
		)
		if revokeCredentialErr != nil {
			return apikeymanagement.MutationResult{}, mapAPIKeyCAS(revokeCredentialErr, "revoke API-key credential")
		}
		result, revokeCredentialErr := tx.ExecContext(ctx, revokeCredentialQuery, namespaceID, keyID, credentialID)
		if revokeCredentialErr != nil {
			return apikeymanagement.MutationResult{}, revokeCredentialErr
		}
		if err := requireOneRow(result, apikeymanagement.ErrCredentialUnavailable); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespaceID), outboxMutation{
			AggregateType: "api_key", AggregateID: keyID, AggregateRevision: updated.Revision,
			Operation: outboxCredentialRevoked, References: map[string]string{"credentialId": credentialID},
		}, meta); err != nil {
			return apikeymanagement.MutationResult{}, err
		}
		return apikeymanagement.MutationResult{Key: updated, HTTPStatus: 204}, nil
	})
}

func (adapter *apiKeyRepositoryAdapter) GetRevealSnapshot(ctx context.Context, namespaceID, keyID, credentialID string) (apikeymanagement.RevealSnapshot, error) {
	record, err := scanCredential(adapter.store.db.QueryRowContext(ctx, managementRevealCredentialQuery, namespaceID, keyID, credentialID))
	if err != nil {
		return apikeymanagement.RevealSnapshot{}, mapCredentialUnavailable(err)
	}
	return apikeymanagement.RevealSnapshot{NamespaceID: namespaceID, Credential: record.Credential}, nil
}

func (adapter *apiKeyRepositoryAdapter) RecordReveal(ctx context.Context, snapshot apikeymanagement.RevealSnapshot, actor apikeymanagement.Actor) error {
	meta, recordRevealErr := apiKeyMutationMeta(actor, "api_key.credential.reveal", "Reveal API-key credential.",
		map[string]string{"credentialId": string(snapshot.Credential.ID)})
	if recordRevealErr != nil {
		return recordRevealErr
	}
	_, recordRevealErr = inTransaction(ctx, adapter.store, func(tx *sql.Tx) (struct{}, error) {
		current, err := scanCredential(tx.QueryRowContext(ctx, managementLockRevealCredentialQuery,
			snapshot.NamespaceID, snapshot.Credential.APIKeyID, snapshot.Credential.ID))
		if err != nil {
			return struct{}{}, mapCredentialUnavailable(err)
		}
		if current.Credential.KID != snapshot.Credential.KID ||
			!bytes.Equal(current.Credential.SecretCiphertext, snapshot.Credential.SecretCiphertext) ||
			!bytes.Equal(current.Credential.CiphertextNonce, snapshot.Credential.CiphertextNonce) ||
			current.Credential.KEKVersion != snapshot.Credential.KEKVersion {
			return struct{}{}, apikeymanagement.ErrCredentialUnavailable
		}
		key, err := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery, snapshot.NamespaceID, snapshot.Credential.APIKeyID))
		if err != nil {
			return struct{}{}, mapAPIKeyReadError(err, "read API key for reveal audit")
		}
		if err := appendObservedAuditEvent(ctx, tx, accesscontrol.NamespaceID(snapshot.NamespaceID), "api_key",
			string(snapshot.Credential.APIKeyID), key.Revision, meta); err != nil {
			return struct{}{}, err
		}
		return struct{}{}, nil
	})
	return recordRevealErr
}

func validateAPIKeyRelationshipsTx(ctx context.Context, tx *sql.Tx, key accesscontrol.APIKey) error {
	relationships := accesscontrol.APIKeyRelationships{}
	switch key.Owner.Kind {
	case accesscontrol.SubjectKindUser:
		var user accesscontrol.User
		if err := tx.QueryRowContext(ctx, managementLockUserRelationshipQuery, key.NamespaceID, key.Owner.ID).Scan(
			&user.ID, &user.NamespaceID, &user.Email, &user.DisplayName, &user.Status, &user.CreatedAt, &user.UpdatedAt); err != nil {
			return mapRelationshipError(err)
		}
		relationships.OwnerUser = &user
		if key.ContextTeamID != "" {
			var team accesscontrol.Team
			if err := tx.QueryRowContext(ctx, managementLockTeamRelationshipQuery, key.NamespaceID, key.ContextTeamID).Scan(
				&team.ID, &team.NamespaceID, &team.Name, &team.Status, &team.CreatedAt, &team.UpdatedAt); err != nil {
				return mapRelationshipError(err)
			}
			var membership accesscontrol.TeamMembership
			if err := tx.QueryRowContext(ctx, managementLockMembershipRelationshipQuery, key.NamespaceID, key.ContextTeamID, key.Owner.ID).Scan(
				&membership.NamespaceID, &membership.TeamID, &membership.UserID, &membership.Role, &membership.Status,
				&membership.CreatedAt, &membership.UpdatedAt); err != nil {
				return mapRelationshipError(err)
			}
			relationships.ContextTeam, relationships.ContextMembership = &team, &membership
		}
	case accesscontrol.SubjectKindTeam:
		var team accesscontrol.Team
		if err := tx.QueryRowContext(ctx, managementLockTeamRelationshipQuery, key.NamespaceID, key.Owner.ID).Scan(
			&team.ID, &team.NamespaceID, &team.Name, &team.Status, &team.CreatedAt, &team.UpdatedAt); err != nil {
			return mapRelationshipError(err)
		}
		relationships.OwnerTeam = &team
	default:
		return apikeymanagement.ErrInvalidRequest
	}
	if err := accesscontrol.ValidateAPIKeyRelationships(key, relationships); err != nil {
		return apikeymanagement.ErrInvalidRequest
	}
	return nil
}

func apiKeyMutationMeta(actor apikeymanagement.Actor, action, reason string, details map[string]string) (MutationMeta, error) {
	if details == nil {
		details = make(map[string]string)
	}
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(actor.ActorChain[index])
	}
	meta := MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: action, Reason: reason, Details: AuditDetails(details),
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationMeta{}, apikeymanagement.ErrInvalidRequest
	}
	return meta, nil
}

func apiKeyPolicyActor(actor apikeymanagement.Actor) policymanagement.Actor {
	return policymanagement.Actor{
		PrincipalID: actor.PrincipalID,
		ActorChain:  append([]string(nil), actor.ActorChain...),
		RequestID:   actor.RequestID,
		SourceIP:    actor.SourceIP,
	}
}

func appendAPIKeyCreateMutationRecords(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	mutations []compoundMutation,
) error {
	if len(mutations) == 1 {
		item := mutations[0]
		_, err := appendMutationRecords(ctx, tx, namespaceID, item.Mutation, item.Meta)
		return err
	}
	_, err := appendCompoundMutationRecords(ctx, tx, string(namespaceID), mutations)
	return err
}

func lockAPIKeySecretCommand(ctx context.Context, tx *sql.Tx, command managementcommand.Command) (apikeymanagement.StoredSecret, bool, error) {
	stored, replayed, err := commandpostgres.Lock(ctx, tx, command)
	if err != nil || !replayed {
		return apikeymanagement.StoredSecret{}, false, err
	}
	result, _, err := storedAPIKeySecret(stored)
	return result, true, err
}

func storedAPIKeySecret(stored managementcommand.StoredResult) (apikeymanagement.StoredSecret, bool, error) {
	if stored.Resource == nil || stored.Secret == nil || stored.Resource.ResourceType != "api_key" {
		return apikeymanagement.StoredSecret{}, true, apikeymanagement.ErrUnavailable
	}
	return apikeymanagement.StoredSecret{Result: *stored.Resource, Secret: *stored.Secret}, true, nil
}

func storedAPIKeyMutation(stored managementcommand.StoredResult) (managementcommand.ResourceResult, error) {
	if stored.Resource == nil || stored.Secret != nil || stored.Resource.ResourceType != "api_key" {
		return managementcommand.ResourceResult{}, apikeymanagement.ErrUnavailable
	}
	return *stored.Resource, nil
}

func mapAPIKeyReadError(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) || errors.Is(err, ErrNotFound) {
		return apikeymanagement.ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapAPIKeyCreateError(err error, action string) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return ErrAlreadyExists
	}
	if err != nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	return nil
}

func mapAPIKeyCAS(err error, action string) error {
	if errors.Is(err, sql.ErrNoRows) || errors.Is(err, ErrRevisionConflict) {
		return apikeymanagement.ErrRevisionConflict
	}
	return mapAPIKeyCreateError(err, action)
}

func mapRelationshipError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return apikeymanagement.ErrInvalidRequest
	}
	return err
}

func mapCredentialUnavailable(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return apikeymanagement.ErrCredentialUnavailable
	}
	return err
}

var _ apikeymanagement.Repository = (*apiKeyRepositoryAdapter)(nil)
