package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func (store *Store) ListToolCredentials(
	ctx context.Context, namespaceID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.ToolCredential], returnErr error) {
	ids := scopedIDs(query.Scope, accesscontrol.ScopeResourceAgentToolCredential)
	statement := credentialSelect + `
 WHERE namespace_id=$1 AND status<>'deleted' AND ($2 OR id=ANY($3::uuid[]))
	AND ($4='' OR lower(name) LIKE $4 ESCAPE '\')
	AND ($5::timestamptz IS NULL OR (created_at,id)<($5,$6::uuid))
 ORDER BY created_at DESC,id DESC LIMIT $7`
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.Timestamp, query.After.ID
	}
	rows, err := store.db.QueryContext(
		ctx, statement, namespaceID, query.Scope.All, pq.Array(ids),
		managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit,
	)
	if err != nil {
		return agentmanagement.ListResult[agentmanagement.ToolCredential]{}, fmt.Errorf("list Agent Tool credentials: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.ToolCredential, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanToolCredential(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.ToolCredential]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.ToolCredential]{}, fmt.Errorf("iterate Agent Tool credentials: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.ToolCredential]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetToolCredential(
	ctx context.Context, namespaceID, id string,
) (agentmanagement.ToolCredential, error) {
	return scanToolCredential(store.db.QueryRowContext(ctx, credentialSelect+`
 WHERE namespace_id=$1 AND id=$2 AND status<>'deleted'`, namespaceID, id))
}

func (store *Store) ResolveToolCredentialSecret(
	ctx context.Context, namespaceID, credentialID, versionID string,
) (agentmanagement.ToolCredentialSecret, error) {
	statement := `SELECT version.id::text,version.secret_ciphertext,version.ciphertext_nonce,
       version.kek_version,version.expires_at
FROM agent_tool_credentials credential
JOIN agent_tool_credential_versions version
  ON version.namespace_id=credential.namespace_id AND version.credential_id=credential.id
WHERE credential.namespace_id=$1 AND credential.id=$2 AND credential.status='active'
  AND version.id=COALESCE(NULLIF($3,'')::uuid,credential.active_version_id)
  AND version.status IN ('active','retiring')
  AND version.not_before<=clock_timestamp()
  AND (version.expires_at IS NULL OR version.expires_at>clock_timestamp())`
	var result agentmanagement.ToolCredentialSecret
	var expires sql.NullTime
	err := store.db.QueryRowContext(ctx, statement, namespaceID, credentialID, versionID).Scan(
		&result.VersionID, &result.Secret.Ciphertext, &result.Secret.Nonce,
		&result.Secret.KEKVersion, &expires,
	)
	if err != nil {
		return agentmanagement.ToolCredentialSecret{}, mapNotFound(err)
	}
	result.CredentialID = credentialID
	if expires.Valid {
		value := expires.Time.UTC()
		result.ExpiresAt = &value
	}
	return result, nil
}

func (store *Store) CreateToolCredential(
	ctx context.Context, namespaceID, id, name string, secret agentmanagement.EncryptedSecret,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	if err := validateEncryptedSecret(secret); err != nil {
		return agentmanagement.ResourceMutationResult{}, err
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentToolCredentialResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		versionID := newResourceID()
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_credentials
  (id,namespace_id,name,status,active_version_id,revision)
VALUES ($1,$2,$3,'active',$4,1)`, id, namespaceID, name, versionID); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_credential_versions
  (id,namespace_id,credential_id,secret_ciphertext,ciphertext_nonce,kek_version,status,not_before)
VALUES ($1,$2,$3,$4,$5,$6,'active',clock_timestamp())`, versionID, namespaceID, id,
			secret.Ciphertext, secret.Nonce, secret.KEKVersion); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		return completeResourceCommand(ctx, tx, mutation, agentToolCredentialResourceType, id, 1, 201)
	})
}

func (store *Store) PatchToolCredential(
	ctx context.Context, namespaceID, id string, expected int64, patch agentmanagement.ToolCredentialPatch,
	_ agentmanagement.MutationContext,
) (agentmanagement.ToolCredential, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ToolCredential, error) {
		current, err := lockToolCredential(ctx, tx, namespaceID, id, expected)
		if err != nil {
			return agentmanagement.ToolCredential{}, err
		}
		name, status := current.Name, current.Status
		if patch.Name != nil {
			name = strings.TrimSpace(*patch.Name)
		}
		if patch.Status != nil {
			status = *patch.Status
		}
		if name == "" || len(name) > 160 || (status != agentmanagement.StatusActive && status != agentmanagement.StatusDisabled) {
			return agentmanagement.ToolCredential{}, agentmanagement.ErrInvalid
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_tool_credentials
SET name=$4,status=$5,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`, namespaceID, id, expected, name, status)
		if err != nil {
			return agentmanagement.ToolCredential{}, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.ToolCredential{}, err
		}
		return scanToolCredential(tx.QueryRowContext(ctx, credentialSelect+` WHERE namespace_id=$1 AND id=$2`, namespaceID, id))
	})
}

func (store *Store) RotateToolCredential(
	ctx context.Context, namespaceID, id string, expected int64, secret agentmanagement.EncryptedSecret,
	retireAt time.Time, mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	if err := validateEncryptedSecret(secret); err != nil || retireAt.IsZero() {
		return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrInvalid
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentToolCredentialResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		current, rotateToolCredentialErr := lockToolCredential(ctx, tx, namespaceID, id, expected)
		if rotateToolCredentialErr != nil {
			return agentmanagement.ResourceMutationResult{}, rotateToolCredentialErr
		}
		if current.Status != agentmanagement.StatusActive {
			return agentmanagement.ResourceMutationResult{}, agentmanagement.ErrDenied
		}
		newVersionID := newResourceID()
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_tool_credential_versions
  (id,namespace_id,credential_id,secret_ciphertext,ciphertext_nonce,kek_version,status,not_before)
VALUES ($1,$2,$3,$4,$5,$6,'active',clock_timestamp())`, newVersionID, namespaceID, id,
			secret.Ciphertext, secret.Nonce, secret.KEKVersion); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE agent_tool_credential_versions
SET status='retiring',expires_at=$4
WHERE namespace_id=$1 AND credential_id=$2 AND id=$3 AND status='active'`,
			namespaceID, id, current.ActiveVersionID, retireAt.UTC()); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		result, rotateToolCredentialErr := tx.ExecContext(ctx, `UPDATE agent_tool_credentials
SET active_version_id=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='active'`, namespaceID, id, expected, newVersionID)
		if rotateToolCredentialErr != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(rotateToolCredentialErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		return completeResourceCommand(
			ctx, tx, mutation, agentToolCredentialResourceType, id, expected+1, 200,
		)
	})
}

func (store *Store) DeleteToolCredential(
	ctx context.Context, namespaceID, id string, expected int64, _ agentmanagement.MutationContext,
) (int64, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (int64, error) {
		if _, err := lockToolCredential(ctx, tx, namespaceID, id, expected); err != nil {
			return 0, err
		}
		var references int
		if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM agent_tool_sources source
JOIN agent_tool_source_revisions revision
  ON revision.source_id=source.id AND revision.revision=source.current_revision
WHERE source.namespace_id=$1 AND source.status<>'deleted' AND revision.credential_id=$2`, namespaceID, id).Scan(&references); err != nil {
			return 0, fmt.Errorf("check Agent Tool credential references: %w", err)
		}
		if references != 0 {
			return 0, agentmanagement.ErrConflict
		}
		if _, err := tx.ExecContext(ctx, `UPDATE agent_tool_credential_versions
SET status='revoked',secret_ciphertext=NULL,ciphertext_nonce=NULL,kek_version=NULL,
    expires_at=NULL,revoked_at=clock_timestamp()
WHERE namespace_id=$1 AND credential_id=$2 AND status<>'revoked'`, namespaceID, id); err != nil {
			return 0, classifyWriteError(err)
		}
		result, err := tx.ExecContext(ctx, `UPDATE agent_tool_credentials
SET status='deleted',active_version_id=NULL,revision=revision+1,
    updated_at=clock_timestamp(),deleted_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3`, namespaceID, id, expected)
		if err != nil {
			return 0, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return 0, err
		}
		return expected + 1, nil
	})
}

func lockToolCredential(
	ctx context.Context, tx *sql.Tx, namespaceID, id string, expected int64,
) (agentmanagement.ToolCredential, error) {
	value, err := scanToolCredential(tx.QueryRowContext(ctx, credentialSelect+`
 WHERE namespace_id=$1 AND id=$2 AND status<>'deleted' FOR UPDATE`, namespaceID, id))
	if err != nil {
		return agentmanagement.ToolCredential{}, err
	}
	if value.Revision != expected {
		return agentmanagement.ToolCredential{}, agentmanagement.ErrConflict
	}
	return value, nil
}

func validateEncryptedSecret(value agentmanagement.EncryptedSecret) error {
	if len(value.Ciphertext) == 0 || len(value.Nonce) == 0 || value.KEKVersion == "" || len(value.KEKVersion) > 64 {
		return agentmanagement.ErrInvalid
	}
	return nil
}

func newResourceID() string { return uuid.NewString() }
