package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const serviceAccountColumns = `account.id::text,account.principal_id::text,principal.display_name,
       account.owner_scope,account.namespace_id::text,account.status,account.revision,
       account.created_at,account.updated_at`

// #nosec G101 -- this is a metadata column list and contains no service credential value.
const serviceCredentialColumns = `credential.id::text,credential.service_account_id::text,
       credential.public_id,credential.workload_class,credential.source_assured_at,
       credential.status,credential.not_before,credential.expires_at,
       credential.revoked_at,credential.created_at`

func (store *Store) ReadyWorkloadIdentity(
	ctx context.Context,
	commands *managementcommand.Codec,
	mtlsListenerEnabled bool,
) error {
	if store == nil || store.database == nil || commands == nil {
		return managementidentity.ErrWorkloadUnavailable
	}
	rows, err := store.database.QueryContext(ctx, `SELECT account.id,credential.id,mapping.id
FROM management_service_accounts account
LEFT JOIN management_service_account_credentials credential ON FALSE
LEFT JOIN management_mtls_mappings mapping ON FALSE LIMIT 0`)
	if err != nil {
		return fmt.Errorf("validate Management workload identity schema: %w", err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close Management workload identity schema validation: %w", err)
	}
	if err := commandpostgres.ValidateReferencedHMACVersions(ctx, store.database, commands); err != nil {
		return err
	}
	if !mtlsListenerEnabled {
		var active int64
		if err := store.database.QueryRowContext(ctx,
			`SELECT count(*) FROM management_mtls_mappings WHERE status='active'`,
		).Scan(&active); err != nil {
			return fmt.Errorf("validate active Management mTLS mappings: %w", err)
		}
		if active != 0 {
			return managementidentity.ErrMTLSListenerUnavailable
		}
	}
	return nil
}

func (store *Store) ReplaySecret(
	ctx context.Context,
	command managementcommand.Command,
) (managementidentity.StoredWorkloadSecret, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, store.database, command)
	if err != nil || !found {
		return managementidentity.StoredWorkloadSecret{}, found, err
	}
	return storedWorkloadSecret(stored)
}

func (store *Store) GetServiceAccount(ctx context.Context, id string) (managementidentity.ServiceAccount, error) {
	if !canonicalUUID(id) {
		return managementidentity.ServiceAccount{}, managementidentity.ErrNotFound
	}
	account, err := scanServiceAccount(store.database.QueryRowContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE account.id=$1`, id))
	return serviceAccountResult(account, err)
}

func (store *Store) ListServiceAccounts(
	ctx context.Context,
	query managementidentity.ServiceAccountQuery,
) (managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount], error) {
	if query.Limit < 1 || query.Limit > maximumPageSize ||
		(!query.Scope.Cluster && !canonicalUUID(query.Scope.NamespaceID)) {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{}, managementidentity.ErrInvalidWorkloadRequest
	}
	var afterTime, afterID any
	if query.After != nil {
		if query.After.CreatedAt.IsZero() || !canonicalUUID(query.After.ID) {
			return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{}, managementidentity.ErrInvalidWorkloadRequest
		}
		afterTime, afterID = query.After.CreatedAt.UTC(), query.After.ID
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE ($1 OR (account.owner_scope='namespace' AND account.namespace_id=$2::uuid
       AND ($3 OR account.id=ANY($4::uuid[]))))
  AND ($5='' OR account.status=$5)
  AND ($6::timestamptz IS NULL OR (account.created_at,account.id)>($6,$7::uuid))
ORDER BY account.created_at,account.id LIMIT $8`, query.Scope.Cluster, nullableWorkloadNamespace(query.Scope.NamespaceID),
		query.Scope.All, pq.Array(query.Scope.IDs), query.Status, afterTime, afterID, query.Limit+1)
	if err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{}, fmt.Errorf("list Management service accounts: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.ServiceAccount, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanServiceAccount(rows)
		if err != nil {
			return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{}, fmt.Errorf("scan Management service-account page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{}, fmt.Errorf("iterate Management service-account page: %w", err)
	}
	page := managementidentity.WorkloadRepositoryPage[managementidentity.ServiceAccount]{Items: items}
	if len(items) > query.Limit {
		page.Items, page.HasMore = items[:query.Limit], true
	}
	return page, nil
}

func (store *Store) ListServiceCredentials(
	ctx context.Context,
	query managementidentity.ServiceCredentialQuery,
) (managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential], error) {
	if !canonicalUUID(query.ServiceAccountID) || query.Limit < 1 || query.Limit > maximumPageSize {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, managementidentity.ErrInvalidWorkloadRequest
	}
	var afterTime, afterID any
	if query.After != nil {
		if query.After.CreatedAt.IsZero() || !canonicalUUID(query.After.ID) {
			return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, managementidentity.ErrInvalidWorkloadRequest
		}
		afterTime, afterID = query.After.CreatedAt.UTC(), query.After.ID
	}
	var exists bool
	if err := store.database.QueryRowContext(ctx,
		`SELECT EXISTS(SELECT 1 FROM management_service_accounts WHERE id=$1)`, query.ServiceAccountID,
	).Scan(&exists); err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, err
	}
	if !exists {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, managementidentity.ErrNotFound
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+serviceCredentialColumns+`
FROM management_service_account_credentials credential
WHERE credential.service_account_id=$1
  AND ($2::timestamptz IS NULL OR (credential.created_at,credential.id)>($2,$3::uuid))
ORDER BY credential.created_at,credential.id LIMIT $4`, query.ServiceAccountID, afterTime, afterID, query.Limit+1)
	if err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, fmt.Errorf("list Management service credentials: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.ServiceCredential, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanServiceCredential(rows)
		if err != nil {
			return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, fmt.Errorf("scan Management service-credential page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{}, fmt.Errorf("iterate Management service-credential page: %w", err)
	}
	page := managementidentity.WorkloadRepositoryPage[managementidentity.ServiceCredential]{Items: items}
	if len(items) > query.Limit {
		page.Items, page.HasMore = items[:query.Limit], true
	}
	return page, nil
}

func (store *Store) GetServiceCredential(
	ctx context.Context,
	accountID string,
	credentialID string,
) (managementidentity.ServiceCredential, error) {
	if !canonicalUUID(accountID) || !canonicalUUID(credentialID) {
		return managementidentity.ServiceCredential{}, managementidentity.ErrNotFound
	}
	credential, err := scanServiceCredential(store.database.QueryRowContext(ctx, `SELECT `+serviceCredentialColumns+`
FROM management_service_account_credentials credential
WHERE credential.service_account_id=$1 AND credential.id=$2`, accountID, credentialID))
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.ServiceCredential{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.ServiceCredential{}, fmt.Errorf("load Management service credential: %w", err)
	}
	return credential, nil
}

func (store *Store) CreateServiceAccount(
	ctx context.Context,
	mutation managementidentity.ServiceAccountCreateMutation,
) (managementidentity.WorkloadMutationResult, error) {
	if err := validateServiceAccountCreate(mutation); err != nil {
		return managementidentity.WorkloadMutationResult{}, err
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		if stored, found, err := commandpostgres.Lock(ctx, tx, mutation.Command); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapCommandError(err)
		} else if found {
			secret, ok, err := storedWorkloadSecret(stored)
			if err != nil || !ok {
				return managementidentity.WorkloadMutationResult{}, managementcommand.ErrConflict
			}
			return managementidentity.WorkloadMutationResult{
				Kind: secret.Result.ResourceType, ID: secret.Result.ResourceID, Revision: secret.Result.ResourceRevision,
				HTTPStatus: secret.Result.ResponseStatus, Replayed: true, Stored: &secret,
			}, nil
		}
		account := mutation.Account
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,attributes,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,'{}'::jsonb,'active',1,$5,$5)`, account.PrincipalID,
			managementidentity.ServiceAccountIssuer, account.ID, account.DisplayName, account.CreatedAt); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWriteError("create Management service-account principal", err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_service_accounts
  (id,principal_id,owner_scope,namespace_id,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,1,$6,$6)`, account.ID, account.PrincipalID, account.OwnerScope,
			nullableWorkloadNamespace(account.NamespaceID), account.Status, account.CreatedAt); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWriteError("create Management service account", err)
		}
		if err := insertServiceCredential(ctx, tx, mutation.Credential, mutation.SecretHMAC, mutation.PepperVersion); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: account.NamespaceID, Action: "service_account.created", ResourceType: "service_account",
			ResourceID: account.ID, AfterRevision: 1, Actor: mutation.Actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		result := managementcommand.ResourceResult{
			ResourceType: "service_account", ResourceID: account.ID, ResourceRevision: 1, ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command, result, managementcommand.SecretResponse{
			Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
			KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: result.ResourceType, ID: result.ResourceID, Revision: 1, HTTPStatus: 201,
		}, nil
	})
}

func (store *Store) PatchServiceAccount(
	ctx context.Context,
	updated managementidentity.ServiceAccount,
	expected uint64,
	actor managementidentity.MutationActor,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(updated.ID) || !canonicalUUID(updated.PrincipalID) || expected == 0 {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		current, patchServiceAccountErr := scanServiceAccount(tx.QueryRowContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE account.id=$1 FOR UPDATE OF account,principal`, updated.ID))
		if errors.Is(patchServiceAccountErr, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if patchServiceAccountErr != nil {
			return managementidentity.WorkloadMutationResult{}, patchServiceAccountErr
		}
		if current.Revision != expected {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		if current.PrincipalID != updated.PrincipalID || current.OwnerScope != updated.OwnerScope ||
			current.NamespaceID != updated.NamespaceID {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
		}
		principalStatus := "active"
		if updated.Status == managementidentity.ServiceAccountDisabled {
			principalStatus = "disabled"
		}
		if _, err := tx.ExecContext(ctx, `UPDATE management_principals
SET display_name=$2,status=$3,revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1`, updated.PrincipalID, updated.DisplayName, principalStatus); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWriteError("update Management service-account principal", err)
		}
		var revision uint64
		if err := tx.QueryRowContext(ctx, `UPDATE management_service_accounts
SET status=$3,revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2 RETURNING revision`, updated.ID, expected, updated.Status).Scan(&revision); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
			}
			return managementidentity.WorkloadMutationResult{}, err
		}
		sessions := []string(nil)
		if updated.Status == managementidentity.ServiceAccountDisabled {
			sessions, patchServiceAccountErr = revokePrincipalSessions(ctx, tx, updated.PrincipalID)
			if patchServiceAccountErr != nil {
				return managementidentity.WorkloadMutationResult{}, patchServiceAccountErr
			}
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: current.NamespaceID, Action: "service_account.updated", ResourceType: "service_account",
			ResourceID: current.ID, BeforeRevision: &expected, AfterRevision: revision, Actor: actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: "service_account", ID: current.ID, Revision: revision, HTTPStatus: 200, SessionIDs: sessions,
		}, nil
	})
}

func (store *Store) DeleteServiceAccount(
	ctx context.Context,
	id string,
	expected uint64,
	actor managementidentity.MutationActor,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(id) || expected == 0 {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		account, err := scanServiceAccount(tx.QueryRowContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE account.id=$1 FOR UPDATE OF account,principal`, id))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if account.Revision != expected {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		if account.Status != managementidentity.ServiceAccountDisabled {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrWorkloadDependency
		}
		var dependencies bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(
  SELECT 1 FROM management_role_bindings WHERE principal_id=$1
  UNION ALL SELECT 1 FROM management_principal_user_links WHERE principal_id=$1
  UNION ALL SELECT 1 FROM management_mtls_mappings WHERE principal_id=$1
  UNION ALL SELECT 1 FROM management_sessions WHERE principal_id=$1 AND status='active' AND expires_at>clock_timestamp()
)`, account.PrincipalID).Scan(&dependencies); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if dependencies {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrWorkloadDependency
		}
		if _, err := tx.ExecContext(ctx,
			`DELETE FROM management_service_account_credentials WHERE service_account_id=$1`, id,
		); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWorkloadDependency("delete Management service credentials", err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM management_service_accounts WHERE id=$1`, id); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWorkloadDependency("delete Management service account", err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE management_principals
SET status='disabled',revision=revision+1,updated_at=clock_timestamp() WHERE id=$1`, account.PrincipalID); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		after := expected + 1
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: account.NamespaceID, Action: "service_account.deleted", ResourceType: "service_account",
			ResourceID: account.ID, BeforeRevision: &expected, AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: "service_account", ID: id, Revision: after, HTTPStatus: 204,
		}, nil
	})
}

func (store *Store) RotateServiceCredential(
	ctx context.Context,
	mutation managementidentity.ServiceCredentialRotateMutation,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(mutation.AccountID) || mutation.ExpectedRevision == 0 ||
		mutation.Credential.ServiceAccountID != mutation.AccountID ||
		mutation.Command.PrincipalID != mutation.Actor.PrincipalID {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		if stored, found, err := commandpostgres.Lock(ctx, tx, mutation.Command); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapCommandError(err)
		} else if found {
			secret, ok, err := storedWorkloadSecret(stored)
			if err != nil || !ok || secret.Result.ResourceID != mutation.AccountID {
				return managementidentity.WorkloadMutationResult{}, managementcommand.ErrConflict
			}
			return managementidentity.WorkloadMutationResult{
				Kind: secret.Result.ResourceType, ID: secret.Result.ResourceID, Revision: secret.Result.ResourceRevision,
				HTTPStatus: secret.Result.ResponseStatus, Replayed: true, Stored: &secret,
			}, nil
		}
		account, rotateServiceCredentialErr := scanServiceAccount(tx.QueryRowContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE account.id=$1 FOR UPDATE OF account`, mutation.AccountID))
		if errors.Is(rotateServiceCredentialErr, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if rotateServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, rotateServiceCredentialErr
		}
		if account.Revision != mutation.ExpectedRevision || account.Status != managementidentity.ServiceAccountActive {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		rows, rotateServiceCredentialErr := tx.QueryContext(ctx, `UPDATE management_service_account_credentials
SET status=CASE WHEN $2<=clock_timestamp() THEN 'revoked' ELSE 'retiring' END,
    expires_at=CASE WHEN $2<=clock_timestamp() THEN expires_at ELSE LEAST(expires_at,$2) END,
    revoked_at=CASE WHEN $2<=clock_timestamp() THEN clock_timestamp() ELSE revoked_at END
WHERE service_account_id=$1 AND status='active'
RETURNING id::text,status`, mutation.AccountID, mutation.RetireAt)
		if rotateServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, rotateServiceCredentialErr
		}
		revoked := make([]string, 0)
		for rows.Next() {
			var credentialID, status string
			if err := rows.Scan(&credentialID, &status); err != nil {
				rows.Close()
				return managementidentity.WorkloadMutationResult{}, err
			}
			if status == string(managementidentity.ServiceCredentialRevoked) {
				revoked = append(revoked, credentialID)
			}
		}
		if err := rows.Close(); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if err := insertServiceCredential(ctx, tx, mutation.Credential, mutation.SecretHMAC, mutation.PepperVersion); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		var revision uint64
		if err := tx.QueryRowContext(ctx, `UPDATE management_service_accounts
SET revision=revision+1,updated_at=clock_timestamp() WHERE id=$1 AND revision=$2 RETURNING revision`,
			mutation.AccountID, mutation.ExpectedRevision).Scan(&revision); err != nil {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		sessions, rotateServiceCredentialErr := revokeSourceSessions(ctx, tx, managementauth.AuthSourceServiceCredential, revoked)
		if rotateServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, rotateServiceCredentialErr
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: account.NamespaceID, Action: "service_account.credential_rotated", ResourceType: "service_account",
			ResourceID: account.ID, BeforeRevision: &mutation.ExpectedRevision, AfterRevision: revision, Actor: mutation.Actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		result := managementcommand.ResourceResult{
			ResourceType: "service_account", ResourceID: account.ID, ResourceRevision: revision, ResponseStatus: 200,
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command, result, managementcommand.SecretResponse{
			Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
			KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: result.ResourceType, ID: result.ResourceID, Revision: revision, HTTPStatus: 200,
			SessionIDs: sessions, RevokedCredentialIDs: revoked,
		}, nil
	})
}

func (store *Store) RevokeServiceCredential(
	ctx context.Context,
	accountID string,
	credentialID string,
	expected uint64,
	actor managementidentity.MutationActor,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(accountID) || !canonicalUUID(credentialID) || expected == 0 {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		account, revokeServiceCredentialErr := scanServiceAccount(tx.QueryRowContext(ctx, `SELECT `+serviceAccountColumns+`
FROM management_service_accounts account
JOIN management_principals principal ON principal.id=account.principal_id
WHERE account.id=$1 FOR UPDATE OF account`, accountID))
		if errors.Is(revokeServiceCredentialErr, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if revokeServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, revokeServiceCredentialErr
		}
		if account.Revision != expected {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		result, revokeServiceCredentialErr := tx.ExecContext(ctx, `UPDATE management_service_account_credentials
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE id=$1 AND service_account_id=$2 AND status IN ('active','retiring')`, credentialID, accountID)
		if revokeServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, revokeServiceCredentialErr
		}
		changed, revokeServiceCredentialErr := result.RowsAffected()
		if revokeServiceCredentialErr != nil || changed != 1 {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		var revision uint64
		if err := tx.QueryRowContext(ctx, `UPDATE management_service_accounts
SET revision=revision+1,updated_at=clock_timestamp() WHERE id=$1 AND revision=$2 RETURNING revision`,
			accountID, expected).Scan(&revision); err != nil {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		sessions, revokeServiceCredentialErr := revokeSourceSessions(ctx, tx, managementauth.AuthSourceServiceCredential, []string{credentialID})
		if revokeServiceCredentialErr != nil {
			return managementidentity.WorkloadMutationResult{}, revokeServiceCredentialErr
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: account.NamespaceID, Action: "service_account.credential_revoked", ResourceType: "service_account",
			ResourceID: account.ID, BeforeRevision: &expected, AfterRevision: revision, Actor: actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: "service_account", ID: accountID, Revision: revision, HTTPStatus: 204,
			SessionIDs: sessions, RevokedCredentialIDs: []string{credentialID},
		}, nil
	})
}

func validateServiceAccountCreate(mutation managementidentity.ServiceAccountCreateMutation) error {
	account, credential := mutation.Account, mutation.Credential
	expectedScope := managementcommand.ScopeCluster
	if account.OwnerScope == managementidentity.ServiceAccountOwnerNamespace {
		expectedScope = managementcommand.ScopeNamespace
	}
	if !canonicalUUID(account.ID) || !canonicalUUID(account.PrincipalID) || !canonicalUUID(credential.ID) ||
		credential.ServiceAccountID != account.ID || account.Revision != 1 || credential.PublicID != credential.ID ||
		mutation.Command.Scope.Kind != expectedScope || mutation.Command.PrincipalID != mutation.Actor.PrincipalID ||
		(expectedScope == managementcommand.ScopeNamespace && mutation.Command.Scope.NamespaceID != account.NamespaceID) ||
		len(mutation.SecretHMAC) != 32 || mutation.PepperVersion == "" || mutation.ResponseExpiresAt.IsZero() {
		return managementidentity.ErrInvalidWorkloadRequest
	}
	return nil
}

func insertServiceCredential(
	ctx context.Context,
	tx *sql.Tx,
	credential managementidentity.ServiceCredential,
	digest []byte,
	pepperVersion string,
) error {
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_service_account_credentials
  (id,service_account_id,public_id,secret_hmac,pepper_version,workload_class,
   source_assured_at,status,not_before,expires_at,revoked_at,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`, credential.ID, credential.ServiceAccountID,
		credential.PublicID, digest, pepperVersion, credential.WorkloadClass, credential.SourceAssuredAt,
		credential.Status, credential.NotBefore, credential.ExpiresAt, credential.RevokedAt, credential.CreatedAt); err != nil {
		return mapWriteError("create Management service credential", err)
	}
	return nil
}

func revokePrincipalSessions(ctx context.Context, tx *sql.Tx, principalID string) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE principal_id=$1 AND status='active' AND expires_at>clock_timestamp()
RETURNING id::text`, principalID)
	return scanRevokedSessionIDs(rows, err)
}

func revokeSourceSessions(
	ctx context.Context,
	tx *sql.Tx,
	kind managementauth.AuthSourceKind,
	sourceIDs []string,
) ([]string, error) {
	if len(sourceIDs) == 0 {
		return nil, nil
	}
	rows, err := tx.QueryContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE auth_source_kind=$1 AND auth_source_id=ANY($2::uuid[])
  AND status='active' AND expires_at>clock_timestamp()
RETURNING id::text`, kind, pq.Array(sourceIDs))
	return scanRevokedSessionIDs(rows, err)
}

func scanRevokedSessionIDs(rows *sql.Rows, err error) ([]string, error) {
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil || !canonicalUUID(id) {
			return nil, managementidentity.ErrWorkloadUnavailable
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func scanServiceAccount(scanner scanner) (managementidentity.ServiceAccount, error) {
	var account managementidentity.ServiceAccount
	var namespace sql.NullString
	if err := scanner.Scan(&account.ID, &account.PrincipalID, &account.DisplayName,
		&account.OwnerScope, &namespace, &account.Status, &account.Revision,
		&account.CreatedAt, &account.UpdatedAt); err != nil {
		return managementidentity.ServiceAccount{}, err
	}
	if namespace.Valid {
		account.NamespaceID = namespace.String
	}
	if !canonicalUUID(account.ID) || !canonicalUUID(account.PrincipalID) || account.DisplayName == "" || account.Revision == 0 ||
		(account.OwnerScope == managementidentity.ServiceAccountOwnerCluster && namespace.Valid) ||
		(account.OwnerScope == managementidentity.ServiceAccountOwnerNamespace && (!namespace.Valid || !canonicalUUID(namespace.String))) ||
		(account.Status != managementidentity.ServiceAccountActive && account.Status != managementidentity.ServiceAccountDisabled) {
		return managementidentity.ServiceAccount{}, errors.New("stored Management service account is invalid")
	}
	account.CreatedAt, account.UpdatedAt = account.CreatedAt.UTC(), account.UpdatedAt.UTC()
	return account, nil
}

func scanServiceCredential(scanner scanner) (managementidentity.ServiceCredential, error) {
	var credential managementidentity.ServiceCredential
	var revoked sql.NullTime
	if err := scanner.Scan(&credential.ID, &credential.ServiceAccountID, &credential.PublicID,
		&credential.WorkloadClass, &credential.SourceAssuredAt, &credential.Status,
		&credential.NotBefore, &credential.ExpiresAt, &revoked, &credential.CreatedAt); err != nil {
		return managementidentity.ServiceCredential{}, err
	}
	if revoked.Valid {
		value := revoked.Time.UTC()
		credential.RevokedAt = &value
	}
	if !canonicalUUID(credential.ID) || !canonicalUUID(credential.ServiceAccountID) || credential.PublicID != credential.ID ||
		(credential.WorkloadClass != managementidentity.WorkloadStandard && credential.WorkloadClass != managementidentity.WorkloadStrong) ||
		(credential.Status != managementidentity.ServiceCredentialActive && credential.Status != managementidentity.ServiceCredentialRetiring && credential.Status != managementidentity.ServiceCredentialRevoked) ||
		credential.SourceAssuredAt.IsZero() || credential.NotBefore.IsZero() || !credential.ExpiresAt.After(credential.NotBefore) ||
		(credential.Status == managementidentity.ServiceCredentialRevoked) != revoked.Valid {
		return managementidentity.ServiceCredential{}, errors.New("stored Management service credential is invalid")
	}
	credential.SourceAssuredAt = credential.SourceAssuredAt.UTC()
	credential.NotBefore, credential.ExpiresAt, credential.CreatedAt = credential.NotBefore.UTC(), credential.ExpiresAt.UTC(), credential.CreatedAt.UTC()
	return credential, nil
}

func serviceAccountResult(account managementidentity.ServiceAccount, err error) (managementidentity.ServiceAccount, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.ServiceAccount{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.ServiceAccount{}, fmt.Errorf("load Management service account: %w", err)
	}
	return account, nil
}

func storedWorkloadSecret(stored managementcommand.StoredResult) (managementidentity.StoredWorkloadSecret, bool, error) {
	if stored.Resource == nil || stored.Operation != nil || stored.Secret == nil || stored.Resource.ResourceType != "service_account" {
		return managementidentity.StoredWorkloadSecret{}, false, managementcommand.ErrConflict
	}
	return managementidentity.StoredWorkloadSecret{Result: *stored.Resource, Secret: *stored.Secret}, true, nil
}

func nullableWorkloadNamespace(namespaceID string) any {
	if namespaceID == "" {
		return nil
	}
	return namespaceID
}

func mapWorkloadDependency(action string, err error) error {
	var pqError *pq.Error
	if errors.As(err, &pqError) && pqError.Code == "23503" {
		return managementidentity.ErrWorkloadDependency
	}
	return fmt.Errorf("%s: %w", action, err)
}

var _ managementidentity.WorkloadIdentityRepository = (*Store)(nil)
