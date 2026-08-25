package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const (
	// #nosec G101 -- this is a database column list and contains no provider secret.
	providerCredentialColumns = `id, namespace_id, name, provider_id, credential_mode, credential_adapter_id, provider_catalog_revision, normalized_origin,
	       status, active_version_id, revision, created_at, updated_at, deleted_at`
	// #nosec G101 -- this is a database column list and contains no provider secret.
	providerCredentialVersionColumns = `id, namespace_id, provider_credential_id,
       secret_ciphertext, ciphertext_nonce, kek_version, status, not_before,
       expires_at, revoked_at, created_at`
	getProviderCredentialQuery = `SELECT ` + providerCredentialColumns + `
FROM provider_credentials WHERE namespace_id = $1 AND id = $2`
	listProviderCredentialsQuery = `SELECT ` + providerCredentialColumns + `
FROM provider_credentials
WHERE namespace_id = $1
  AND ($2 = '' OR provider_id = $2)
  AND ($3 = '' OR status = $3)
  AND ($4 OR id = ANY($5::uuid[]))
  AND ($6 = '' OR (status, id) > ($6, $7::uuid))
ORDER BY status ASC, id ASC
LIMIT $8`
	getProviderCredentialByIDQuery = `SELECT ` + providerCredentialColumns + `
FROM provider_credentials WHERE id = $1`
	getProviderCredentialVersionQuery = `SELECT ` + providerCredentialVersionColumns + `
FROM provider_credential_versions
WHERE provider_credential_id = $1 AND id = $2`
	insertProviderCredentialQuery = `INSERT INTO provider_credentials
	  (id, namespace_id, name, provider_id, credential_mode, credential_adapter_id, provider_catalog_revision, normalized_origin, status,
	   active_version_id, revision, created_at, updated_at)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 1, $11, $12)
	RETURNING ` + providerCredentialColumns
	// #nosec G101 -- this is a parameterized insert statement and contains no provider secret.
	insertProviderCredentialVersionQuery = `INSERT INTO provider_credential_versions
  (id, namespace_id, provider_credential_id, secret_ciphertext,
   ciphertext_nonce, kek_version, status, not_before, expires_at,
   revoked_at, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`
	renameProviderCredentialQuery = `UPDATE provider_credentials
SET name = $4, revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
  AND status <> 'deleted'
RETURNING ` + providerCredentialColumns
	rotateProviderCredentialQuery = `UPDATE provider_credentials
SET active_version_id = $5, revision = revision + 1,
    updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
  AND status = 'active' AND active_version_id = $4
RETURNING ` + providerCredentialColumns
	retireProviderCredentialVersionQuery = `UPDATE provider_credential_versions
SET status = 'retiring', expires_at = $4
WHERE namespace_id = $1 AND provider_credential_id = $2 AND id = $3
  AND status = 'active'`
	reactivateProviderCredentialQuery = `UPDATE provider_credentials
SET status = 'active', active_version_id = $4, revision = revision + 1,
    updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
  AND status = 'disabled' AND active_version_id IS NULL
RETURNING ` + providerCredentialColumns
	disableProviderCredentialQuery = `UPDATE provider_credentials
SET status = 'disabled', active_version_id = NULL,
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
  AND status = 'active'
RETURNING ` + providerCredentialColumns
	deleteProviderCredentialQuery = `UPDATE provider_credentials
SET status = 'deleted', active_version_id = NULL, deleted_at = clock_timestamp(),
    revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
  AND status IN ('active', 'disabled')
RETURNING ` + providerCredentialColumns
	revokeProviderCredentialVersionsQuery = `UPDATE provider_credential_versions
SET status = 'revoked', revoked_at = COALESCE(revoked_at, clock_timestamp()),
    secret_ciphertext = NULL, ciphertext_nonce = NULL, kek_version = NULL
WHERE namespace_id = $1 AND provider_credential_id = $2
  AND status IN ('active', 'retiring')`
)

// ValidateManagementCommandHMACVersions fails readiness when a live durable
// command references a key version that this replica cannot validate.
func (s *Store) ValidateManagementCommandHMACVersions(
	ctx context.Context,
	codec *managementcommand.Codec,
) error {
	if s == nil || s.db == nil {
		return errors.New("provider credential repository is unavailable")
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, s.db, codec)
}

func (s *Store) GetProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
) (providercredential.Credential, error) {
	if err := validateIdentityIDs(namespaceID, id); err != nil {
		return providercredential.Credential{}, err
	}
	credential, err := scanProviderCredential(s.db.QueryRowContext(ctx, getProviderCredentialQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return providercredential.Credential{}, ErrNotFound
	}
	if err != nil {
		return providercredential.Credential{}, fmt.Errorf("get provider credential: %w", err)
	}
	return credential, nil
}

func (s *Store) ListProviderCredentials(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	request ProviderCredentialListRequest,
) (ProviderCredentialListResult, error) {
	if err := validateUUID("namespace id", string(namespaceID)); err != nil {
		return ProviderCredentialListResult{}, err
	}
	if request.PageSize < 1 || request.PageSize > 200 {
		return ProviderCredentialListResult{}, fmt.Errorf("provider credential page size must be between 1 and 200")
	}
	if _, err := request.Scope.Digest(); err != nil || request.Scope.NamespaceID != namespaceID {
		return ProviderCredentialListResult{}, fmt.Errorf("provider credential result scope is invalid")
	}
	if !request.Scope.All && len(request.Scope.IDs(accesscontrol.ScopeResourceProviderCredential)) == 0 {
		return ProviderCredentialListResult{Credentials: []providercredential.Credential{}}, nil
	}
	if request.ProviderID != "" && providercredential.ValidateProviderID(request.ProviderID) != nil {
		return ProviderCredentialListResult{}, fmt.Errorf("provider credential provider filter is invalid")
	}
	if request.Status != "" && !validProviderCredentialStatus(request.Status) {
		return ProviderCredentialListResult{}, fmt.Errorf("provider credential status filter is invalid")
	}
	if (request.AfterStatus == "") != (request.AfterID == "") {
		return ProviderCredentialListResult{}, fmt.Errorf("provider credential cursor is incomplete")
	}
	if request.AfterStatus != "" {
		if !validProviderCredentialStatus(request.AfterStatus) {
			return ProviderCredentialListResult{}, fmt.Errorf("provider credential cursor status is invalid")
		}
		if err := validateUUID("provider credential cursor id", request.AfterID); err != nil {
			return ProviderCredentialListResult{}, err
		}
	}
	rows, err := s.db.QueryContext(ctx, listProviderCredentialsQuery,
		namespaceID, request.ProviderID, request.Status, request.Scope.All,
		pq.Array(request.Scope.IDs(accesscontrol.ScopeResourceProviderCredential)), request.AfterStatus,
		nullProviderCredentialCursorID(request.AfterID), request.PageSize+1,
	)
	if err != nil {
		return ProviderCredentialListResult{}, fmt.Errorf("list provider credentials: %w", err)
	}
	defer rows.Close()
	credentials := make([]providercredential.Credential, 0, request.PageSize+1)
	for rows.Next() {
		credential, err := scanProviderCredential(rows)
		if err != nil {
			return ProviderCredentialListResult{}, fmt.Errorf("scan provider credential page: %w", err)
		}
		credentials = append(credentials, credential)
	}
	if err := rows.Err(); err != nil {
		return ProviderCredentialListResult{}, fmt.Errorf("read provider credential page: %w", err)
	}
	result := ProviderCredentialListResult{Credentials: credentials}
	if len(credentials) > request.PageSize {
		result.Credentials = credentials[:request.PageSize]
		result.HasMore = true
	}
	return result, nil
}

// ReplayProviderCredentialCommand performs a committed-result fast path before
// callers consult mutable catalog state. The mutation transaction still calls
// commandpostgres.Lock and remains authoritative for races.
func (s *Store) ReplayProviderCredentialCommand(
	ctx context.Context,
	command managementcommand.Command,
) (MutationResult[providercredential.Credential], bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, s.db, command)
	if err != nil || !found {
		return MutationResult[providercredential.Credential]{}, false, err
	}
	result, err := loadProviderCredentialCommandResult(stored)
	if err != nil {
		return MutationResult[providercredential.Credential]{}, false, err
	}
	return result, true, nil
}

// LoadActiveProviderCredential is the backend resolver's O(1) active-pointer
// lookup. The UUID is globally unique; provider and origin bindings are checked
// by providercredential.Codec before any secret is returned.
func (s *Store) LoadActiveProviderCredential(
	ctx context.Context,
	id string,
) (providercredential.Credential, providercredential.Version, error) {
	if err := validateUUID("provider credential id", id); err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	pair, err := inReadTransaction(ctx, s, func(tx *sql.Tx) (providerCredentialPair, error) {
		credential, err := scanProviderCredential(tx.QueryRowContext(ctx, getProviderCredentialByIDQuery, id))
		if errors.Is(err, sql.ErrNoRows) {
			return providerCredentialPair{}, ErrNotFound
		}
		if err != nil {
			return providerCredentialPair{}, fmt.Errorf("load active provider credential: %w", err)
		}
		if credential.Status != providercredential.StatusActive || credential.ActiveVersionID == nil {
			return providerCredentialPair{}, providercredential.ErrUnavailable
		}
		version, err := loadProviderCredentialVersion(ctx, tx, credential.ID, *credential.ActiveVersionID)
		return providerCredentialPair{Credential: credential, Version: version}, err
	})
	return pair.Credential, pair.Version, err
}

// LoadPinnedProviderCredential retrieves the exact version recorded in a
// dispatch journal. It never falls back to the current active pointer.
func (s *Store) LoadPinnedProviderCredential(
	ctx context.Context,
	id string,
	versionID string,
) (providercredential.Credential, providercredential.Version, error) {
	if err := validateUUID("provider credential id", id); err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	if err := validateUUID("provider credential version id", versionID); err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	pair, err := inReadTransaction(ctx, s, func(tx *sql.Tx) (providerCredentialPair, error) {
		credential, err := scanProviderCredential(tx.QueryRowContext(ctx, getProviderCredentialByIDQuery, id))
		if errors.Is(err, sql.ErrNoRows) {
			return providerCredentialPair{}, ErrNotFound
		}
		if err != nil {
			return providerCredentialPair{}, fmt.Errorf("load pinned provider credential: %w", err)
		}
		version, err := loadProviderCredentialVersion(ctx, tx, credential.ID, versionID)
		return providerCredentialPair{Credential: credential, Version: version}, err
	})
	return pair.Credential, pair.Version, err
}

type providerCredentialPair struct {
	Credential providercredential.Credential
	Version    providercredential.Version
}

func (s *Store) CreateProviderCredential(
	ctx context.Context,
	credential providercredential.Credential,
	version providercredential.Version,
	command managementcommand.Command,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	if err := validateNewProviderCredential(credential, version); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	if err := validateProviderCredentialCommand(command, credential.NamespaceID, meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[providercredential.Credential], error) {
		stored, replayed, createProviderCredentialErr := commandpostgres.Lock(ctx, tx, command)
		if createProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, createProviderCredentialErr
		}
		if replayed {
			return loadProviderCredentialCommandResult(stored)
		}
		created, createProviderCredentialErr := scanProviderCredential(tx.QueryRowContext(ctx, insertProviderCredentialQuery,
			credential.ID, credential.NamespaceID, credential.Name, credential.ProviderID,
			credential.CredentialMode, credential.CredentialAdapterID, credential.CatalogRevision, credential.NormalizedOrigin,
			credential.Status, *credential.ActiveVersionID,
			credential.CreatedAt, credential.UpdatedAt))
		if createProviderCredentialErr != nil {
			if providerCredentialUniqueViolation(createProviderCredentialErr) {
				return MutationResult[providercredential.Credential]{}, ErrAlreadyExists
			}
			return MutationResult[providercredential.Credential]{}, fmt.Errorf("insert provider credential: %w", createProviderCredentialErr)
		}
		if err := insertProviderCredentialVersion(ctx, tx, version); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		receipt, createProviderCredentialErr := appendProviderCredentialMutation(
			ctx, tx, created, outboxCreated, meta, map[string]string{"versionId": version.ID},
		)
		if createProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, createProviderCredentialErr
		}
		if err := commandpostgres.CompleteResource(ctx, tx, command, managementcommand.ResourceResult{
			ResourceType: "provider_credential", ResourceID: created.ID,
			ResourceRevision: created.Revision,
			ResponseStatus:   201,
		}); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		return MutationResult[providercredential.Credential]{
			Value: created, Receipt: receipt, ResourceID: created.ID,
			ResourceRevision: accesscontrol.Revision(created.Revision), ResponseStatus: 201,
		}, nil
	})
}

func (s *Store) RenameProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	name string,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	if err := validateProviderCredentialMutation(namespaceID, id, expected, meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	if err := providercredential.ValidateName(name); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	expectedValue, _ := revisionAsInt64(expected)
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[providercredential.Credential], error) {
		updated, renameProviderCredentialErr := scanProviderCredential(tx.QueryRowContext(ctx, renameProviderCredentialQuery,
			namespaceID, id, expectedValue, name))
		if err := mapProviderCredentialMutationError(renameProviderCredentialErr, "rename provider credential"); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		receipt, renameProviderCredentialErr := appendProviderCredentialMutation(ctx, tx, updated, outboxUpdated, meta, nil)
		if renameProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, renameProviderCredentialErr
		}
		return MutationResult[providercredential.Credential]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) RotateProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	rotation ProviderCredentialRotation,
	command managementcommand.Command,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	if err := validateProviderCredentialRotation(namespaceID, id, expected, rotation, meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	if err := validateProviderCredentialCommand(command, string(namespaceID), meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	expectedValue, _ := revisionAsInt64(expected)
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[providercredential.Credential], error) {
		stored, replayed, rotateProviderCredentialErr := commandpostgres.Lock(ctx, tx, command)
		if rotateProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, rotateProviderCredentialErr
		}
		if replayed {
			return loadProviderCredentialCommandResult(stored)
		}
		if err := insertProviderCredentialVersion(ctx, tx, rotation.Version); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		result, rotateProviderCredentialErr := tx.ExecContext(ctx, retireProviderCredentialVersionQuery,
			namespaceID, id, rotation.PreviousVersionID, rotation.RetireAt)
		if rotateProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, fmt.Errorf("retire provider credential version: %w", rotateProviderCredentialErr)
		}
		if err := requireOneRow(result, ErrRevisionConflict); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		updated, rotateProviderCredentialErr := scanProviderCredential(tx.QueryRowContext(ctx, rotateProviderCredentialQuery,
			namespaceID, id, expectedValue, rotation.PreviousVersionID, rotation.Version.ID))
		if err := mapProviderCredentialMutationError(rotateProviderCredentialErr, "rotate provider credential"); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		receipt, rotateProviderCredentialErr := appendProviderCredentialMutation(ctx, tx, updated, outboxCredentialRotated, meta,
			map[string]string{"versionId": rotation.Version.ID, "retiredVersionId": rotation.PreviousVersionID})
		if rotateProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, rotateProviderCredentialErr
		}
		if err := commandpostgres.CompleteResource(ctx, tx, command, managementcommand.ResourceResult{
			ResourceType: "provider_credential", ResourceID: updated.ID,
			ResourceRevision: updated.Revision,
			ResponseStatus:   200,
		}); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		return MutationResult[providercredential.Credential]{
			Value: updated, Receipt: receipt, ResourceID: updated.ID,
			ResourceRevision: accesscontrol.Revision(updated.Revision), ResponseStatus: 200,
		}, nil
	})
}

func (s *Store) ReactivateProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	version providercredential.Version,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	if err := validateProviderCredentialMutation(namespaceID, id, expected, meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	if err := validateNewProviderCredentialVersion(namespaceID, id, version); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	expectedValue, _ := revisionAsInt64(expected)
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[providercredential.Credential], error) {
		if err := insertProviderCredentialVersion(ctx, tx, version); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		updated, reactivateProviderCredentialErr := scanProviderCredential(tx.QueryRowContext(ctx, reactivateProviderCredentialQuery,
			namespaceID, id, expectedValue, version.ID))
		if err := mapProviderCredentialMutationError(reactivateProviderCredentialErr, "reactivate provider credential"); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		receipt, reactivateProviderCredentialErr := appendProviderCredentialMutation(ctx, tx, updated, outboxCredentialRotated, meta,
			map[string]string{"versionId": version.ID})
		if reactivateProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, reactivateProviderCredentialErr
		}
		return MutationResult[providercredential.Credential]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) DisableProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	return s.terminateProviderCredential(ctx, namespaceID, id, expected, meta, false)
}

func (s *Store) DeleteProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[providercredential.Credential], error) {
	return s.terminateProviderCredential(ctx, namespaceID, id, expected, meta, true)
}

func (s *Store) terminateProviderCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	meta MutationMeta,
	deleted bool,
) (MutationResult[providercredential.Credential], error) {
	if err := validateProviderCredentialMutation(namespaceID, id, expected, meta); err != nil {
		return MutationResult[providercredential.Credential]{}, err
	}
	expectedValue, _ := revisionAsInt64(expected)
	query := disableProviderCredentialQuery
	operation := outboxUpdated
	verb := "disable"
	if deleted {
		query = deleteProviderCredentialQuery
		operation = outboxDeleted
		verb = "delete"
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[providercredential.Credential], error) {
		updated, terminateProviderCredentialErr := scanProviderCredential(tx.QueryRowContext(ctx, query, namespaceID, id, expectedValue))
		if err := mapProviderCredentialMutationError(terminateProviderCredentialErr, verb+" provider credential"); err != nil {
			return MutationResult[providercredential.Credential]{}, err
		}
		if _, err := tx.ExecContext(ctx, revokeProviderCredentialVersionsQuery, namespaceID, id); err != nil {
			return MutationResult[providercredential.Credential]{}, fmt.Errorf("erase provider credential versions: %w", err)
		}
		receipt, terminateProviderCredentialErr := appendProviderCredentialMutation(ctx, tx, updated, operation, meta, nil)
		if terminateProviderCredentialErr != nil {
			return MutationResult[providercredential.Credential]{}, terminateProviderCredentialErr
		}
		return MutationResult[providercredential.Credential]{Value: updated, Receipt: receipt}, nil
	})
}

func insertProviderCredentialVersion(ctx context.Context, tx *sql.Tx, version providercredential.Version) error {
	if err := version.Validate(); err != nil {
		return err
	}
	if version.Status != providercredential.VersionActive {
		return fmt.Errorf("new provider credential version must be active")
	}
	if _, err := tx.ExecContext(ctx, insertProviderCredentialVersionQuery,
		version.ID, version.NamespaceID, version.CredentialID,
		version.Envelope.Ciphertext, version.Envelope.Nonce, version.Envelope.KeyVersion,
		version.Status, version.NotBefore, version.ExpiresAt, version.RevokedAt, version.CreatedAt); err != nil {
		return fmt.Errorf("insert provider credential version: %w", err)
	}
	return nil
}

func loadProviderCredentialVersion(
	ctx context.Context,
	tx *sql.Tx,
	credentialID string,
	versionID string,
) (providercredential.Version, error) {
	version, err := scanProviderCredentialVersion(tx.QueryRowContext(ctx,
		getProviderCredentialVersionQuery, credentialID, versionID))
	if errors.Is(err, sql.ErrNoRows) {
		return providercredential.Version{}, ErrNotFound
	}
	if err != nil {
		return providercredential.Version{}, fmt.Errorf("load provider credential version: %w", err)
	}
	return version, nil
}

func appendProviderCredentialMutation(
	ctx context.Context,
	tx *sql.Tx,
	credential providercredential.Credential,
	operation outboxOperation,
	meta MutationMeta,
	references map[string]string,
) (MutationReceipt, error) {
	return appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(credential.NamespaceID), outboxMutation{
		AggregateType: "provider_credential", AggregateID: credential.ID,
		AggregateRevision: accesscontrol.Revision(credential.Revision), Operation: operation,
		References: references,
	}, meta)
}

func validateNewProviderCredential(
	credential providercredential.Credential,
	version providercredential.Version,
) error {
	if err := credential.Validate(); err != nil {
		return err
	}
	if credential.Revision != 1 || credential.Status != providercredential.StatusActive ||
		credential.ActiveVersionID == nil || *credential.ActiveVersionID != version.ID {
		return fmt.Errorf("new provider credential must be active at revision one with its initial version")
	}
	return validateNewProviderCredentialVersion(
		accesscontrol.NamespaceID(credential.NamespaceID), credential.ID, version,
	)
}

func validateNewProviderCredentialVersion(
	namespaceID accesscontrol.NamespaceID,
	credentialID string,
	version providercredential.Version,
) error {
	if err := version.Validate(); err != nil {
		return err
	}
	if version.NamespaceID != string(namespaceID) || version.CredentialID != credentialID ||
		version.Status != providercredential.VersionActive {
		return fmt.Errorf("provider credential version binding or status is invalid")
	}
	return nil
}

func validateProviderCredentialRotation(
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	rotation ProviderCredentialRotation,
	meta MutationMeta,
) error {
	if err := validateProviderCredentialMutation(namespaceID, id, expected, meta); err != nil {
		return err
	}
	if err := validateNewProviderCredentialVersion(namespaceID, id, rotation.Version); err != nil {
		return err
	}
	if err := validateUUID("previous provider credential version id", rotation.PreviousVersionID); err != nil {
		return err
	}
	if rotation.Version.ID == rotation.PreviousVersionID || rotation.RetireAt.IsZero() ||
		!rotation.RetireAt.After(rotation.Version.NotBefore) {
		return fmt.Errorf("provider credential rotation requires a distinct version and bounded overlap")
	}
	return nil
}

func validateProviderCredentialMutation(
	namespaceID accesscontrol.NamespaceID,
	id string,
	expected accesscontrol.Revision,
	meta MutationMeta,
) error {
	if err := validateIdentityIDs(namespaceID, id); err != nil {
		return err
	}
	if _, err := revisionAsInt64(expected); err != nil {
		return err
	}
	return validateMutationMeta(meta)
}

func mapProviderCredentialMutationError(err error, operation string) error {
	if errors.Is(err, sql.ErrNoRows) {
		return ErrRevisionConflict
	}
	if err != nil {
		return fmt.Errorf("%s: %w", operation, err)
	}
	return nil
}

func loadProviderCredentialCommandResult(
	stored managementcommand.StoredResult,
) (MutationResult[providercredential.Credential], error) {
	if stored.Resource == nil || stored.Operation != nil ||
		stored.Resource.ResourceType != "provider_credential" {
		return MutationResult[providercredential.Credential]{}, fmt.Errorf("stored provider credential command result is invalid")
	}
	return MutationResult[providercredential.Credential]{
		ResourceID:       stored.Resource.ResourceID,
		ResourceRevision: accesscontrol.Revision(stored.Resource.ResourceRevision),
		Replayed:         true, ResponseStatus: stored.Resource.ResponseStatus,
	}, nil
}

func validateProviderCredentialCommand(
	command managementcommand.Command,
	namespaceID string,
	meta MutationMeta,
) error {
	if err := command.Validate(time.Now().UTC()); err != nil {
		return err
	}
	if command.Scope.Kind != managementcommand.ScopeNamespace || command.Scope.NamespaceID != namespaceID || meta.ActorPrincipalID == nil ||
		command.PrincipalID != string(*meta.ActorPrincipalID) {
		return fmt.Errorf("provider credential command binding is invalid")
	}
	return nil
}

func validProviderCredentialStatus(status providercredential.Status) bool {
	return status == providercredential.StatusActive || status == providercredential.StatusDisabled ||
		status == providercredential.StatusDeleted
}

func nullProviderCredentialCursorID(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func providerCredentialUniqueViolation(err error) bool {
	var databaseError *pq.Error
	return errors.As(err, &databaseError) && databaseError.Code == "23505"
}

func scanProviderCredential(scanner rowScanner) (providercredential.Credential, error) {
	var credential providercredential.Credential
	var activeVersion sql.NullString
	var revision int64
	var deletedAt sql.NullTime
	if err := scanner.Scan(
		&credential.ID, &credential.NamespaceID, &credential.Name, &credential.ProviderID,
		&credential.CredentialMode, &credential.CredentialAdapterID, &credential.CatalogRevision, &credential.NormalizedOrigin,
		&credential.Status, &activeVersion, &revision,
		&credential.CreatedAt, &credential.UpdatedAt, &deletedAt,
	); err != nil {
		return providercredential.Credential{}, err
	}
	if activeVersion.Valid {
		credential.ActiveVersionID = &activeVersion.String
	}
	if revision <= 0 {
		return providercredential.Credential{}, fmt.Errorf("database returned invalid provider credential revision")
	}
	credential.Revision = uint64(revision)
	credential.DeletedAt = nullTimePointer(deletedAt)
	if err := credential.Validate(); err != nil {
		return providercredential.Credential{}, fmt.Errorf("validate stored provider credential: %w", err)
	}
	return credential, nil
}

func scanProviderCredentialVersion(scanner rowScanner) (providercredential.Version, error) {
	var version providercredential.Version
	var ciphertext, nonce []byte
	var keyVersion sql.NullString
	var expiresAt, revokedAt sql.NullTime
	if err := scanner.Scan(
		&version.ID, &version.NamespaceID, &version.CredentialID,
		&ciphertext, &nonce, &keyVersion, &version.Status, &version.NotBefore,
		&expiresAt, &revokedAt, &version.CreatedAt,
	); err != nil {
		return providercredential.Version{}, err
	}
	version.Envelope = accesscredential.Envelope{
		Ciphertext: ciphertext, Nonce: nonce, KeyVersion: keyVersion.String,
	}
	version.ExpiresAt = nullTimePointer(expiresAt)
	version.RevokedAt = nullTimePointer(revokedAt)
	if err := version.Validate(); err != nil {
		return providercredential.Version{}, fmt.Errorf("validate stored provider credential version: %w", err)
	}
	return version, nil
}
