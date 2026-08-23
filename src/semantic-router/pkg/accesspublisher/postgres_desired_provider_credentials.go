package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func loadProviderCredentialCandidates(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	bundle routingsnapshot.Bundle,
) (_ []ProviderCredentialCandidate, returnErr error) {
	referenceSet := make(map[string]struct{})
	for _, model := range bundle.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID != "" {
				referenceSet[backend.ProviderCredentialID] = struct{}{}
			}
		}
	}
	if len(referenceSet) == 0 {
		return nil, nil
	}
	references := make([]string, 0, len(referenceSet))
	for credentialID := range referenceSet {
		references = append(references, credentialID)
	}
	sort.Strings(references)

	rows, queryContextErr := tx.QueryContext(ctx, `SELECT
  id, namespace_id, name, provider_id, credential_mode, credential_adapter_id,
  provider_catalog_revision, normalized_origin, status, active_version_id,
  revision, created_at, updated_at, deleted_at
FROM provider_credentials
WHERE namespace_id = $1 AND id = ANY($2::uuid[])
ORDER BY id`, namespace.ID, pq.Array(references))
	if queryContextErr != nil {
		return nil, fmt.Errorf("list referenced provider credentials: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	byID := make(map[string]*ProviderCredentialCandidate, len(references))
	for rows.Next() {
		var credential providercredential.Credential
		var activeVersion sql.NullString
		var revision int64
		var deletedAt sql.NullTime
		if err := rows.Scan(
			&credential.ID, &credential.NamespaceID, &credential.Name, &credential.ProviderID,
			&credential.CredentialMode, &credential.CredentialAdapterID, &credential.CatalogRevision,
			&credential.NormalizedOrigin, &credential.Status, &activeVersion, &revision,
			&credential.CreatedAt, &credential.UpdatedAt, &deletedAt,
		); err != nil {
			return nil, fmt.Errorf("scan referenced provider credential: %w", err)
		}
		if revision <= 0 {
			return nil, fmt.Errorf("provider credential %s revision is invalid", credential.ID)
		}
		credential.Revision = uint64(revision)
		credential.CreatedAt = credential.CreatedAt.UTC()
		credential.UpdatedAt = credential.UpdatedAt.UTC()
		if activeVersion.Valid {
			value := activeVersion.String
			credential.ActiveVersionID = &value
		}
		if deletedAt.Valid {
			value := deletedAt.Time.UTC()
			credential.DeletedAt = &value
		}
		candidate := &ProviderCredentialCandidate{Credential: credential}
		byID[credential.ID] = candidate
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read referenced provider credentials: %w", err)
	}
	for _, credentialID := range references {
		if byID[credentialID] == nil {
			return nil, fmt.Errorf("referenced provider credential %s is absent", credentialID)
		}
	}

	versionRows, queryContextErr := tx.QueryContext(ctx, `SELECT
  id, namespace_id, provider_credential_id, secret_ciphertext, ciphertext_nonce,
  kek_version, status, not_before, expires_at, revoked_at, created_at
FROM provider_credential_versions
WHERE namespace_id = $1 AND provider_credential_id = ANY($2::uuid[])
  AND (status = 'active' OR (status = 'retiring' AND expires_at > CURRENT_TIMESTAMP))
ORDER BY provider_credential_id, status, id`, namespace.ID, pq.Array(references))
	if queryContextErr != nil {
		return nil, fmt.Errorf("list referenced provider credential versions: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, versionRows.Close())
	}()
	for versionRows.Next() {
		var version providercredential.Version
		var expiresAt, revokedAt sql.NullTime
		if err := versionRows.Scan(
			&version.ID, &version.NamespaceID, &version.CredentialID,
			&version.Envelope.Ciphertext, &version.Envelope.Nonce, &version.Envelope.KeyVersion,
			&version.Status, &version.NotBefore, &expiresAt, &revokedAt, &version.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan referenced provider credential version: %w", err)
		}
		candidate := byID[version.CredentialID]
		if candidate == nil {
			return nil, fmt.Errorf("provider credential version references an unpublished credential")
		}
		if len(candidate.Versions) == maximumPublishedProviderCredentialVersions {
			return nil, fmt.Errorf(
				"provider credential %s has more than %d publishable versions",
				version.CredentialID, maximumPublishedProviderCredentialVersions,
			)
		}
		version.NotBefore = version.NotBefore.UTC()
		version.CreatedAt = version.CreatedAt.UTC()
		if expiresAt.Valid {
			value := expiresAt.Time.UTC()
			version.ExpiresAt = &value
		}
		if revokedAt.Valid {
			value := revokedAt.Time.UTC()
			version.RevokedAt = &value
		}
		candidate.Versions = append(candidate.Versions, version)
	}
	if err := versionRows.Err(); err != nil {
		return nil, fmt.Errorf("read referenced provider credential versions: %w", err)
	}

	result := make([]ProviderCredentialCandidate, 0, len(references))
	for _, credentialID := range references {
		result = append(result, *byID[credentialID])
	}
	return result, nil
}
