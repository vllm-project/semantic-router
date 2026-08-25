package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

// LoadActivePublishedProviderCredential reads credential material only from the
// immutable PostgreSQL publication named by the dispatch capability. It never
// follows the mutable ProviderCredential active pointer.
func (s *PostgresStore) LoadActivePublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
) (providercredential.Credential, providercredential.Version, error) {
	document, err := s.loadPublishedProviderCredential(ctx, identity, credentialID)
	if err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	return activePublishedProviderCredential(document)
}

// LoadPinnedPublishedProviderCredential reads the requested version from the
// same immutable PostgreSQL publication. A later credential rotation cannot
// substitute a different version while an older Router generation is active.
func (s *PostgresStore) LoadPinnedPublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
	versionID string,
) (providercredential.Credential, providercredential.Version, error) {
	document, err := s.loadPublishedProviderCredential(ctx, identity, credentialID)
	if err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	return pinnedPublishedProviderCredential(document, versionID)
}

func (s *PostgresStore) loadPublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
) (ProviderCredentialDocument, error) {
	if err := identity.Validate(); err != nil {
		return ProviderCredentialDocument{}, err
	}
	if s == nil || s.db == nil {
		return ProviderCredentialDocument{}, providercredential.ErrUnavailable
	}
	var payload []byte
	err := s.db.QueryRowContext(ctx, `SELECT publication.publication_blob
FROM routing_publication_heads heads
JOIN access_namespaces namespace ON namespace.id=heads.namespace_id
JOIN routing_publications publication
  ON publication.namespace_id=heads.namespace_id
 AND publication.publication_id=heads.active_publication_id
WHERE heads.namespace_id=$1
  AND heads.quota_partition_id=$2
  AND heads.active_publication_id=$3
  AND namespace.status='active'
  AND publication.state IN ('active','applied','finalized')`,
		identity.NamespaceID, identity.QuotaPartition, identity.PublicationID,
	).Scan(&payload)
	if errors.Is(err, sql.ErrNoRows) {
		return ProviderCredentialDocument{}, providercredential.ErrUnavailable
	}
	if err != nil {
		return ProviderCredentialDocument{}, fmt.Errorf("read PostgreSQL provider credential publication: %w", err)
	}
	var publication Publication
	if err := decodeStrict(payload, &publication); err != nil ||
		verifyPublication(publication) != nil || publication.ID != identity.PublicationID ||
		publication.NamespaceID != identity.NamespaceID ||
		publication.QuotaPartition != identity.QuotaPartition {
		return ProviderCredentialDocument{}, fmt.Errorf(
			"%w: PostgreSQL provider credential publication is invalid", ErrStagedCorrupt)
	}
	entry, exists := publication.Manifest.ProviderCredentials[credentialID]
	if !exists {
		return ProviderCredentialDocument{}, providercredential.ErrUnavailable
	}
	for _, document := range publication.ProviderCredentials {
		if document.Credential.ID != credentialID {
			continue
		}
		if document.NamespaceID != identity.NamespaceID ||
			document.QuotaPartition != identity.QuotaPartition ||
			document.DesiredRevision != entry.Revision || document.Digest != entry.Digest ||
			verifyProviderCredentialDocument(document) != nil {
			return ProviderCredentialDocument{}, fmt.Errorf(
				"%w: PostgreSQL provider credential document %s is invalid",
				ErrStagedCorrupt, credentialID)
		}
		return document, nil
	}
	return ProviderCredentialDocument{}, fmt.Errorf(
		"%w: PostgreSQL provider credential document %s is absent", ErrStagedCorrupt, credentialID)
}
