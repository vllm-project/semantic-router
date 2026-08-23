package accesspublisher

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

// LoadActivePublishedProviderCredential resolves the exact immutable document
// named by a dispatch capability. It never follows a mutable active pointer.
func (s *RedisStore) LoadActivePublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
) (providercredential.Credential, providercredential.Version, error) {
	document, err := s.loadPublishedProviderCredential(ctx, identity, credentialID)
	if err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	if document.Credential.Status != providercredential.StatusActive || document.Credential.ActiveVersionID == nil {
		return providercredential.Credential{}, providercredential.Version{}, providercredential.ErrUnavailable
	}
	for _, version := range document.Versions {
		if version.ID == *document.Credential.ActiveVersionID && version.Status == providercredential.VersionActive {
			return document.Credential, version, nil
		}
	}
	return providercredential.Credential{}, providercredential.Version{}, providercredential.ErrUnavailable
}

// LoadPinnedPublishedProviderCredential retrieves only the requested version
// from the same immutable publication used by the routing plan. It never
// substitutes the current active version or another publication.
func (s *RedisStore) LoadPinnedPublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
	versionID string,
) (providercredential.Credential, providercredential.Version, error) {
	document, err := s.loadPublishedProviderCredential(ctx, identity, credentialID)
	if err != nil {
		return providercredential.Credential{}, providercredential.Version{}, err
	}
	for _, version := range document.Versions {
		if version.ID == versionID {
			return document.Credential, version, nil
		}
	}
	return providercredential.Credential{}, providercredential.Version{}, providercredential.ErrUnavailable
}

func (s *RedisStore) loadPublishedProviderCredential(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
	credentialID string,
) (ProviderCredentialDocument, error) {
	if err := identity.Validate(); err != nil {
		return ProviderCredentialDocument{}, err
	}
	keys, err := NewKeyspace(s.keyPrefix, identity.NamespaceID, identity.QuotaPartition)
	if err != nil {
		return ProviderCredentialDocument{}, err
	}
	manifestPayload, err := s.client.Get(ctx, keys.Manifest(identity.PublicationID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return ProviderCredentialDocument{}, providercredential.ErrUnavailable
	}
	if err != nil {
		return ProviderCredentialDocument{}, fmt.Errorf("read provider credential manifest: %w", err)
	}
	var manifest Manifest
	if err := decodeStrict(manifestPayload, &manifest); err != nil ||
		manifest.PublicationID != identity.PublicationID || manifest.NamespaceID != identity.NamespaceID ||
		manifest.QuotaPartition != identity.QuotaPartition {
		return ProviderCredentialDocument{}, fmt.Errorf("%w: provider credential manifest identity is invalid", ErrStagedCorrupt)
	}
	if err := verifyManifest(manifest); err != nil {
		return ProviderCredentialDocument{}, err
	}
	entry, exists := manifest.ProviderCredentials[credentialID]
	if !exists {
		return ProviderCredentialDocument{}, providercredential.ErrUnavailable
	}
	return s.loadProviderCredentialDocument(ctx, keys, identity.PublicationID, credentialID, entry)
}

func (s *RedisStore) loadProviderCredentialDocument(
	ctx context.Context,
	keys Keyspace,
	publicationID string,
	credentialID string,
	entry ManifestEntry,
) (ProviderCredentialDocument, error) {
	payload, err := s.client.Get(ctx, keys.ProviderCredentialDocument(publicationID, credentialID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return ProviderCredentialDocument{}, fmt.Errorf("%w: provider credential document %s is absent", ErrStagedCorrupt, credentialID)
	}
	if err != nil {
		return ProviderCredentialDocument{}, fmt.Errorf("read provider credential document: %w", err)
	}
	var document ProviderCredentialDocument
	if err := decodeStrict(payload, &document); err != nil || document.Digest != entry.Digest ||
		document.DesiredRevision != entry.Revision || document.NamespaceID != keys.namespaceID ||
		document.QuotaPartition != keys.partition || document.Credential.ID != credentialID {
		return ProviderCredentialDocument{}, fmt.Errorf("%w: provider credential document %s is invalid", ErrStagedCorrupt, credentialID)
	}
	if err := verifyProviderCredentialDocument(document); err != nil {
		return ProviderCredentialDocument{}, err
	}
	return document, nil
}
