package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
)

func (s *PostgresStore) ReadPublicationHeads(
	ctx context.Context,
	reference NamespacePublication,
) (PublicationHeads, error) {
	if err := reference.Validate(); err != nil {
		return PublicationHeads{}, err
	}
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return PublicationHeads{}, fmt.Errorf("begin PostgreSQL publication head read: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	var activeID, candidateID sql.NullString
	var partition string
	if err := tx.QueryRowContext(ctx, `SELECT quota_partition_id, active_publication_id, candidate_publication_id
FROM routing_publication_heads WHERE namespace_id=$1`, reference.NamespaceID).Scan(
		&partition, &activeID, &candidateID,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return PublicationHeads{Namespace: reference}, nil
		}
		return PublicationHeads{}, fmt.Errorf("read PostgreSQL publication heads: %w", err)
	}
	if partition != reference.QuotaPartition {
		return PublicationHeads{}, fmt.Errorf("%w: publication namespace partition changed", ErrStagedCorrupt)
	}
	heads := PublicationHeads{Namespace: reference}
	if activeID.Valid {
		identity, err := loadPostgresPublicationIdentity(ctx, tx, reference.NamespaceID, activeID.String)
		if err != nil {
			return PublicationHeads{}, err
		}
		if !identity.Activated() {
			return PublicationHeads{}, fmt.Errorf("%w: active PostgreSQL publication is not active", ErrStagedCorrupt)
		}
		heads.Active = &identity
	}
	if candidateID.Valid && (!activeID.Valid || candidateID.String != activeID.String) {
		identity, err := loadPostgresPublicationIdentity(ctx, tx, reference.NamespaceID, candidateID.String)
		if err != nil {
			return PublicationHeads{}, err
		}
		heads.Candidate = &identity
	}
	if err := tx.Commit(); err != nil {
		return PublicationHeads{}, fmt.Errorf("commit PostgreSQL publication head read: %w", err)
	}
	return heads, nil
}

func loadPostgresPublicationIdentity(
	ctx context.Context,
	executor postgresExecutor,
	namespaceID, publicationID string,
) (RuntimePublicationIdentity, error) {
	var identity RuntimePublicationIdentity
	var revision, epoch int64
	err := executor.QueryRowContext(ctx, `SELECT publication_id, namespace_id, quota_partition_id,
       desired_revision, runtime_epoch, publication_digest, manifest_digest, routing_digest,
       state, restrictive
FROM routing_publications WHERE namespace_id=$1 AND publication_id=$2`, namespaceID, publicationID).Scan(
		&identity.PublicationID, &identity.NamespaceID, &identity.QuotaPartition,
		&revision, &epoch, &identity.PublicationDigest, &identity.ManifestDigest,
		&identity.RoutingDigest, &identity.State, &identity.Restrictive,
	)
	if err != nil {
		return RuntimePublicationIdentity{}, fmt.Errorf("read PostgreSQL publication identity: %w", err)
	}
	if revision <= 0 || epoch <= 0 {
		return RuntimePublicationIdentity{}, fmt.Errorf("%w: PostgreSQL publication identity is invalid", ErrStagedCorrupt)
	}
	identity.DesiredRevision = uint64(revision)
	identity.RuntimeEpoch = uint64(epoch)
	if err := identity.Validate(); err != nil {
		return RuntimePublicationIdentity{}, fmt.Errorf("%w: %w", ErrStagedCorrupt, err)
	}
	return identity, nil
}

func (s *PostgresStore) LoadRoutingPublication(
	ctx context.Context,
	identity RuntimePublicationIdentity,
) (LoadedRoutingPublication, error) {
	if err := identity.Validate(); err != nil {
		return LoadedRoutingPublication{}, err
	}
	if !identity.Loadable() {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: runtime publication is not validated", ErrNotReady)
	}
	var payload []byte
	stored, err := loadPostgresPublicationIdentity(ctx, s.db, identity.NamespaceID, identity.PublicationID)
	if err != nil {
		return LoadedRoutingPublication{}, err
	}
	if !stored.SameGeneration(identity) || !stored.Loadable() {
		return LoadedRoutingPublication{}, ErrPublicationChanged
	}
	if err := s.db.QueryRowContext(ctx, `SELECT publication_blob FROM routing_publications
WHERE namespace_id=$1 AND publication_id=$2`, identity.NamespaceID, identity.PublicationID).Scan(&payload); err != nil {
		return LoadedRoutingPublication{}, fmt.Errorf("read PostgreSQL routing publication: %w", err)
	}
	var publication Publication
	if err := decodeStrict(payload, &publication); err != nil || verifyPublication(publication) != nil {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: PostgreSQL routing publication is invalid", ErrStagedCorrupt)
	}
	if publication.ID != identity.PublicationID || publication.NamespaceID != identity.NamespaceID ||
		publication.QuotaPartition != identity.QuotaPartition || publication.DesiredRevision != identity.DesiredRevision ||
		publication.RuntimeEpoch != identity.RuntimeEpoch || publication.Digest != identity.PublicationDigest ||
		publication.Manifest.Digest != identity.ManifestDigest || publication.Routing.Digest != identity.RoutingDigest {
		return LoadedRoutingPublication{}, fmt.Errorf("%w: PostgreSQL publication envelope changed", ErrStagedCorrupt)
	}
	return LoadedRoutingPublication{
		Identity: stored,
		Manifest: publication.Manifest,
		Routing:  publication.Routing,
		Snapshot: publication.Routing.Snapshot,
	}, nil
}
