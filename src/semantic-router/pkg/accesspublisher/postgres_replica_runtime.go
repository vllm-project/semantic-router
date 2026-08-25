package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

func (s *PostgresStore) RegisterReplica(
	ctx context.Context,
	namespaceID, partition string,
	registration ReplicaRegistration,
) (time.Time, error) {
	if err := registration.Validate(); err != nil {
		return time.Time{}, err
	}
	reference := NamespacePublication{NamespaceID: namespaceID, QuotaPartition: partition}
	if err := reference.Validate(); err != nil {
		return time.Time{}, err
	}
	runtimeEpoch, err := postgresBigint(registration.RuntimeEpoch, "runtime epoch")
	if err != nil {
		return time.Time{}, err
	}
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return time.Time{}, fmt.Errorf("begin PostgreSQL replica registration: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	var epoch int64
	var storedPartition string
	var activeID sql.NullString
	if lockErr := tx.QueryRowContext(ctx, `SELECT n.runtime_epoch, n.quota_partition_id, h.active_publication_id
	FROM access_namespaces n
	LEFT JOIN routing_publication_heads h ON h.namespace_id=n.id
	WHERE n.id=$1 FOR UPDATE OF n`, namespaceID).Scan(&epoch, &storedPartition, &activeID); lockErr != nil {
		return time.Time{}, fmt.Errorf("lock PostgreSQL replica namespace: %w", lockErr)
	}
	if epoch != runtimeEpoch || storedPartition != partition {
		return time.Time{}, ErrEpochMismatch
	}
	if activeID.Valid && (registration.AccessPublication != activeID.String || registration.RoutingPublication != activeID.String) {
		return time.Time{}, ErrPublicationChanged
	}
	if !activeID.Valid && (registration.AccessPublication != "" || registration.RoutingPublication != "") {
		return time.Time{}, ErrPublicationChanged
	}
	var expiry time.Time
	err = tx.QueryRowContext(ctx, `INSERT INTO routing_replica_leases
  (namespace_id, replica_id, runtime_epoch, access_publication_id, routing_publication_id, lease_expires_at)
VALUES ($1,$2,$3,NULLIF($4,''),NULLIF($5,''),clock_timestamp() + ($6 * interval '1 millisecond'))
ON CONFLICT (namespace_id, replica_id) DO UPDATE
SET runtime_epoch=EXCLUDED.runtime_epoch,
    access_publication_id=EXCLUDED.access_publication_id,
    routing_publication_id=EXCLUDED.routing_publication_id,
    lease_expires_at=EXCLUDED.lease_expires_at,
    updated_at=clock_timestamp()
	RETURNING lease_expires_at`, namespaceID, registration.ReplicaID, runtimeEpoch,
		registration.AccessPublication, registration.RoutingPublication, s.replicaLease.Milliseconds()).Scan(&expiry)
	if err != nil {
		return time.Time{}, fmt.Errorf("register PostgreSQL namespace replica: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return time.Time{}, fmt.Errorf("commit PostgreSQL replica registration: %w", err)
	}
	return expiry.UTC(), nil
}

func (s *PostgresStore) AcknowledgeBarriers(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest string,
) error {
	return s.acknowledgePostgres(ctx, namespaceID, partition, replicaID, publicationID, publicationDigest, "barrier")
}

func (s *PostgresStore) AcknowledgeRouting(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest string,
) error {
	return s.acknowledgePostgres(ctx, namespaceID, partition, replicaID, publicationID, publicationDigest, "routing")
}

func (s *PostgresStore) acknowledgePostgres(
	ctx context.Context,
	namespaceID, partition, replicaID, publicationID, publicationDigest, kind string,
) error {
	if strings.TrimSpace(replicaID) == "" || strings.TrimSpace(publicationID) == "" || !validDigest(publicationDigest) {
		return fmt.Errorf("replica, publication, and digest are required")
	}
	if kind != "routing" && kind != "barrier" {
		return fmt.Errorf("publication acknowledgement kind is invalid")
	}
	result, err := s.db.ExecContext(ctx, `INSERT INTO routing_publication_acknowledgements
  (namespace_id, publication_id, replica_id, kind, publication_digest)
SELECT p.namespace_id, p.publication_id, required.replica_id, $6, p.publication_digest
FROM routing_publications p
JOIN routing_publication_heads heads ON heads.namespace_id=p.namespace_id
JOIN routing_publication_required_replicas required
  ON required.namespace_id=p.namespace_id AND required.publication_id=p.publication_id
JOIN access_namespaces n ON n.id=p.namespace_id
WHERE p.namespace_id=$1 AND n.quota_partition_id=$2 AND required.replica_id=$3
  AND p.publication_id=$4 AND p.publication_digest=$5
  AND heads.candidate_publication_id=p.publication_id
  AND p.state IN ('validated','active','applied','finalized')
ON CONFLICT (namespace_id, publication_id, replica_id, kind) DO UPDATE
SET publication_digest=EXCLUDED.publication_digest, acknowledged_at=clock_timestamp()`,
		namespaceID, partition, replicaID, publicationID, publicationDigest, kind)
	if err != nil {
		return fmt.Errorf("acknowledge PostgreSQL publication: %w", err)
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrPublicationChanged
	}
	return nil
}

// VerifyActiveCredentialPublication prevents a direct PostgreSQL credential
// lookup from being used with a stale or unrelated routing capability.
func (s *PostgresStore) VerifyActiveCredentialPublication(
	ctx context.Context,
	identity backendinvoker.CredentialPublication,
) error {
	if err := identity.Validate(); err != nil {
		return err
	}
	var activeID string
	err := s.db.QueryRowContext(ctx, `SELECT heads.active_publication_id
FROM routing_publication_heads heads
JOIN access_namespaces n ON n.id=heads.namespace_id
JOIN routing_publications p
  ON p.namespace_id=heads.namespace_id AND p.publication_id=heads.active_publication_id
WHERE heads.namespace_id=$1 AND heads.quota_partition_id=$2
  AND p.state IN ('active','applied','finalized')`, identity.NamespaceID, identity.QuotaPartition).Scan(&activeID)
	if errors.Is(err, sql.ErrNoRows) {
		return ErrNotReady
	}
	if err != nil {
		return fmt.Errorf("verify active PostgreSQL credential publication: %w", err)
	}
	if activeID != identity.PublicationID {
		return ErrPublicationChanged
	}
	return nil
}

var (
	_ RuntimeStore = (*PostgresStore)(nil)
	_ interface {
		RegisterFleetReplica(context.Context, string) (time.Time, error)
		ListPublicationNamespaces(context.Context) ([]NamespacePublication, error)
		ReadPublicationHeads(context.Context, NamespacePublication) (PublicationHeads, error)
		LoadRoutingPublication(context.Context, RuntimePublicationIdentity) (LoadedRoutingPublication, error)
	} = (*PostgresStore)(nil)
)
