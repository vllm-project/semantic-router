package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (store *Store) ListSnapshots(
	ctx context.Context,
	namespaceID string,
	query routingmanagement.SnapshotListQuery,
) (_ routingmanagement.ListResult[routingmanagement.SnapshotMetadata], returnErr error) {
	if query.Limit < 1 || query.Limit > 200 ||
		(query.BeforeRevision != nil && *query.BeforeRevision <= 0) {
		return routingmanagement.ListResult[routingmanagement.SnapshotMetadata]{},
			fmt.Errorf("%w: routing snapshot list query is invalid", routingmanagement.ErrInvalid)
	}
	var before any
	if query.BeforeRevision != nil {
		before = *query.BeforeRevision
	}
	rows, err := store.db.QueryContext(ctx, `SELECT
  s.namespace_id, s.routing_revision, s.content_digest, s.status, s.failure_reason,
  count(m.resource_id), s.created_at, s.activated_at
FROM routing_snapshots s
LEFT JOIN routing_snapshot_members m
  ON m.namespace_id=s.namespace_id AND m.routing_revision=s.routing_revision
WHERE s.namespace_id=$1 AND ($2::BIGINT IS NULL OR s.routing_revision < $2)
GROUP BY s.namespace_id, s.routing_revision
ORDER BY s.routing_revision DESC
LIMIT $3`, namespaceID, before, query.Limit+1)
	if err != nil {
		return routingmanagement.ListResult[routingmanagement.SnapshotMetadata]{},
			fmt.Errorf("list routing snapshots: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]routingmanagement.SnapshotMetadata, 0, query.Limit+1)
	for rows.Next() {
		metadata, scanErr := scanSnapshotMetadata(rows)
		if scanErr != nil {
			return routingmanagement.ListResult[routingmanagement.SnapshotMetadata]{}, scanErr
		}
		items = append(items, metadata)
	}
	if err := rows.Err(); err != nil {
		return routingmanagement.ListResult[routingmanagement.SnapshotMetadata]{},
			fmt.Errorf("iterate routing snapshots: %w", err)
	}
	result := routingmanagement.ListResult[routingmanagement.SnapshotMetadata]{Items: items}
	if len(result.Items) > query.Limit {
		result.Items = result.Items[:query.Limit]
		result.HasMore = true
	}
	return result, nil
}

func (store *Store) GetSnapshot(
	ctx context.Context,
	namespaceID string,
	routingRevision int64,
) (routingmanagement.SnapshotDetail, error) {
	if routingRevision <= 0 {
		return routingmanagement.SnapshotDetail{}, routingmanagement.ErrInvalid
	}
	return inReadTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.SnapshotDetail, error) {
		var payload []byte
		metadata, getSnapshotErr := loadSnapshotMetadata(ctx, tx, namespaceID, routingRevision, &payload)
		if getSnapshotErr != nil {
			return routingmanagement.SnapshotDetail{}, getSnapshotErr
		}
		members, getSnapshotErr := loadSnapshotMembers(ctx, tx, namespaceID, routingRevision)
		if getSnapshotErr != nil {
			return routingmanagement.SnapshotDetail{}, getSnapshotErr
		}
		var stored routingsnapshot.Snapshot
		if err := strictJSON(payload, &stored); err != nil {
			return routingmanagement.SnapshotDetail{}, corruptSnapshot("compiled export is invalid")
		}
		exported, getSnapshotErr := routingsnapshot.Compile(stored.Bundle)
		if getSnapshotErr != nil || stored.Digest != exported.Digest ||
			exported.NamespaceID != namespaceID || exported.Revision != routingRevision ||
			metadata.ContentDigest != "sha256:"+exported.Digest {
			return routingmanagement.SnapshotDetail{}, corruptSnapshot("compiled export does not match immutable metadata")
		}
		if err := validateSnapshotMembers(*exported, members); err != nil {
			return routingmanagement.SnapshotDetail{}, err
		}
		if metadata.MemberCount != len(members) {
			return routingmanagement.SnapshotDetail{}, corruptSnapshot("member count does not match immutable members")
		}
		return routingmanagement.SnapshotDetail{
			Metadata: metadata,
			Members:  members,
			Export:   *exported,
		}, nil
	})
}

type snapshotMetadataScanner interface {
	Scan(...any) error
}

func scanSnapshotMetadata(row snapshotMetadataScanner) (routingmanagement.SnapshotMetadata, error) {
	var metadata routingmanagement.SnapshotMetadata
	var contentDigest []byte
	var status string
	var failure sql.NullString
	var activated sql.NullTime
	if err := row.Scan(
		&metadata.NamespaceID, &metadata.RoutingRevision, &contentDigest, &status, &failure,
		&metadata.MemberCount, &metadata.CreatedAt, &activated,
	); err != nil {
		return routingmanagement.SnapshotMetadata{}, fmt.Errorf("scan routing snapshot metadata: %w", err)
	}
	metadata.Status = routingmanagement.SnapshotStatus(status)
	if metadata.RoutingRevision <= 0 || metadata.NamespaceID == "" || !metadata.Status.Valid() ||
		metadata.MemberCount < 0 || metadata.CreatedAt.IsZero() {
		return routingmanagement.SnapshotMetadata{}, corruptSnapshot("metadata is invalid")
	}
	var err error
	metadata.ContentDigest, err = publicSnapshotDigest(contentDigest)
	if err != nil {
		return routingmanagement.SnapshotMetadata{}, err
	}
	if failure.Valid {
		metadata.FailureReason = failure.String
	}
	metadata.CreatedAt = metadata.CreatedAt.UTC()
	if activated.Valid {
		value := activated.Time.UTC()
		metadata.ActivatedAt = &value
	}
	return metadata, nil
}

func loadSnapshotMetadata(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	routingRevision int64,
	payload *[]byte,
) (routingmanagement.SnapshotMetadata, error) {
	row := tx.QueryRowContext(ctx, `SELECT
  s.namespace_id, s.routing_revision, s.content_digest, s.status, s.failure_reason,
  (SELECT count(*) FROM routing_snapshot_members m
   WHERE m.namespace_id=s.namespace_id AND m.routing_revision=s.routing_revision),
  s.created_at, s.activated_at, s.compiled_blob
FROM routing_snapshots s
WHERE s.namespace_id=$1 AND s.routing_revision=$2`, namespaceID, routingRevision)
	var metadata routingmanagement.SnapshotMetadata
	var digest []byte
	var status string
	var failure sql.NullString
	var activated sql.NullTime
	if err := row.Scan(
		&metadata.NamespaceID, &metadata.RoutingRevision, &digest, &status, &failure,
		&metadata.MemberCount, &metadata.CreatedAt, &activated, payload,
	); errors.Is(err, sql.ErrNoRows) {
		return routingmanagement.SnapshotMetadata{}, routingmanagement.ErrNotFound
	} else if err != nil {
		return routingmanagement.SnapshotMetadata{}, fmt.Errorf("read routing snapshot metadata: %w", err)
	}
	metadata.Status = routingmanagement.SnapshotStatus(status)
	if metadata.RoutingRevision <= 0 || metadata.NamespaceID == "" || !metadata.Status.Valid() ||
		metadata.MemberCount < 0 || metadata.CreatedAt.IsZero() {
		return routingmanagement.SnapshotMetadata{}, corruptSnapshot("metadata is invalid")
	}
	var err error
	metadata.ContentDigest, err = publicSnapshotDigest(digest)
	if err != nil {
		return routingmanagement.SnapshotMetadata{}, err
	}
	if failure.Valid {
		metadata.FailureReason = failure.String
	}
	metadata.CreatedAt = metadata.CreatedAt.UTC()
	if activated.Valid {
		value := activated.Time.UTC()
		metadata.ActivatedAt = &value
	}
	return metadata, nil
}

func loadSnapshotMembers(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	routingRevision int64,
) (_ []routingmanagement.SnapshotMember, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT resource_type, resource_id, resource_revision
FROM routing_snapshot_members
WHERE namespace_id=$1 AND routing_revision=$2
ORDER BY resource_type, resource_id`, namespaceID, routingRevision)
	if err != nil {
		return nil, fmt.Errorf("list routing snapshot members: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	members := make([]routingmanagement.SnapshotMember, 0)
	for rows.Next() {
		var member routingmanagement.SnapshotMember
		if err := rows.Scan(&member.ResourceType, &member.ResourceID, &member.ResourceRevision); err != nil {
			return nil, fmt.Errorf("scan routing snapshot member: %w", err)
		}
		members = append(members, member)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate routing snapshot members: %w", err)
	}
	return members, nil
}

func validateSnapshotMembers(
	snapshot routingsnapshot.Snapshot,
	members []routingmanagement.SnapshotMember,
) error {
	expected := make(map[string]int64, len(snapshot.Models)+len(snapshot.Recipes)+len(snapshot.Entrypoints))
	for _, model := range snapshot.Models {
		expected["model\x00"+model.ID] = model.Revision
	}
	for _, recipe := range snapshot.Recipes {
		expected["recipe\x00"+recipe.ID] = recipe.Revision
	}
	for _, entrypoint := range snapshot.Entrypoints {
		expected["entrypoint\x00"+entrypoint.ID] = entrypoint.Revision
	}
	if len(expected) != len(members) {
		return corruptSnapshot("member set does not match compiled export")
	}
	seen := make(map[string]struct{}, len(members))
	for _, member := range members {
		key := member.ResourceType + "\x00" + member.ResourceID
		revision, exists := expected[key]
		if !exists || revision != member.ResourceRevision || member.ResourceRevision <= 0 {
			return corruptSnapshot("member set does not match compiled export")
		}
		if _, duplicate := seen[key]; duplicate {
			return corruptSnapshot("member set contains a duplicate")
		}
		seen[key] = struct{}{}
	}
	return nil
}

func publicSnapshotDigest(value []byte) (string, error) {
	if len(value) != 32 {
		return "", corruptSnapshot("content digest is invalid")
	}
	return "sha256:" + hex.EncodeToString(value), nil
}

func corruptSnapshot(reason string) error {
	return fmt.Errorf("%w: routing snapshot %s", routingmanagement.ErrPublication, reason)
}
