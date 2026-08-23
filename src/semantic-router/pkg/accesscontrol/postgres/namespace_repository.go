package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getNamespaceQuery = `SELECT id, name, quota_partition_id, billing_currency, status,
       revision, runtime_epoch, created_at, updated_at
FROM access_namespaces
WHERE id = $1`
	insertNamespaceQuery = `INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING id, name, quota_partition_id, billing_currency, status,
          revision, runtime_epoch, created_at, updated_at`
	updateNamespaceStatusQuery = `UPDATE access_namespaces
SET status = $3, revision = revision + 1, updated_at = clock_timestamp()
WHERE id = $1 AND revision = $2
RETURNING id, name, quota_partition_id, billing_currency, status,
          revision, runtime_epoch, created_at, updated_at`
	insertNamespaceSecurityPolicyQuery = `INSERT INTO management_security_policies
  (namespace_id,action_requirements,seed_version,revision)
VALUES ($1,'{"unknown_usage_fence.waive":{"any_of":[{"kind":"human","human":{"minimum_aal":"aal2","accepted_amr":[],"max_authentication_age_seconds":900}},{"kind":"workload","workload":{"minimum_workload_class":"workload_strong","max_source_age_seconds":2592000}}]}}'::jsonb,1,1)`
)

type rowScanner interface {
	Scan(...any) error
}

func (s *Store) GetNamespace(ctx context.Context, id accesscontrol.NamespaceID) (accesscontrol.Namespace, error) {
	if err := validateUUID("namespace id", string(id)); err != nil {
		return accesscontrol.Namespace{}, err
	}
	namespace, err := scanNamespace(s.db.QueryRowContext(ctx, getNamespaceQuery, id))
	if errors.Is(err, sql.ErrNoRows) {
		return accesscontrol.Namespace{}, ErrNotFound
	}
	if err != nil {
		return accesscontrol.Namespace{}, fmt.Errorf("get namespace: %w", err)
	}
	return namespace, nil
}

func (s *Store) CreateNamespace(
	ctx context.Context,
	namespace accesscontrol.Namespace,
	meta MutationMeta,
) (MutationResult[accesscontrol.Namespace], error) {
	if err := namespace.Validate(); err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	if namespace.Revision != 1 {
		return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("new namespace revision must be 1")
	}
	if err := validateUUID("namespace id", string(namespace.ID)); err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	revision, err := revisionAsInt64(namespace.Revision)
	if err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	runtimeEpoch, err := revisionAsInt64(accesscontrol.Revision(namespace.RuntimeEpoch))
	if err != nil {
		return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("runtime epoch: %w", err)
	}

	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.Namespace], error) {
		created, createNamespaceErr := scanNamespace(tx.QueryRowContext(ctx, insertNamespaceQuery,
			namespace.ID, namespace.Name, namespace.QuotaPartitionID, namespace.BillingCurrency,
			namespace.Status, revision, runtimeEpoch, namespace.CreatedAt, namespace.UpdatedAt))
		if createNamespaceErr != nil {
			return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("insert namespace: %w", createNamespaceErr)
		}
		if _, err := tx.ExecContext(ctx, insertNamespaceSecurityPolicyQuery, created.ID); err != nil {
			return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("insert namespace security policy: %w", err)
		}
		receipt, createNamespaceErr := appendMutationRecords(ctx, tx, created.ID, outboxMutation{
			AggregateType: "namespace", AggregateID: string(created.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
		}, meta)
		if createNamespaceErr != nil {
			return MutationResult[accesscontrol.Namespace]{}, createNamespaceErr
		}
		return MutationResult[accesscontrol.Namespace]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) SetNamespaceStatus(
	ctx context.Context,
	id accesscontrol.NamespaceID,
	expected accesscontrol.Revision,
	status accesscontrol.NamespaceStatus,
	meta MutationMeta,
) (MutationResult[accesscontrol.Namespace], error) {
	if !status.Valid() {
		return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("invalid namespace status %q", status)
	}
	if err := validateUUID("namespace id", string(id)); err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.Namespace]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.Namespace], error) {
		updated, err := scanNamespace(tx.QueryRowContext(ctx, updateNamespaceStatusQuery, id, expectedRevision, status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.Namespace]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[accesscontrol.Namespace]{}, fmt.Errorf("update namespace status: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, id, outboxMutation{
			AggregateType: "namespace", AggregateID: string(id),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.Namespace]{}, err
		}
		return MutationResult[accesscontrol.Namespace]{Value: updated, Receipt: receipt}, nil
	})
}

func scanNamespace(scanner rowScanner) (accesscontrol.Namespace, error) {
	var namespace accesscontrol.Namespace
	var revision, runtimeEpoch int64
	if err := scanner.Scan(
		&namespace.ID, &namespace.Name, &namespace.QuotaPartitionID, &namespace.BillingCurrency,
		&namespace.Status, &revision, &runtimeEpoch, &namespace.CreatedAt, &namespace.UpdatedAt,
	); err != nil {
		return accesscontrol.Namespace{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.Namespace{}, err
	}
	if runtimeEpoch <= 0 {
		return accesscontrol.Namespace{}, fmt.Errorf("database returned invalid runtime epoch %d", runtimeEpoch)
	}
	namespace.Revision = parsedRevision
	namespace.RuntimeEpoch = uint64(runtimeEpoch)
	return namespace, nil
}
