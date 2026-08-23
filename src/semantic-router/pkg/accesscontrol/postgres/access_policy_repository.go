package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getAccessPolicyQuery = `SELECT id, namespace_id, name, status, revision, created_at, updated_at
FROM access_policies
WHERE namespace_id = $1 AND id = $2`
	listAccessPolicyGrantsQuery = `SELECT policy_id, resource_type, resource_id, permission, effect
FROM access_policy_grants
WHERE policy_id = $1
ORDER BY resource_type, resource_id, permission, effect`
	insertAccessPolicyQuery = `INSERT INTO access_policies
  (id, namespace_id, name, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, 1, $5, $6)
RETURNING id, namespace_id, name, status, revision, created_at, updated_at`
	updateAccessPolicyQuery = `UPDATE access_policies
SET name = $4, status = $5, revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
RETURNING id, namespace_id, name, status, revision, created_at, updated_at`
	deleteAccessPolicyGrantsQuery = `DELETE FROM access_policy_grants WHERE policy_id = $1`
	insertAccessPolicyGrantQuery  = `INSERT INTO access_policy_grants
  (policy_id, resource_type, resource_id, permission, effect)
VALUES ($1, $2, $3, $4, $5)`
)

func (s *Store) GetAccessPolicy(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.AccessPolicyID,
) (accesscontrol.AccessPolicy, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return accesscontrol.AccessPolicy{}, err
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (accesscontrol.AccessPolicy, error) {
		policy, err := scanAccessPolicy(tx.QueryRowContext(ctx, getAccessPolicyQuery, namespaceID, id))
		if errors.Is(err, sql.ErrNoRows) {
			return accesscontrol.AccessPolicy{}, ErrNotFound
		}
		if err != nil {
			return accesscontrol.AccessPolicy{}, fmt.Errorf("get access policy: %w", err)
		}
		grants, err := listAccessPolicyGrants(ctx, tx, id)
		if err != nil {
			return accesscontrol.AccessPolicy{}, err
		}
		policy.Grants = grants
		if err := policy.Validate(); err != nil {
			return accesscontrol.AccessPolicy{}, fmt.Errorf("validate stored access policy: %w", err)
		}
		return policy, nil
	})
}

func (s *Store) CreateAccessPolicy(
	ctx context.Context,
	policy accesscontrol.AccessPolicy,
	meta MutationMeta,
) (MutationResult[accesscontrol.AccessPolicy], error) {
	if err := validateAccessPolicyForWrite(policy, 1); err != nil {
		return MutationResult[accesscontrol.AccessPolicy]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.AccessPolicy]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.AccessPolicy], error) {
		created, createAccessPolicyErr := scanAccessPolicy(tx.QueryRowContext(ctx, insertAccessPolicyQuery,
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			policy.CreatedAt, policy.UpdatedAt))
		if createAccessPolicyErr != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, fmt.Errorf("insert access policy: %w", createAccessPolicyErr)
		}
		if err := replaceAccessPolicyGrants(ctx, tx, created.ID, policy.Grants, false); err != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, err
		}
		created.Grants = policy.Grants
		receipt, createAccessPolicyErr := appendMutationRecords(ctx, tx, policy.NamespaceID, outboxMutation{
			AggregateType: "access_policy", AggregateID: string(policy.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
		}, meta)
		if createAccessPolicyErr != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, createAccessPolicyErr
		}
		return MutationResult[accesscontrol.AccessPolicy]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateAccessPolicy(
	ctx context.Context,
	policy accesscontrol.AccessPolicy,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[accesscontrol.AccessPolicy], error) {
	if err := validateAccessPolicyForWrite(policy, expected); err != nil {
		return MutationResult[accesscontrol.AccessPolicy]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.AccessPolicy]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.AccessPolicy]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.AccessPolicy], error) {
		updated, updateAccessPolicyErr := scanAccessPolicy(tx.QueryRowContext(ctx, updateAccessPolicyQuery,
			policy.NamespaceID, policy.ID, expectedRevision, policy.DisplayName, policy.Status))
		if errors.Is(updateAccessPolicyErr, sql.ErrNoRows) {
			return MutationResult[accesscontrol.AccessPolicy]{}, ErrRevisionConflict
		}
		if updateAccessPolicyErr != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, fmt.Errorf("update access policy: %w", updateAccessPolicyErr)
		}
		if err := replaceAccessPolicyGrants(ctx, tx, updated.ID, policy.Grants, true); err != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, err
		}
		updated.Grants = policy.Grants
		receipt, updateAccessPolicyErr := appendMutationRecords(ctx, tx, policy.NamespaceID, outboxMutation{
			AggregateType: "access_policy", AggregateID: string(policy.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if updateAccessPolicyErr != nil {
			return MutationResult[accesscontrol.AccessPolicy]{}, updateAccessPolicyErr
		}
		return MutationResult[accesscontrol.AccessPolicy]{Value: updated, Receipt: receipt}, nil
	})
}

func validateAccessPolicyForWrite(policy accesscontrol.AccessPolicy, expected accesscontrol.Revision) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if policy.Revision != expected {
		return fmt.Errorf("access-policy revision must match expected revision")
	}
	if err := validateIdentityIDs(policy.NamespaceID, string(policy.ID)); err != nil {
		return err
	}
	return nil
}

func replaceAccessPolicyGrants(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.AccessPolicyID,
	grants []accesscontrol.AccessPolicyGrant,
	removeExisting bool,
) error {
	if removeExisting {
		if _, err := tx.ExecContext(ctx, deleteAccessPolicyGrantsQuery, policyID); err != nil {
			return fmt.Errorf("replace access-policy grants: %w", err)
		}
	}
	for _, grant := range grants {
		if _, err := tx.ExecContext(ctx, insertAccessPolicyGrantQuery,
			policyID, grant.Resource.Type, grant.Resource.ID, grant.Permission, grant.Effect); err != nil {
			return fmt.Errorf("insert access-policy grant: %w", err)
		}
	}
	return nil
}

func listAccessPolicyGrants(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.AccessPolicyID,
) ([]accesscontrol.AccessPolicyGrant, error) {
	rows, err := tx.QueryContext(ctx, listAccessPolicyGrantsQuery, policyID)
	if err != nil {
		return nil, fmt.Errorf("list access-policy grants: %w", err)
	}
	defer rows.Close()
	grants := make([]accesscontrol.AccessPolicyGrant, 0)
	for rows.Next() {
		var grant accesscontrol.AccessPolicyGrant
		if err := rows.Scan(
			&grant.PolicyID, &grant.Resource.Type, &grant.Resource.ID,
			&grant.Permission, &grant.Effect,
		); err != nil {
			return nil, fmt.Errorf("scan access-policy grant: %w", err)
		}
		grants = append(grants, grant)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate access-policy grants: %w", err)
	}
	return grants, nil
}

func scanAccessPolicy(scanner rowScanner) (accesscontrol.AccessPolicy, error) {
	var policy accesscontrol.AccessPolicy
	var revision int64
	if err := scanner.Scan(
		&policy.ID, &policy.NamespaceID, &policy.DisplayName, &policy.Status,
		&revision, &policy.CreatedAt, &policy.UpdatedAt,
	); err != nil {
		return accesscontrol.AccessPolicy{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.AccessPolicy{}, err
	}
	policy.Revision = parsedRevision
	return policy, nil
}
