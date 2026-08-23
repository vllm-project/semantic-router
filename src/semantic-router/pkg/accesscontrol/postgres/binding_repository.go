package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getAccessPolicyBindingQuery = `SELECT b.id, b.namespace_id, b.subject_id, s.kind,
       b.policy_id, b.status, b.revision
FROM access_policy_bindings b
JOIN access_subjects s
  ON s.namespace_id = b.namespace_id AND s.id = b.subject_id
WHERE b.namespace_id = $1 AND b.id = $2`
	insertAccessPolicyBindingQuery = `WITH inserted AS (
  INSERT INTO access_policy_bindings
    (id, namespace_id, subject_id, policy_id, status, revision)
  SELECT $1, $2, $3, $5, $6, 1
  WHERE EXISTS (
    SELECT 1 FROM access_subjects
    WHERE namespace_id = $2 AND id = $3 AND kind = $4
  )
  RETURNING id, namespace_id, subject_id, policy_id, status, revision
)
SELECT i.id, i.namespace_id, i.subject_id, s.kind, i.policy_id, i.status, i.revision
FROM inserted i
JOIN access_subjects s
  ON s.namespace_id = i.namespace_id AND s.id = i.subject_id`
	updateAccessPolicyBindingStatusQuery = `WITH updated AS (
  UPDATE access_policy_bindings
  SET status = $4, revision = revision + 1, updated_at = clock_timestamp()
  WHERE namespace_id = $1 AND id = $2 AND revision = $3
  RETURNING id, namespace_id, subject_id, policy_id, status, revision
)
SELECT u.id, u.namespace_id, u.subject_id, s.kind, u.policy_id, u.status, u.revision
FROM updated u
JOIN access_subjects s
  ON s.namespace_id = u.namespace_id AND s.id = u.subject_id`
	getRateLimitBindingQuery = `SELECT b.id, b.namespace_id, b.subject_id, s.kind,
       b.policy_id, b.binding_mode, b.quota_partition_id, b.status, b.revision
FROM rate_limit_bindings b
JOIN access_subjects s
  ON s.namespace_id = b.namespace_id AND s.id = b.subject_id
WHERE b.namespace_id = $1 AND b.id = $2`
	insertRateLimitBindingQuery = `WITH inserted AS (
  INSERT INTO rate_limit_bindings
    (id, namespace_id, subject_id, policy_id, binding_mode, quota_partition_id, status, revision)
  SELECT $1, $2, $3, $5, $6, $7, $8, 1
  WHERE EXISTS (
    SELECT 1 FROM access_subjects
    WHERE namespace_id = $2 AND id = $3 AND kind = $4
  )
  RETURNING id, namespace_id, subject_id, policy_id,
            binding_mode, quota_partition_id, status, revision
)
SELECT i.id, i.namespace_id, i.subject_id, s.kind, i.policy_id,
       i.binding_mode, i.quota_partition_id, i.status, i.revision
FROM inserted i
JOIN access_subjects s
  ON s.namespace_id = i.namespace_id AND s.id = i.subject_id`
	updateRateLimitBindingStatusQuery = `WITH updated AS (
  UPDATE rate_limit_bindings
  SET status = $4, revision = revision + 1, updated_at = clock_timestamp()
  WHERE namespace_id = $1 AND id = $2 AND revision = $3
  RETURNING id, namespace_id, subject_id, policy_id,
            binding_mode, quota_partition_id, status, revision
)
SELECT u.id, u.namespace_id, u.subject_id, s.kind, u.policy_id,
       u.binding_mode, u.quota_partition_id, u.status, u.revision
FROM updated u
JOIN access_subjects s
  ON s.namespace_id = u.namespace_id AND s.id = u.subject_id`
)

func (s *Store) GetAccessPolicyBinding(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.PolicyBindingID,
) (accesscontrol.AccessPolicyBinding, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return accesscontrol.AccessPolicyBinding{}, err
	}
	binding, err := scanAccessPolicyBinding(s.db.QueryRowContext(ctx, getAccessPolicyBindingQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return accesscontrol.AccessPolicyBinding{}, ErrNotFound
	}
	if err != nil {
		return accesscontrol.AccessPolicyBinding{}, fmt.Errorf("get access-policy binding: %w", err)
	}
	return binding, nil
}

func (s *Store) CreateAccessPolicyBinding(
	ctx context.Context,
	binding accesscontrol.AccessPolicyBinding,
	meta MutationMeta,
) (MutationResult[accesscontrol.AccessPolicyBinding], error) {
	if err := validateAccessPolicyBindingForWrite(binding, 1); err != nil {
		return MutationResult[accesscontrol.AccessPolicyBinding]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.AccessPolicyBinding]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.AccessPolicyBinding], error) {
		created, err := scanAccessPolicyBinding(tx.QueryRowContext(ctx, insertAccessPolicyBindingQuery,
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, ErrNotFound
		}
		if err != nil {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, fmt.Errorf("insert access-policy binding: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, binding.NamespaceID, outboxMutation{
			AggregateType: "access_policy_binding", AggregateID: string(binding.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
			References: map[string]string{
				"policyId": string(created.PolicyID), "subjectId": string(created.Subject.ID),
			},
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, err
		}
		return MutationResult[accesscontrol.AccessPolicyBinding]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) SetAccessPolicyBindingStatus(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.PolicyBindingID,
	expected accesscontrol.Revision,
	status accesscontrol.BindingStatus,
	meta MutationMeta,
) (MutationResult[accesscontrol.AccessPolicyBinding], error) {
	if err := validateBindingStatusMutation(namespaceID, id, expected, status, meta); err != nil {
		return MutationResult[accesscontrol.AccessPolicyBinding]{}, err
	}
	expectedRevision, _ := revisionAsInt64(expected)
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.AccessPolicyBinding], error) {
		updated, err := scanAccessPolicyBinding(tx.QueryRowContext(ctx, updateAccessPolicyBindingStatusQuery,
			namespaceID, id, expectedRevision, status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, fmt.Errorf("update access-policy binding: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "access_policy_binding", AggregateID: string(id),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
			References: map[string]string{
				"policyId": string(updated.PolicyID), "subjectId": string(updated.Subject.ID),
			},
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.AccessPolicyBinding]{}, err
		}
		return MutationResult[accesscontrol.AccessPolicyBinding]{Value: updated, Receipt: receipt}, nil
	})
}

func (s *Store) GetRateLimitBinding(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.PolicyBindingID,
) (accesscontrol.RateLimitBinding, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return accesscontrol.RateLimitBinding{}, err
	}
	binding, err := scanRateLimitBinding(s.db.QueryRowContext(ctx, getRateLimitBindingQuery, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return accesscontrol.RateLimitBinding{}, ErrNotFound
	}
	if err != nil {
		return accesscontrol.RateLimitBinding{}, fmt.Errorf("get rate-limit binding: %w", err)
	}
	return binding, nil
}

func (s *Store) CreateRateLimitBinding(
	ctx context.Context,
	binding accesscontrol.RateLimitBinding,
	meta MutationMeta,
) (MutationResult[accesscontrol.RateLimitBinding], error) {
	if err := validateRateLimitBindingForWrite(binding, 1); err != nil {
		return MutationResult[accesscontrol.RateLimitBinding]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.RateLimitBinding]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.RateLimitBinding], error) {
		created, err := scanRateLimitBinding(tx.QueryRowContext(ctx, insertRateLimitBindingQuery,
			binding.ID, binding.NamespaceID, binding.Subject.ID, binding.Subject.Kind,
			binding.PolicyID, binding.Mode, binding.QuotaPartitionID, binding.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.RateLimitBinding]{}, ErrNotFound
		}
		if err != nil {
			return MutationResult[accesscontrol.RateLimitBinding]{}, fmt.Errorf("insert rate-limit binding: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, binding.NamespaceID, outboxMutation{
			AggregateType: "rate_limit_binding", AggregateID: string(binding.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
			References: map[string]string{
				"policyId": string(created.PolicyID), "subjectId": string(created.Subject.ID),
			},
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.RateLimitBinding]{}, err
		}
		return MutationResult[accesscontrol.RateLimitBinding]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) SetRateLimitBindingStatus(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.PolicyBindingID,
	expected accesscontrol.Revision,
	status accesscontrol.BindingStatus,
	meta MutationMeta,
) (MutationResult[accesscontrol.RateLimitBinding], error) {
	if err := validateBindingStatusMutation(namespaceID, id, expected, status, meta); err != nil {
		return MutationResult[accesscontrol.RateLimitBinding]{}, err
	}
	expectedRevision, _ := revisionAsInt64(expected)
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.RateLimitBinding], error) {
		updated, err := scanRateLimitBinding(tx.QueryRowContext(ctx, updateRateLimitBindingStatusQuery,
			namespaceID, id, expectedRevision, status))
		if errors.Is(err, sql.ErrNoRows) {
			return MutationResult[accesscontrol.RateLimitBinding]{}, ErrRevisionConflict
		}
		if err != nil {
			return MutationResult[accesscontrol.RateLimitBinding]{}, fmt.Errorf("update rate-limit binding: %w", err)
		}
		receipt, err := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "rate_limit_binding", AggregateID: string(id),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
			References: map[string]string{
				"policyId": string(updated.PolicyID), "subjectId": string(updated.Subject.ID),
			},
		}, meta)
		if err != nil {
			return MutationResult[accesscontrol.RateLimitBinding]{}, err
		}
		return MutationResult[accesscontrol.RateLimitBinding]{Value: updated, Receipt: receipt}, nil
	})
}

func validateAccessPolicyBindingForWrite(
	binding accesscontrol.AccessPolicyBinding,
	expected accesscontrol.Revision,
) error {
	if err := binding.Validate(); err != nil {
		return err
	}
	if binding.Revision != expected {
		return fmt.Errorf("access-policy binding revision must match expected revision")
	}
	if err := validateIdentityIDs(binding.NamespaceID, string(binding.ID)); err != nil {
		return err
	}
	if err := validateUUID("binding subject id", string(binding.Subject.ID)); err != nil {
		return err
	}
	return validateUUID("access-policy id", string(binding.PolicyID))
}

func validateRateLimitBindingForWrite(
	binding accesscontrol.RateLimitBinding,
	expected accesscontrol.Revision,
) error {
	if err := binding.Validate(); err != nil {
		return err
	}
	if binding.Revision != expected {
		return fmt.Errorf("rate-limit binding revision must match expected revision")
	}
	if err := validateIdentityIDs(binding.NamespaceID, string(binding.ID)); err != nil {
		return err
	}
	if err := validateUUID("binding subject id", string(binding.Subject.ID)); err != nil {
		return err
	}
	return validateUUID("rate-limit policy id", string(binding.PolicyID))
}

func validateBindingStatusMutation(
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.PolicyBindingID,
	expected accesscontrol.Revision,
	status accesscontrol.BindingStatus,
	meta MutationMeta,
) error {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return err
	}
	if _, err := revisionAsInt64(expected); err != nil {
		return err
	}
	if !status.Valid() {
		return fmt.Errorf("invalid binding status %q", status)
	}
	return validateMutationMeta(meta)
}

func scanAccessPolicyBinding(scanner rowScanner) (accesscontrol.AccessPolicyBinding, error) {
	var binding accesscontrol.AccessPolicyBinding
	var revision int64
	if err := scanner.Scan(
		&binding.ID, &binding.NamespaceID, &binding.Subject.ID, &binding.Subject.Kind,
		&binding.PolicyID, &binding.Status, &revision,
	); err != nil {
		return accesscontrol.AccessPolicyBinding{}, err
	}
	binding.Subject.NamespaceID = binding.NamespaceID
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.AccessPolicyBinding{}, err
	}
	binding.Revision = parsedRevision
	if err := binding.Validate(); err != nil {
		return accesscontrol.AccessPolicyBinding{}, fmt.Errorf("validate stored access-policy binding: %w", err)
	}
	return binding, nil
}

func scanRateLimitBinding(scanner rowScanner) (accesscontrol.RateLimitBinding, error) {
	var binding accesscontrol.RateLimitBinding
	var revision int64
	if err := scanner.Scan(
		&binding.ID, &binding.NamespaceID, &binding.Subject.ID, &binding.Subject.Kind,
		&binding.PolicyID, &binding.Mode, &binding.QuotaPartitionID,
		&binding.Status, &revision,
	); err != nil {
		return accesscontrol.RateLimitBinding{}, err
	}
	binding.Subject.NamespaceID = binding.NamespaceID
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.RateLimitBinding{}, err
	}
	binding.Revision = parsedRevision
	if err := binding.Validate(); err != nil {
		return accesscontrol.RateLimitBinding{}, fmt.Errorf("validate stored rate-limit binding: %w", err)
	}
	return binding, nil
}
