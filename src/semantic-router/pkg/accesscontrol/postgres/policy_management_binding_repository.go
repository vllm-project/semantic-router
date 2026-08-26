package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	managedAccessBindingColumns = `b.id, b.namespace_id, b.policy_id, b.subject_id,
       s.kind, b.status, b.revision, b.created_at, b.updated_at`
	managedRateBindingColumns = `b.id, b.namespace_id, b.policy_id, b.subject_id,
       s.kind, b.binding_mode, b.quota_partition_id, b.status, b.revision,
       b.created_at, b.updated_at`

	getManagedAccessBindingQuery = `SELECT ` + managedAccessBindingColumns + `
FROM access_policy_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND b.id=$2`
	listManagedAccessBindingsQuery = `SELECT ` + managedAccessBindingColumns + `
FROM access_policy_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND ($2='' OR b.policy_id=NULLIF($2,'')::uuid)
  AND ($3='' OR s.kind=$3) AND (NULLIF($4,'')::uuid IS NULL OR b.subject_id=NULLIF($4,'')::uuid)
  AND ($5='' OR b.status=$5)
  AND ($6 OR b.policy_id=ANY($7::uuid[]))
  AND ($8::timestamptz IS NULL OR b.created_at < $8 OR (b.created_at=$8 AND b.id > $9::uuid))
ORDER BY b.created_at DESC, b.id ASC LIMIT $10`
	countFilteredManagedAccessBindingsQuery = `SELECT count(*)
FROM access_policy_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND ($2='' OR b.policy_id=NULLIF($2,'')::uuid)
  AND ($3='' OR s.kind=$3) AND (NULLIF($4,'')::uuid IS NULL OR b.subject_id=NULLIF($4,'')::uuid)
  AND ($5='' OR b.status=$5)
  AND ($6 OR b.policy_id=ANY($7::uuid[]))`
	lockManagedAccessBindingQuery   = getManagedAccessBindingQuery + ` FOR UPDATE OF b`
	insertManagedAccessBindingQuery = `INSERT INTO access_policy_bindings
  (id,namespace_id,policy_id,subject_id,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)
RETURNING id`
	updateManagedAccessBindingQuery = `UPDATE access_policy_bindings
SET status=$4, revision=revision+1, updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3`
	deleteManagedAccessBindingQuery = `DELETE FROM access_policy_bindings
WHERE namespace_id=$1 AND id=$2 AND revision=$3`

	getManagedRateBindingQuery = `SELECT ` + managedRateBindingColumns + `
FROM rate_limit_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND b.id=$2`
	listManagedRateBindingsQuery = `SELECT ` + managedRateBindingColumns + `
FROM rate_limit_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND ($2='' OR b.policy_id=NULLIF($2,'')::uuid)
  AND ($3='' OR s.kind=$3) AND (NULLIF($4,'')::uuid IS NULL OR b.subject_id=NULLIF($4,'')::uuid)
  AND ($5='' OR b.status=$5) AND ($6='' OR b.binding_mode=$6)
  AND ($7 OR b.policy_id=ANY($8::uuid[]))
  AND ($9::timestamptz IS NULL OR b.created_at < $9 OR (b.created_at=$9 AND b.id > $10::uuid))
ORDER BY b.created_at DESC, b.id ASC LIMIT $11`
	countFilteredManagedRateBindingsQuery = `SELECT count(*)
FROM rate_limit_bindings b
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND ($2='' OR b.policy_id=NULLIF($2,'')::uuid)
  AND ($3='' OR s.kind=$3) AND (NULLIF($4,'')::uuid IS NULL OR b.subject_id=NULLIF($4,'')::uuid)
  AND ($5='' OR b.status=$5) AND ($6='' OR b.binding_mode=$6)
  AND ($7 OR b.policy_id=ANY($8::uuid[]))`
	lockManagedRateBindingQuery   = getManagedRateBindingQuery + ` FOR UPDATE OF b`
	insertManagedRateBindingQuery = `INSERT INTO rate_limit_bindings
  (id,namespace_id,policy_id,subject_id,binding_mode,quota_partition_id,
   status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,'active',1,$7,$7)
RETURNING id`
	updateManagedRateBindingQuery = `UPDATE rate_limit_bindings
SET status=$4, revision=revision+1, updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3`
	deleteManagedRateBindingQuery = `DELETE FROM rate_limit_bindings
WHERE namespace_id=$1 AND id=$2 AND revision=$3`

	lockManagedActiveAccessPolicyQuery = `SELECT id FROM access_policies
WHERE namespace_id=$1 AND id=$2 AND status='active' FOR KEY SHARE`
	lockManagedActiveRatePolicyQuery = `SELECT id FROM rate_limit_policies
WHERE namespace_id=$1 AND id=$2 AND status='active' FOR KEY SHARE`
	lockManagedUserSubjectQuery = `SELECT id FROM access_users
WHERE namespace_id=$1 AND id=$2 AND deleted_at IS NULL FOR KEY SHARE`
	lockManagedTeamSubjectQuery = `SELECT id FROM access_teams
WHERE namespace_id=$1 AND id=$2 AND deleted_at IS NULL FOR KEY SHARE`
	lockManagedKeySubjectQuery = `SELECT id FROM access_api_keys
WHERE namespace_id=$1 AND id=$2 AND deleted_at IS NULL FOR KEY SHARE`
	lockManagedQuotaPartitionQuery = `SELECT quota_partition_id FROM access_namespaces
WHERE id=$1 AND status='active' FOR KEY SHARE`
	managedBindingFenceQuery = `SELECT EXISTS (
  SELECT 1 FROM unknown_usage_fence_bindings fb
  JOIN unknown_usage_fences f
    ON f.id=fb.fence_id
  WHERE f.namespace_id=$1 AND fb.binding_id=$2 AND f.state IN ('open','reconciling')
)`
)

func (s *Store) GetManagedAccessBinding(ctx context.Context, namespaceID, bindingID string) (policymanagement.AccessPolicyBinding, error) {
	if validateManagedPolicyIDs(namespaceID, bindingID) != nil {
		return policymanagement.AccessPolicyBinding{}, policymanagement.ErrInvalidRequest
	}
	binding, err := scanManagedAccessBinding(s.db.QueryRowContext(ctx, getManagedAccessBindingQuery, namespaceID, bindingID))
	if err != nil {
		return policymanagement.AccessPolicyBinding{}, mapManagedPolicyRead(err, "get AccessPolicy binding")
	}
	return binding, nil
}

func (s *Store) ListManagedAccessBindings(
	ctx context.Context,
	query policymanagement.BindingQuery,
) (policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding], error) {
	if validateManagedBindingQuery(query, false) != nil {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding]{}, policymanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)) == 0 {
		return emptyManagedBindingPage[policymanagement.AccessPolicyBinding](query.IncludeTotal), nil
	}
	afterTime, afterID := managedPolicyCursorArgs(query.After)
	subjectType, subjectID := managedSubjectArgs(query.Subject)
	totalCount, err := s.countManagedAccessBindings(ctx, query, subjectType, subjectID)
	if err != nil {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding]{}, err
	}
	rows, err := s.db.QueryContext(ctx, listManagedAccessBindingsQuery, query.NamespaceID,
		query.PolicyID, subjectType, subjectID, query.Status, query.Scope.All,
		pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)),
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding]{}, fmt.Errorf("list AccessPolicy bindings: %w", err)
	}
	defer rows.Close()
	items := make([]policymanagement.AccessPolicyBinding, 0, query.Limit+1)
	for rows.Next() {
		item, scanErr := scanManagedAccessBinding(rows)
		if scanErr != nil {
			return policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding]{}, fmt.Errorf("scan AccessPolicy binding page: %w", scanErr)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicyBinding]{}, fmt.Errorf("read AccessPolicy binding page: %w", err)
	}
	page := trimManagedPage(items, query.Limit)
	page.TotalCount = totalCount
	return page, nil
}

func (s *Store) CreateManagedAccessBinding(
	ctx context.Context,
	mutation policymanagement.CreateAccessBindingMutation,
) (policymanagement.MutationResult, error) {
	if validateNewManagedAccessBinding(mutation.Binding) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(mutation.Actor, "access_policy_binding.create",
		"Create AccessPolicy binding.", managedBindingAuditDetails(mutation.Binding.PolicyID, mutation.Binding.Subject))
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		if replay, ok, err := lockManagedPolicyCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		created, err := materializeManagedAccessBinding(ctx, tx, mutation.Binding)
		if err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, created.NamespaceID, "access_policy_binding",
			created.ID, created.Revision, outboxCreated, meta, managedBindingReferences(created.PolicyID, created.Subject)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return completeManagedPolicyCommand(ctx, tx, mutation.Command, "access_policy_binding", created.ID, created.Revision, 201)
	})
}

func (s *Store) UpdateManagedAccessBinding(
	ctx context.Context,
	namespaceID, bindingID string,
	expected uint64,
	status accesscontrol.BindingStatus,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedBindingMutation(namespaceID, bindingID, expected, status, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "access_policy_binding.update", "Update AccessPolicy binding.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, updateManagedAccessBindingErr := scanManagedAccessBinding(tx.QueryRowContext(ctx, lockManagedAccessBindingQuery, namespaceID, bindingID))
		if updateManagedAccessBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(updateManagedAccessBindingErr, "lock AccessPolicy binding")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if current.Status == status {
			return policymanagement.MutationResult{
				Kind: "access_policy_binding", ID: current.ID,
				Revision: current.Revision, HTTPStatus: 200,
			}, nil
		}
		if status == accesscontrol.BindingStatusActive {
			if err := lockManagedBindingReferences(ctx, tx, namespaceID, current.PolicyID, current.Subject, false); err != nil {
				return policymanagement.MutationResult{}, err
			}
		}
		result, updateManagedAccessBindingErr := tx.ExecContext(ctx, updateManagedAccessBindingQuery, namespaceID, bindingID, expected, status)
		if updateManagedAccessBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyCAS(updateManagedAccessBindingErr, "update AccessPolicy binding")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "access_policy_binding", bindingID,
			revision, outboxUpdated, meta, managedBindingReferences(current.PolicyID, current.Subject)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "access_policy_binding", ID: bindingID,
			Revision: revision, HTTPStatus: 200,
		}, nil
	})
}

func (s *Store) DeleteManagedAccessBinding(
	ctx context.Context,
	namespaceID, bindingID string,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedPolicyMutation(namespaceID, bindingID, expected, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "access_policy_binding.delete", "Delete AccessPolicy binding.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, err := scanManagedAccessBinding(tx.QueryRowContext(ctx, lockManagedAccessBindingQuery, namespaceID, bindingID))
		if err != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(err, "lock AccessPolicy binding")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		result, err := tx.ExecContext(ctx, deleteManagedAccessBindingQuery, namespaceID, bindingID, expected)
		if err != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyDelete(err, "delete AccessPolicy binding")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "access_policy_binding", bindingID,
			revision, outboxDeleted, meta, managedBindingReferences(current.PolicyID, current.Subject)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "access_policy_binding", ID: bindingID,
			Revision: revision, HTTPStatus: 204,
		}, nil
	})
}

func (s *Store) GetManagedRateBinding(ctx context.Context, namespaceID, bindingID string) (policymanagement.RateLimitBinding, error) {
	if validateManagedPolicyIDs(namespaceID, bindingID) != nil {
		return policymanagement.RateLimitBinding{}, policymanagement.ErrInvalidRequest
	}
	binding, err := scanManagedRateBinding(s.db.QueryRowContext(ctx, getManagedRateBindingQuery, namespaceID, bindingID))
	if err != nil {
		return policymanagement.RateLimitBinding{}, mapManagedPolicyRead(err, "get RateLimit binding")
	}
	return binding, nil
}

func (s *Store) ListManagedRateBindings(
	ctx context.Context,
	query policymanagement.BindingQuery,
) (policymanagement.RepositoryPage[policymanagement.RateLimitBinding], error) {
	if validateManagedBindingQuery(query, true) != nil {
		return policymanagement.RepositoryPage[policymanagement.RateLimitBinding]{}, policymanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)) == 0 {
		return emptyManagedBindingPage[policymanagement.RateLimitBinding](query.IncludeTotal), nil
	}
	afterTime, afterID := managedPolicyCursorArgs(query.After)
	subjectType, subjectID := managedSubjectArgs(query.Subject)
	totalCount, err := s.countManagedRateBindings(ctx, query, subjectType, subjectID)
	if err != nil {
		return policymanagement.RepositoryPage[policymanagement.RateLimitBinding]{}, err
	}
	rows, err := s.db.QueryContext(ctx, listManagedRateBindingsQuery, query.NamespaceID,
		query.PolicyID, subjectType, subjectID, query.Status, query.Mode,
		query.Scope.All,
		pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)),
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return policymanagement.RepositoryPage[policymanagement.RateLimitBinding]{}, fmt.Errorf("list RateLimit bindings: %w", err)
	}
	defer rows.Close()
	items := make([]policymanagement.RateLimitBinding, 0, query.Limit+1)
	for rows.Next() {
		item, scanErr := scanManagedRateBinding(rows)
		if scanErr != nil {
			return policymanagement.RepositoryPage[policymanagement.RateLimitBinding]{}, fmt.Errorf("scan RateLimit binding page: %w", scanErr)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return policymanagement.RepositoryPage[policymanagement.RateLimitBinding]{}, fmt.Errorf("read RateLimit binding page: %w", err)
	}
	page := trimManagedPage(items, query.Limit)
	page.TotalCount = totalCount
	return page, nil
}

func emptyManagedBindingPage[T any](includeTotal bool) policymanagement.RepositoryPage[T] {
	page := policymanagement.RepositoryPage[T]{Items: []T{}}
	if includeTotal {
		count := uint64(0)
		page.TotalCount = &count
	}
	return page
}

func (s *Store) countManagedAccessBindings(
	ctx context.Context,
	query policymanagement.BindingQuery,
	subjectType, subjectID any,
) (*uint64, error) {
	if !query.IncludeTotal {
		return nil, nil
	}
	var count int64
	if err := s.db.QueryRowContext(ctx, countFilteredManagedAccessBindingsQuery, query.NamespaceID,
		query.PolicyID, subjectType, subjectID, query.Status, query.Scope.All,
		pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy))).Scan(&count); err != nil {
		return nil, fmt.Errorf("count AccessPolicy bindings: %w", err)
	}
	return managedBindingCount(count)
}

func (s *Store) countManagedRateBindings(
	ctx context.Context,
	query policymanagement.BindingQuery,
	subjectType, subjectID any,
) (*uint64, error) {
	if !query.IncludeTotal {
		return nil, nil
	}
	var count int64
	if err := s.db.QueryRowContext(ctx, countFilteredManagedRateBindingsQuery, query.NamespaceID,
		query.PolicyID, subjectType, subjectID, query.Status, query.Mode, query.Scope.All,
		pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy))).Scan(&count); err != nil {
		return nil, fmt.Errorf("count RateLimit bindings: %w", err)
	}
	return managedBindingCount(count)
}

func managedBindingCount(value int64) (*uint64, error) {
	if value < 0 {
		return nil, errors.New("policy binding count is negative")
	}
	count := uint64(value)
	return &count, nil
}

func (s *Store) CreateManagedRateBinding(
	ctx context.Context,
	mutation policymanagement.CreateRateBindingMutation,
) (policymanagement.MutationResult, error) {
	if validateNewManagedRateBinding(mutation.Binding) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(mutation.Actor, "rate_limit_binding.create",
		"Create RateLimit binding.", managedBindingAuditDetails(mutation.Binding.PolicyID, mutation.Binding.Subject))
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		if replay, ok, err := lockManagedPolicyCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		created, err := materializeManagedRateBinding(ctx, tx, mutation.Binding)
		if err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, created.NamespaceID, "rate_limit_binding",
			created.ID, created.Revision, outboxCreated, meta, managedRateBindingReferences(created)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return completeManagedPolicyCommand(ctx, tx, mutation.Command, "rate_limit_binding", created.ID, created.Revision, 201)
	})
}

func (s *Store) CreateManagedInlineRateBinding(
	ctx context.Context,
	mutation policymanagement.CreateInlineRateBindingMutation,
) (policymanagement.InlineRateBindingResult, error) {
	if validateNewManagedRatePolicy(mutation.Policy) != nil || validateNewManagedRateBinding(mutation.Binding) != nil ||
		mutation.Binding.PolicyID != mutation.Policy.ID || mutation.Binding.NamespaceID != mutation.Policy.NamespaceID {
		return policymanagement.InlineRateBindingResult{}, policymanagement.ErrInvalidRequest
	}
	policyMeta, err := managedPolicyMutationMeta(mutation.Actor, "rate_limit_policy.create",
		"Create reusable inline RateLimitPolicy.", nil)
	if err != nil {
		return policymanagement.InlineRateBindingResult{}, err
	}
	bindingMeta, err := managedPolicyMutationMeta(mutation.Actor, "rate_limit_binding.create",
		"Bind reusable inline RateLimitPolicy.", managedBindingAuditDetails(mutation.Binding.PolicyID, mutation.Binding.Subject))
	if err != nil {
		return policymanagement.InlineRateBindingResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.InlineRateBindingResult, error) {
		if replay, ok, err := lockManagedPolicyCommand(ctx, tx, mutation.Command); err != nil {
			return policymanagement.InlineRateBindingResult{}, err
		} else if ok {
			binding, getErr := scanManagedRateBinding(tx.QueryRowContext(ctx, getManagedRateBindingQuery,
				mutation.Policy.NamespaceID, replay.ID))
			if getErr != nil {
				return policymanagement.InlineRateBindingResult{}, mapManagedPolicyRead(getErr, "replay inline RateLimit binding")
			}
			policy, getErr := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, getManagedRatePolicyQuery,
				mutation.Policy.NamespaceID, binding.PolicyID))
			if getErr != nil {
				return policymanagement.InlineRateBindingResult{}, mapManagedPolicyRead(getErr, "replay inline RateLimitPolicy")
			}
			return policymanagement.InlineRateBindingResult{
				Policy: policy, Binding: binding, Created: true, MutationResult: replay,
			}, nil
		}
		materialized, createManagedInlineRateBindingErr := materializeManagedInlineRateLimit(ctx, tx, mutation.Policy, mutation.Binding)
		if createManagedInlineRateBindingErr != nil {
			return policymanagement.InlineRateBindingResult{}, createManagedInlineRateBindingErr
		}
		policy, binding := materialized.Policy, materialized.Binding
		mutations := []compoundMutation{
			{Mutation: outboxMutation{
				AggregateType: "rate_limit_policy", AggregateID: policy.ID,
				AggregateRevision: accesscontrol.Revision(policy.Revision), Operation: outboxCreated,
			}, Meta: policyMeta},
			{Mutation: outboxMutation{
				AggregateType: "rate_limit_binding", AggregateID: binding.ID,
				AggregateRevision: accesscontrol.Revision(binding.Revision), Operation: outboxCreated,
				References: managedRateBindingReferences(binding),
			}, Meta: bindingMeta},
		}
		if _, err := appendCompoundMutationRecords(ctx, tx, mutation.Policy.NamespaceID, mutations); err != nil {
			return policymanagement.InlineRateBindingResult{}, err
		}
		result, createManagedInlineRateBindingErr := completeManagedPolicyCommand(ctx, tx, mutation.Command,
			"rate_limit_binding", binding.ID, binding.Revision, 201)
		return policymanagement.InlineRateBindingResult{
			Policy: policy, Binding: binding, Created: true, MutationResult: result,
		}, createManagedInlineRateBindingErr
	})
}

func (s *Store) UpdateManagedRateBinding(
	ctx context.Context,
	namespaceID, bindingID string,
	expected uint64,
	status accesscontrol.BindingStatus,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedBindingMutation(namespaceID, bindingID, expected, status, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "rate_limit_binding.update", "Update RateLimit binding.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, updateManagedRateBindingErr := scanManagedRateBinding(tx.QueryRowContext(ctx, lockManagedRateBindingQuery, namespaceID, bindingID))
		if updateManagedRateBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(updateManagedRateBindingErr, "lock RateLimit binding")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if current.Status == status {
			return policymanagement.MutationResult{
				Kind: "rate_limit_binding", ID: current.ID,
				Revision: current.Revision, HTTPStatus: 200,
			}, nil
		}
		if fenced, err := managedRateBindingFenced(ctx, tx, namespaceID, bindingID); err != nil {
			return policymanagement.MutationResult{}, err
		} else if fenced {
			return policymanagement.MutationResult{}, policymanagement.ErrUnknownUsageFence
		}
		if status == accesscontrol.BindingStatusActive {
			if err := lockManagedBindingReferences(ctx, tx, namespaceID, current.PolicyID, current.Subject, true); err != nil {
				return policymanagement.MutationResult{}, err
			}
		}
		result, updateManagedRateBindingErr := tx.ExecContext(ctx, updateManagedRateBindingQuery, namespaceID, bindingID, expected, status)
		if updateManagedRateBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedRateBindingWrite(updateManagedRateBindingErr, "update RateLimit binding")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "rate_limit_binding", bindingID,
			revision, outboxUpdated, meta, managedRateBindingReferences(current)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "rate_limit_binding", ID: bindingID,
			Revision: revision, HTTPStatus: 200,
		}, nil
	})
}

func (s *Store) DeleteManagedRateBinding(
	ctx context.Context,
	namespaceID, bindingID string,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedPolicyMutation(namespaceID, bindingID, expected, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "rate_limit_binding.delete", "Delete RateLimit binding.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, deleteManagedRateBindingErr := scanManagedRateBinding(tx.QueryRowContext(ctx, lockManagedRateBindingQuery, namespaceID, bindingID))
		if deleteManagedRateBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(deleteManagedRateBindingErr, "lock RateLimit binding")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if fenced, err := managedRateBindingFenced(ctx, tx, namespaceID, bindingID); err != nil {
			return policymanagement.MutationResult{}, err
		} else if fenced {
			return policymanagement.MutationResult{}, policymanagement.ErrUnknownUsageFence
		}
		result, deleteManagedRateBindingErr := tx.ExecContext(ctx, deleteManagedRateBindingQuery, namespaceID, bindingID, expected)
		if deleteManagedRateBindingErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyDelete(deleteManagedRateBindingErr, "delete RateLimit binding")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "rate_limit_binding", bindingID,
			revision, outboxDeleted, meta, managedRateBindingReferences(current)); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "rate_limit_binding", ID: bindingID,
			Revision: revision, HTTPStatus: 204,
		}, nil
	})
}

func createManagedRateBinding(
	ctx context.Context,
	tx *sql.Tx,
	binding policymanagement.RateLimitBinding,
) (policymanagement.RateLimitBinding, error) {
	if err := lockManagedBindingReferences(ctx, tx, binding.NamespaceID, binding.PolicyID, binding.Subject, true); err != nil {
		return policymanagement.RateLimitBinding{}, err
	}
	var partition string
	if err := tx.QueryRowContext(ctx, lockManagedQuotaPartitionQuery, binding.NamespaceID).Scan(&partition); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return policymanagement.RateLimitBinding{}, policymanagement.ErrNotFound
		}
		return policymanagement.RateLimitBinding{}, fmt.Errorf("lock quota partition: %w", err)
	}
	binding.QuotaPartitionID = partition
	var id string
	if err := tx.QueryRowContext(ctx, insertManagedRateBindingQuery, binding.ID, binding.NamespaceID,
		binding.PolicyID, binding.Subject.ID, binding.Mode, binding.QuotaPartitionID, binding.CreatedAt).Scan(&id); err != nil {
		return policymanagement.RateLimitBinding{}, mapManagedRateBindingWrite(err, "insert RateLimit binding")
	}
	created, err := scanManagedRateBinding(tx.QueryRowContext(ctx, getManagedRateBindingQuery, binding.NamespaceID, id))
	if err != nil {
		return policymanagement.RateLimitBinding{}, mapManagedPolicyRead(err, "read created RateLimit binding")
	}
	return created, nil
}

func insertAndReadManagedAccessBinding(
	ctx context.Context,
	tx *sql.Tx,
	binding policymanagement.AccessPolicyBinding,
) (policymanagement.AccessPolicyBinding, error) {
	var id string
	if err := tx.QueryRowContext(ctx, insertManagedAccessBindingQuery, binding.ID, binding.NamespaceID,
		binding.PolicyID, binding.Subject.ID, binding.CreatedAt).Scan(&id); err != nil {
		return policymanagement.AccessPolicyBinding{}, mapManagedPolicyCreate(err, "insert AccessPolicy binding")
	}
	created, err := scanManagedAccessBinding(tx.QueryRowContext(ctx, getManagedAccessBindingQuery, binding.NamespaceID, id))
	if err != nil {
		return policymanagement.AccessPolicyBinding{}, mapManagedPolicyRead(err, "read created AccessPolicy binding")
	}
	return created, nil
}

func lockManagedBindingReferences(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID, policyID string,
	subject policymanagement.Subject,
	rate bool,
) error {
	query := lockManagedActiveAccessPolicyQuery
	if rate {
		query = lockManagedActiveRatePolicyQuery
	}
	var id string
	if err := tx.QueryRowContext(ctx, query, namespaceID, policyID).Scan(&id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return policymanagement.ErrNotFound
		}
		return fmt.Errorf("lock active policy: %w", err)
	}
	return lockManagedSubject(ctx, tx, namespaceID, subject)
}

func lockManagedSubject(ctx context.Context, tx *sql.Tx, namespaceID string, subject policymanagement.Subject) error {
	query := lockManagedUserSubjectQuery
	switch subject.Type {
	case accesscontrol.SubjectKindTeam:
		query = lockManagedTeamSubjectQuery
	case accesscontrol.SubjectKindAPIKey:
		query = lockManagedKeySubjectQuery
	case accesscontrol.SubjectKindUser:
	default:
		return policymanagement.ErrInvalidRequest
	}
	var id string
	if err := tx.QueryRowContext(ctx, query, namespaceID, subject.ID).Scan(&id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return policymanagement.ErrNotFound
		}
		return fmt.Errorf("lock policy subject: %w", err)
	}
	return nil
}

func managedRateBindingFenced(ctx context.Context, tx *sql.Tx, namespaceID, bindingID string) (bool, error) {
	var fenced bool
	if err := tx.QueryRowContext(ctx, managedBindingFenceQuery, namespaceID, bindingID).Scan(&fenced); err != nil {
		return false, fmt.Errorf("check RateLimit binding usage fence: %w", err)
	}
	return fenced, nil
}

func scanManagedAccessBinding(scanner rowScanner) (policymanagement.AccessPolicyBinding, error) {
	var binding policymanagement.AccessPolicyBinding
	if err := scanner.Scan(&binding.ID, &binding.NamespaceID, &binding.PolicyID, &binding.Subject.ID,
		&binding.Subject.Type, &binding.Status, &binding.Revision, &binding.CreatedAt, &binding.UpdatedAt); err != nil {
		return policymanagement.AccessPolicyBinding{}, err
	}
	binding.CreatedAt, binding.UpdatedAt = binding.CreatedAt.UTC(), binding.UpdatedAt.UTC()
	if validateManagedAccessBinding(binding) != nil {
		return policymanagement.AccessPolicyBinding{}, errors.New("stored AccessPolicy binding violates its domain contract")
	}
	return binding, nil
}

func scanManagedRateBinding(scanner rowScanner) (policymanagement.RateLimitBinding, error) {
	var binding policymanagement.RateLimitBinding
	if err := scanner.Scan(&binding.ID, &binding.NamespaceID, &binding.PolicyID, &binding.Subject.ID,
		&binding.Subject.Type, &binding.Mode, &binding.QuotaPartitionID, &binding.Status,
		&binding.Revision, &binding.CreatedAt, &binding.UpdatedAt); err != nil {
		return policymanagement.RateLimitBinding{}, err
	}
	binding.CreatedAt, binding.UpdatedAt = binding.CreatedAt.UTC(), binding.UpdatedAt.UTC()
	if validateManagedRateBinding(binding) != nil {
		return policymanagement.RateLimitBinding{}, errors.New("stored RateLimit binding violates its domain contract")
	}
	return binding, nil
}

func validateNewManagedAccessBinding(binding policymanagement.AccessPolicyBinding) error {
	if validateManagedAccessBinding(binding) != nil || binding.Revision != 1 ||
		binding.Status != accesscontrol.BindingStatusActive || binding.CreatedAt.IsZero() ||
		!binding.CreatedAt.Equal(binding.UpdatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedAccessBinding(binding policymanagement.AccessPolicyBinding) error {
	domain := accesscontrol.AccessPolicyBinding{
		ID:          accesscontrol.PolicyBindingID(binding.ID),
		NamespaceID: accesscontrol.NamespaceID(binding.NamespaceID),
		Subject: accesscontrol.SubjectRef{
			NamespaceID: accesscontrol.NamespaceID(binding.NamespaceID),
			ID:          accesscontrol.SubjectID(binding.Subject.ID), Kind: binding.Subject.Type,
		},
		PolicyID: accesscontrol.AccessPolicyID(binding.PolicyID), Status: binding.Status,
		Revision: accesscontrol.Revision(binding.Revision),
	}
	if domain.Validate() != nil || validateManagedPolicyIDs(binding.NamespaceID, binding.ID) != nil ||
		validateUUID("AccessPolicy id", binding.PolicyID) != nil || validateUUID("policy subject id", binding.Subject.ID) != nil ||
		binding.CreatedAt.IsZero() || binding.UpdatedAt.Before(binding.CreatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateNewManagedRateBinding(binding policymanagement.RateLimitBinding) error {
	if binding.QuotaPartitionID == "" {
		binding.QuotaPartitionID = "pending"
	}
	if validateManagedRateBinding(binding) != nil || binding.Revision != 1 ||
		binding.Status != accesscontrol.BindingStatusActive || binding.CreatedAt.IsZero() ||
		!binding.CreatedAt.Equal(binding.UpdatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedRateBinding(binding policymanagement.RateLimitBinding) error {
	partition := binding.QuotaPartitionID
	if partition == "" {
		partition = "pending"
	}
	domain := accesscontrol.RateLimitBinding{
		ID:          accesscontrol.PolicyBindingID(binding.ID),
		NamespaceID: accesscontrol.NamespaceID(binding.NamespaceID),
		Subject: accesscontrol.SubjectRef{
			NamespaceID: accesscontrol.NamespaceID(binding.NamespaceID),
			ID:          accesscontrol.SubjectID(binding.Subject.ID), Kind: binding.Subject.Type,
		},
		PolicyID: accesscontrol.RateLimitPolicyID(binding.PolicyID), Mode: binding.Mode,
		QuotaPartitionID: accesscontrol.QuotaPartitionID(partition), Status: binding.Status,
		Revision: accesscontrol.Revision(binding.Revision),
	}
	if domain.Validate() != nil || validateManagedPolicyIDs(binding.NamespaceID, binding.ID) != nil ||
		validateUUID("RateLimitPolicy id", binding.PolicyID) != nil || validateUUID("policy subject id", binding.Subject.ID) != nil ||
		binding.CreatedAt.IsZero() || binding.UpdatedAt.Before(binding.CreatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedBindingQuery(query policymanagement.BindingQuery, rate bool) error {
	if validateUUID("namespace id", query.NamespaceID) != nil || query.Limit < 1 || query.Limit > 200 ||
		(query.PolicyID != "" && validateUUID("policy id", query.PolicyID) != nil) ||
		(query.Status != "" && !query.Status.Valid()) || (!rate && query.Mode != "") ||
		(rate && query.Mode != "" && !query.Mode.Valid()) {
		return policymanagement.ErrInvalidRequest
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return policymanagement.ErrInvalidRequest
	}
	if query.Subject != nil && (!query.Subject.Type.Valid() || validateUUID("subject id", query.Subject.ID) != nil) {
		return policymanagement.ErrInvalidRequest
	}
	if query.After != nil && (query.After.CreatedAt.IsZero() || validateUUID("cursor id", query.After.ID) != nil) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedBindingMutation(namespaceID, bindingID string, expected uint64,
	status accesscontrol.BindingStatus, actor policymanagement.Actor,
) error {
	if validateManagedPolicyMutation(namespaceID, bindingID, expected, actor) != nil || !status.Valid() {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func managedSubjectArgs(subject *policymanagement.Subject) (any, any) {
	if subject == nil {
		return "", nil
	}
	return subject.Type, subject.ID
}
