package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	insertManagedAccessPolicyQuery = `INSERT INTO access_policies
  (id,namespace_id,name,description,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,1,$6,$6)
RETURNING ` + managedAccessPolicyColumns
	lockManagedAccessPolicyQuery = `SELECT ` + managedAccessPolicyColumns + `
FROM access_policies WHERE namespace_id=$1 AND id=$2 FOR UPDATE`
	updateManagedAccessPolicyQuery = `UPDATE access_policies
SET name=$4, description=$5, status=$6, revision=revision+1,
    updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3
RETURNING ` + managedAccessPolicyColumns
	countManagedAccessBindingsQuery = `SELECT count(*) FROM access_policy_bindings
WHERE namespace_id=$1 AND policy_id=$2`
	deleteManagedAccessPolicyQuery = `DELETE FROM access_policies
WHERE namespace_id=$1 AND id=$2 AND revision=$3`

	insertManagedRatePolicyQuery = `INSERT INTO rate_limit_policies
  (id,namespace_id,name,description,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,1,$6,$6)
RETURNING ` + managedRatePolicyColumns
	lockManagedRatePolicyQuery = `SELECT ` + managedRatePolicyColumns + `
FROM rate_limit_policies WHERE namespace_id=$1 AND id=$2 FOR UPDATE`
	updateManagedRatePolicyQuery = `UPDATE rate_limit_policies
SET name=$4, description=$5, status=$6, revision=revision+1,
    updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3
RETURNING ` + managedRatePolicyColumns
	countManagedRateBindingsQuery = `SELECT count(*) FROM rate_limit_bindings
WHERE namespace_id=$1 AND policy_id=$2`
	deleteManagedRatePolicyQuery = `DELETE FROM rate_limit_policies
WHERE namespace_id=$1 AND id=$2 AND revision=$3`
	deleteManagedRateRulesQuery = `DELETE FROM rate_limit_rules WHERE policy_id=$1`

	lockManagedRoutingModelQuery = `SELECT id FROM routing_models
WHERE namespace_id=$1 AND id=$2 AND status <> 'deleted' AND deleted_at IS NULL FOR KEY SHARE`
	lockManagedRoutingEntrypointQuery = `SELECT id FROM routing_entrypoints
WHERE namespace_id=$1 AND id=$2 AND status <> 'deleted' AND deleted_at IS NULL FOR KEY SHARE`
	managedRatePolicyFenceQuery = `SELECT EXISTS (
  SELECT 1 FROM rate_limit_bindings b
  JOIN unknown_usage_fence_bindings fb
    ON fb.binding_id=b.id
  JOIN unknown_usage_fences f
    ON f.namespace_id=b.namespace_id AND f.id=fb.fence_id
  WHERE b.namespace_id=$1 AND b.policy_id=$2 AND f.state IN ('open','reconciling')
)`
)

func (s *Store) CreateManagedAccessPolicy(
	ctx context.Context,
	mutation policymanagement.CreateAccessPolicyMutation,
) (policymanagement.MutationResult, error) {
	if err := validateNewManagedAccessPolicy(mutation.Policy); err != nil {
		return policymanagement.MutationResult{}, err
	}
	meta, err := managedPolicyMutationMeta(mutation.Actor, "access_policy.create", "Create AccessPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		if replay, ok, err := lockManagedPolicyCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		if err := lockManagedGrantResources(ctx, tx, mutation.Policy.NamespaceID, mutation.Policy.Grants); err != nil {
			return policymanagement.MutationResult{}, err
		}
		policy := mutation.Policy
		created, err := scanManagedAccessPolicy(tx.QueryRowContext(ctx, insertManagedAccessPolicyQuery,
			policy.ID, policy.NamespaceID, policy.Name, policy.Description, policy.Status, policy.CreatedAt))
		if err != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyCreate(err, "insert AccessPolicy")
		}
		if err := replaceAccessPolicyGrants(ctx, tx, accesscontrol.AccessPolicyID(created.ID),
			managedAccessPolicyDomain(policy).Grants, false); err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, created.NamespaceID, "access_policy",
			created.ID, created.Revision, outboxCreated, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return completeManagedPolicyCommand(ctx, tx, mutation.Command, "access_policy", created.ID, created.Revision, 201)
	})
}

func (s *Store) UpdateManagedAccessPolicy(
	ctx context.Context,
	policy policymanagement.AccessPolicy,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if err := validateManagedAccessPolicy(policy); err != nil || expected == 0 || policy.Revision != expected {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "access_policy.update", "Update AccessPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, scanManagedAccessPolicyErr := scanManagedAccessPolicy(tx.QueryRowContext(ctx, lockManagedAccessPolicyQuery,
			policy.NamespaceID, policy.ID))
		if scanManagedAccessPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(scanManagedAccessPolicyErr, "lock AccessPolicy")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if err := lockManagedGrantResources(ctx, tx, policy.NamespaceID, policy.Grants); err != nil {
			return policymanagement.MutationResult{}, err
		}
		updated, scanManagedAccessPolicyErr := scanManagedAccessPolicy(tx.QueryRowContext(ctx, updateManagedAccessPolicyQuery,
			policy.NamespaceID, policy.ID, expected, policy.Name, policy.Description, policy.Status))
		if scanManagedAccessPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyCAS(scanManagedAccessPolicyErr, "update AccessPolicy")
		}
		if err := replaceAccessPolicyGrants(ctx, tx, accesscontrol.AccessPolicyID(updated.ID),
			managedAccessPolicyDomain(policy).Grants, true); err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, updated.NamespaceID, "access_policy",
			updated.ID, updated.Revision, outboxUpdated, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "access_policy", ID: updated.ID,
			Revision: updated.Revision, HTTPStatus: 200,
		}, nil
	})
}

func (s *Store) DeleteManagedAccessPolicy(
	ctx context.Context,
	namespaceID, policyID string,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedPolicyMutation(namespaceID, policyID, expected, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "access_policy.delete", "Delete AccessPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, deleteManagedAccessPolicyErr := scanManagedAccessPolicy(tx.QueryRowContext(ctx, lockManagedAccessPolicyQuery, namespaceID, policyID))
		if deleteManagedAccessPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(deleteManagedAccessPolicyErr, "lock AccessPolicy")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if err := requireManagedPolicyUnbound(ctx, tx, countManagedAccessBindingsQuery, namespaceID, policyID); err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, deleteAccessPolicyGrantsQuery, policyID); err != nil {
			return policymanagement.MutationResult{}, fmt.Errorf("delete AccessPolicy grants: %w", err)
		}
		result, deleteManagedAccessPolicyErr := tx.ExecContext(ctx, deleteManagedAccessPolicyQuery, namespaceID, policyID, expected)
		if deleteManagedAccessPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyDelete(deleteManagedAccessPolicyErr, "delete AccessPolicy")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "access_policy", policyID,
			revision, outboxDeleted, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "access_policy", ID: policyID,
			Revision: revision, HTTPStatus: 204,
		}, nil
	})
}

func (s *Store) CreateManagedRateLimitPolicy(
	ctx context.Context,
	mutation policymanagement.CreateRateLimitPolicyMutation,
) (policymanagement.MutationResult, error) {
	if err := validateNewManagedRatePolicy(mutation.Policy); err != nil {
		return policymanagement.MutationResult{}, err
	}
	meta, err := managedPolicyMutationMeta(mutation.Actor, "rate_limit_policy.create", "Create RateLimitPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		if replay, ok, err := lockManagedPolicyCommand(ctx, tx, mutation.Command); err != nil || ok {
			return replay, err
		}
		created, err := insertManagedRatePolicy(ctx, tx, mutation.Policy)
		if err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, created.NamespaceID, "rate_limit_policy",
			created.ID, created.Revision, outboxCreated, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return completeManagedPolicyCommand(ctx, tx, mutation.Command, "rate_limit_policy", created.ID, created.Revision, 201)
	})
}

func (s *Store) UpdateManagedRateLimitPolicy(
	ctx context.Context,
	policy policymanagement.RateLimitPolicy,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if err := validateManagedRatePolicy(policy); err != nil || expected == 0 || policy.Revision != expected {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "rate_limit_policy.update", "Update RateLimitPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, updateManagedRateLimitPolicyErr := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, lockManagedRatePolicyQuery,
			policy.NamespaceID, policy.ID))
		if updateManagedRateLimitPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(updateManagedRateLimitPolicyErr, "lock RateLimitPolicy")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		currentRules, updateManagedRateLimitPolicyErr := listRateLimitRules(ctx, tx, accesscontrol.RateLimitPolicyID(policy.ID))
		if updateManagedRateLimitPolicyErr != nil {
			return policymanagement.MutationResult{}, updateManagedRateLimitPolicyErr
		}
		desiredRules := managedRatePolicyDomain(policy).Rules
		if ratePolicyRuntimeChanged(current, policy, currentRules, desiredRules) {
			fenced, err := managedRatePolicyFenced(ctx, tx, policy.NamespaceID, policy.ID)
			if err != nil {
				return policymanagement.MutationResult{}, err
			}
			if fenced && !ratePolicyFenceCompatible(current, policy, currentRules, desiredRules) {
				return policymanagement.MutationResult{}, policymanagement.ErrUnknownUsageFence
			}
		}
		if err := validateRetainedRuleSemantics(managedRulesByID(currentRules), desiredRules); err != nil {
			return policymanagement.MutationResult{}, policymanagement.ErrCounterSemantics
		}
		updated, updateManagedRateLimitPolicyErr := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, updateManagedRatePolicyQuery,
			policy.NamespaceID, policy.ID, expected, policy.Name, policy.Description, policy.Status))
		if updateManagedRateLimitPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyCAS(updateManagedRateLimitPolicyErr, "update RateLimitPolicy")
		}
		if err := syncRateLimitRules(ctx, tx, accesscontrol.RateLimitPolicyID(updated.ID), desiredRules); err != nil {
			if errors.Is(err, ErrRevisionConflict) {
				return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
			}
			return policymanagement.MutationResult{}, err
		}
		if _, err := appendManagedPolicyMutation(ctx, tx, updated.NamespaceID, "rate_limit_policy",
			updated.ID, updated.Revision, outboxUpdated, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "rate_limit_policy", ID: updated.ID,
			Revision: updated.Revision, HTTPStatus: 200,
		}, nil
	})
}

func (s *Store) DeleteManagedRateLimitPolicy(
	ctx context.Context,
	namespaceID, policyID string,
	expected uint64,
	actor policymanagement.Actor,
) (policymanagement.MutationResult, error) {
	if validateManagedPolicyMutation(namespaceID, policyID, expected, actor) != nil {
		return policymanagement.MutationResult{}, policymanagement.ErrInvalidRequest
	}
	meta, err := managedPolicyMutationMeta(actor, "rate_limit_policy.delete", "Delete RateLimitPolicy.", nil)
	if err != nil {
		return policymanagement.MutationResult{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.MutationResult, error) {
		current, deleteManagedRateLimitPolicyErr := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, lockManagedRatePolicyQuery, namespaceID, policyID))
		if deleteManagedRateLimitPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyRead(deleteManagedRateLimitPolicyErr, "lock RateLimitPolicy")
		}
		if current.Revision != expected {
			return policymanagement.MutationResult{}, policymanagement.ErrRevisionConflict
		}
		if err := requireManagedPolicyUnbound(ctx, tx, countManagedRateBindingsQuery, namespaceID, policyID); err != nil {
			return policymanagement.MutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, deleteManagedRateRulesQuery, policyID); err != nil {
			return policymanagement.MutationResult{}, fmt.Errorf("delete RateLimitPolicy rules: %w", err)
		}
		result, deleteManagedRateLimitPolicyErr := tx.ExecContext(ctx, deleteManagedRatePolicyQuery, namespaceID, policyID, expected)
		if deleteManagedRateLimitPolicyErr != nil {
			return policymanagement.MutationResult{}, mapManagedPolicyDelete(deleteManagedRateLimitPolicyErr, "delete RateLimitPolicy")
		}
		if err := requireOneRow(result, policymanagement.ErrRevisionConflict); err != nil {
			return policymanagement.MutationResult{}, err
		}
		revision := expected + 1
		if _, err := appendManagedPolicyMutation(ctx, tx, namespaceID, "rate_limit_policy", policyID,
			revision, outboxDeleted, meta, nil); err != nil {
			return policymanagement.MutationResult{}, err
		}
		return policymanagement.MutationResult{
			Kind: "rate_limit_policy", ID: policyID,
			Revision: revision, HTTPStatus: 204,
		}, nil
	})
}

func insertManagedRatePolicy(
	ctx context.Context,
	tx *sql.Tx,
	policy policymanagement.RateLimitPolicy,
) (policymanagement.RateLimitPolicy, error) {
	created, err := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, insertManagedRatePolicyQuery,
		policy.ID, policy.NamespaceID, policy.Name, policy.Description, policy.Status, policy.CreatedAt))
	if err != nil {
		return policymanagement.RateLimitPolicy{}, mapManagedPolicyCreate(err, "insert RateLimitPolicy")
	}
	if err := insertRateLimitRules(ctx, tx, accesscontrol.RateLimitPolicyID(created.ID),
		managedRatePolicyDomain(policy).Rules); err != nil {
		return policymanagement.RateLimitPolicy{}, err
	}
	created.Rules = policy.Rules
	return created, nil
}

func lockManagedGrantResources(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	grants []policymanagement.AccessGrant,
) error {
	seen := make(map[string]struct{}, len(grants))
	for _, grant := range grants {
		key := string(grant.ResourceType) + "\x00" + grant.ResourceID
		if _, found := seen[key]; found {
			continue
		}
		seen[key] = struct{}{}
		query := lockManagedRoutingModelQuery
		if grant.ResourceType == accesscontrol.GrantResourceEntrypoint {
			query = lockManagedRoutingEntrypointQuery
		}
		var id string
		if err := tx.QueryRowContext(ctx, query, namespaceID, grant.ResourceID).Scan(&id); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return policymanagement.ErrNotFound
			}
			return fmt.Errorf("lock AccessPolicy grant resource: %w", err)
		}
	}
	return nil
}

func requireManagedPolicyUnbound(
	ctx context.Context,
	tx *sql.Tx,
	query, namespaceID, policyID string,
) error {
	var count int64
	if err := tx.QueryRowContext(ctx, query, namespaceID, policyID).Scan(&count); err != nil {
		return fmt.Errorf("count policy bindings: %w", err)
	}
	if count != 0 {
		return policymanagement.ErrResourceInUse
	}
	return nil
}

func managedRatePolicyFenced(ctx context.Context, tx *sql.Tx, namespaceID, policyID string) (bool, error) {
	var fenced bool
	if err := tx.QueryRowContext(ctx, managedRatePolicyFenceQuery, namespaceID, policyID).Scan(&fenced); err != nil {
		return false, fmt.Errorf("check RateLimitPolicy usage fences: %w", err)
	}
	return fenced, nil
}

func ratePolicyRuntimeChanged(
	current policymanagement.RateLimitPolicy,
	desired policymanagement.RateLimitPolicy,
	currentRules, desiredRules []accesscontrol.RateLimitRule,
) bool {
	return current.Status != desired.Status || !reflect.DeepEqual(currentRules, desiredRules)
}

func ratePolicyFenceCompatible(
	current policymanagement.RateLimitPolicy,
	desired policymanagement.RateLimitPolicy,
	currentRules, desiredRules []accesscontrol.RateLimitRule,
) bool {
	if current.Status != desired.Status || len(currentRules) != len(desiredRules) {
		return false
	}
	currentByID := managedRulesByID(currentRules)
	for _, desiredRule := range desiredRules {
		currentRule, found := currentByID[desiredRule.ID]
		if !found {
			return false
		}
		currentRule.Limit = ""
		desiredRule.Limit = ""
		if !reflect.DeepEqual(currentRule, desiredRule) {
			return false
		}
	}
	return true
}

func managedRulesByID(rules []accesscontrol.RateLimitRule) map[accesscontrol.RateLimitRuleID]accesscontrol.RateLimitRule {
	result := make(map[accesscontrol.RateLimitRuleID]accesscontrol.RateLimitRule, len(rules))
	for _, rule := range rules {
		result[rule.ID] = rule
	}
	return result
}

func validateNewManagedAccessPolicy(policy policymanagement.AccessPolicy) error {
	if validateManagedAccessPolicy(policy) != nil || policy.Revision != 1 ||
		policy.CreatedAt.IsZero() || !policy.CreatedAt.Equal(policy.UpdatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedAccessPolicy(policy policymanagement.AccessPolicy) error {
	if validateManagedPolicyIDs(policy.NamespaceID, policy.ID) != nil || len(policy.Grants) > 512 ||
		policy.Description != stringsTrimmed(policy.Description) || len(policy.Description) > 1000 ||
		managedAccessPolicyDomain(policy).Validate() != nil {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateNewManagedRatePolicy(policy policymanagement.RateLimitPolicy) error {
	if validateManagedRatePolicy(policy) != nil || policy.Revision != 1 ||
		policy.CreatedAt.IsZero() || !policy.CreatedAt.Equal(policy.UpdatedAt) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedRatePolicy(policy policymanagement.RateLimitPolicy) error {
	if validateManagedPolicyIDs(policy.NamespaceID, policy.ID) != nil || len(policy.Rules) > 128 ||
		policy.Description != stringsTrimmed(policy.Description) || len(policy.Description) > 1000 ||
		managedRatePolicyDomain(policy).Validate() != nil {
		return policymanagement.ErrInvalidRequest
	}
	for _, rule := range policy.Rules {
		if validateUUID("RateLimitRule id", rule.ID) != nil {
			return policymanagement.ErrInvalidRequest
		}
		if _, err := encodeRateLimitRule(managedRateRuleDomain(policy.ID, rule)); err != nil {
			return policymanagement.ErrInvalidRequest
		}
	}
	return nil
}

func validateManagedPolicyMutation(namespaceID, policyID string, expected uint64, actor policymanagement.Actor) error {
	if validateManagedPolicyIDs(namespaceID, policyID) != nil || expected == 0 || !canonicalManagedPolicyActor(actor) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedPolicyIDs(namespaceID, resourceID string) error {
	if validateUUID("namespace id", namespaceID) != nil || validateUUID("policy resource id", resourceID) != nil {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}

func validateManagedPolicyQuery(query policymanagement.PolicyQuery) error {
	if validateUUID("namespace id", query.NamespaceID) != nil || query.Limit < 1 || query.Limit > 200 ||
		(query.Status != "" && !query.Status.Valid()) {
		return policymanagement.ErrInvalidRequest
	}
	normalizedSearch, err := managementsearch.Normalize(query.Search)
	if err != nil || normalizedSearch != query.Search {
		return policymanagement.ErrInvalidRequest
	}
	if _, err := query.Scope.Digest(); err != nil || query.Scope.NamespaceID != accesscontrol.NamespaceID(query.NamespaceID) {
		return policymanagement.ErrInvalidRequest
	}
	if query.After != nil && (query.After.CreatedAt.IsZero() || validateUUID("cursor id", query.After.ID) != nil) {
		return policymanagement.ErrInvalidRequest
	}
	return nil
}
