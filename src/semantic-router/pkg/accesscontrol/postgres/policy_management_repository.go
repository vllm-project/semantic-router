package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	managedAccessPolicyColumns = `id, namespace_id, name, description, status,
       revision, created_at, updated_at`
	managedRatePolicyColumns = `id, namespace_id, name, description, status,
       revision, created_at, updated_at`

	getManagedAccessPolicyQuery = `SELECT ` + managedAccessPolicyColumns + `
FROM access_policies WHERE namespace_id=$1 AND id=$2`
	listManagedAccessPoliciesQuery = `SELECT ` + managedAccessPolicyColumns + `
FROM access_policies p
WHERE p.namespace_id=$1 AND ($2='' OR p.status=$2)
  AND ($3 OR p.id = ANY($4::uuid[]))
  AND ($5::timestamptz IS NULL OR p.created_at < $5 OR (p.created_at=$5 AND p.id > $6::uuid))
ORDER BY p.created_at DESC, p.id ASC LIMIT $7`
	searchManagedAccessPoliciesQuery = `SELECT ` + managedAccessPolicyColumns + `
FROM access_policies p
WHERE p.namespace_id=$1 AND ($2='' OR p.status=$2)
  AND ($3 OR p.id = ANY($4::uuid[]))
  AND (lower(p.name) LIKE $5 ESCAPE E'\\' OR p.id::text LIKE $5 ESCAPE E'\\')
  AND ($6::timestamptz IS NULL OR p.created_at < $6 OR (p.created_at=$6 AND p.id > $7::uuid))
ORDER BY p.created_at DESC, p.id ASC LIMIT $8`
	listManagedAccessGrantsQuery = `SELECT policy_id, resource_type, resource_id, permission, effect
FROM access_policy_grants WHERE policy_id = ANY($1::uuid[])
ORDER BY policy_id, resource_type, resource_id, permission, effect`

	getManagedRatePolicyQuery = `SELECT ` + managedRatePolicyColumns + `
FROM rate_limit_policies WHERE namespace_id=$1 AND id=$2`
	listManagedRatePoliciesQuery = `SELECT ` + managedRatePolicyColumns + `
FROM rate_limit_policies p
WHERE p.namespace_id=$1 AND ($2='' OR p.status=$2)
  AND ($3 OR p.id = ANY($4::uuid[]))
  AND ($5::timestamptz IS NULL OR p.created_at < $5 OR (p.created_at=$5 AND p.id > $6::uuid))
ORDER BY p.created_at DESC, p.id ASC LIMIT $7`
	searchManagedRatePoliciesQuery = `SELECT ` + managedRatePolicyColumns + `
FROM rate_limit_policies p
WHERE p.namespace_id=$1 AND ($2='' OR p.status=$2)
  AND ($3 OR p.id = ANY($4::uuid[]))
  AND (lower(p.name) LIKE $5 ESCAPE E'\\' OR p.id::text LIKE $5 ESCAPE E'\\')
  AND ($6::timestamptz IS NULL OR p.created_at < $6 OR (p.created_at=$6 AND p.id > $7::uuid))
ORDER BY p.created_at DESC, p.id ASC LIMIT $8`
	listManagedRateRulesQuery = `SELECT ` + rateLimitRuleColumns + `
FROM rate_limit_rules WHERE policy_id = ANY($1::uuid[])
ORDER BY policy_id, ordinal, id`
)

func (s *Store) ReadyPolicyManagement(ctx context.Context, codec *managementcommand.Codec) error {
	if s == nil || s.db == nil || codec == nil {
		return policymanagement.ErrUnavailable
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, s.db, codec)
}

func (s *Store) ReplayPolicyCommand(
	ctx context.Context,
	command managementcommand.Command,
) (policymanagement.MutationResult, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, s.db, command)
	if err != nil || !found {
		return policymanagement.MutationResult{}, false, err
	}
	result, err := policyMutationResult(stored)
	return result, true, err
}

func (s *Store) GetManagedAccessPolicy(
	ctx context.Context,
	namespaceID, policyID string,
) (policymanagement.AccessPolicy, error) {
	if validateManagedPolicyIDs(namespaceID, policyID) != nil {
		return policymanagement.AccessPolicy{}, policymanagement.ErrInvalidRequest
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.AccessPolicy, error) {
		policy, err := scanManagedAccessPolicy(tx.QueryRowContext(ctx, getManagedAccessPolicyQuery, namespaceID, policyID))
		if err != nil {
			return policymanagement.AccessPolicy{}, mapManagedPolicyRead(err, "get AccessPolicy")
		}
		grants, err := loadManagedAccessGrants(ctx, tx, []string{policy.ID})
		if err != nil {
			return policymanagement.AccessPolicy{}, err
		}
		policy.Grants = grants[policy.ID]
		return policy, validateStoredManagedAccessPolicy(policy)
	})
}

func (s *Store) ListManagedAccessPolicies(
	ctx context.Context,
	query policymanagement.PolicyQuery,
) (policymanagement.RepositoryPage[policymanagement.AccessPolicy], error) {
	if validateManagedPolicyQuery(query) != nil {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, policymanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)) == 0 {
		return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{Items: []policymanagement.AccessPolicy{}}, nil
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.RepositoryPage[policymanagement.AccessPolicy], error) {
		afterTime, afterID := managedPolicyCursorArgs(query.After)
		var rows *sql.Rows
		var listManagedAccessPoliciesErr error
		if query.Search == "" {
			rows, listManagedAccessPoliciesErr = tx.QueryContext(ctx, listManagedAccessPoliciesQuery, query.NamespaceID,
				query.Status, query.Scope.All, pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)),
				afterTime, afterID, query.Limit+1)
		} else {
			rows, listManagedAccessPoliciesErr = tx.QueryContext(ctx, searchManagedAccessPoliciesQuery, query.NamespaceID,
				query.Status, query.Scope.All, pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)),
				managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit+1)
		}
		if listManagedAccessPoliciesErr != nil {
			return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, fmt.Errorf("list AccessPolicies: %w", listManagedAccessPoliciesErr)
		}
		items := make([]policymanagement.AccessPolicy, 0, query.Limit+1)
		for rows.Next() {
			item, scanErr := scanManagedAccessPolicy(rows)
			if scanErr != nil {
				_ = rows.Close()
				return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, fmt.Errorf("scan AccessPolicy page: %w", scanErr)
			}
			items = append(items, item)
		}
		if err := rows.Close(); err != nil {
			return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, fmt.Errorf("close AccessPolicy page: %w", err)
		}
		if err := rows.Err(); err != nil {
			return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, fmt.Errorf("read AccessPolicy page: %w", err)
		}
		page := trimManagedPage(items, query.Limit)
		ids := managedAccessPolicyIDs(page.Items)
		grants, listManagedAccessPoliciesErr := loadManagedAccessGrants(ctx, tx, ids)
		if listManagedAccessPoliciesErr != nil {
			return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, listManagedAccessPoliciesErr
		}
		for index := range page.Items {
			page.Items[index].Grants = grants[page.Items[index].ID]
			if err := validateStoredManagedAccessPolicy(page.Items[index]); err != nil {
				return policymanagement.RepositoryPage[policymanagement.AccessPolicy]{}, err
			}
		}
		return page, nil
	})
}

func (s *Store) GetManagedRateLimitPolicy(
	ctx context.Context,
	namespaceID, policyID string,
) (policymanagement.RateLimitPolicy, error) {
	if validateManagedPolicyIDs(namespaceID, policyID) != nil {
		return policymanagement.RateLimitPolicy{}, policymanagement.ErrInvalidRequest
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.RateLimitPolicy, error) {
		policy, err := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx, getManagedRatePolicyQuery, namespaceID, policyID))
		if err != nil {
			return policymanagement.RateLimitPolicy{}, mapManagedPolicyRead(err, "get RateLimitPolicy")
		}
		rules, err := loadManagedRateRules(ctx, tx, []string{policy.ID})
		if err != nil {
			return policymanagement.RateLimitPolicy{}, err
		}
		policy.Rules = rules[policy.ID]
		return policy, validateStoredManagedRatePolicy(policy)
	})
}

func (s *Store) ListManagedRateLimitPolicies(
	ctx context.Context,
	query policymanagement.PolicyQuery,
) (policymanagement.RepositoryPage[policymanagement.RateLimitPolicy], error) {
	if validateManagedPolicyQuery(query) != nil {
		return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, policymanagement.ErrInvalidRequest
	}
	if !query.Scope.All && len(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)) == 0 {
		return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{Items: []policymanagement.RateLimitPolicy{}}, nil
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (policymanagement.RepositoryPage[policymanagement.RateLimitPolicy], error) {
		afterTime, afterID := managedPolicyCursorArgs(query.After)
		var rows *sql.Rows
		var listManagedRateLimitPoliciesErr error
		if query.Search == "" {
			rows, listManagedRateLimitPoliciesErr = tx.QueryContext(ctx, listManagedRatePoliciesQuery, query.NamespaceID,
				query.Status, query.Scope.All, pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)),
				afterTime, afterID, query.Limit+1)
		} else {
			rows, listManagedRateLimitPoliciesErr = tx.QueryContext(ctx, searchManagedRatePoliciesQuery, query.NamespaceID,
				query.Status, query.Scope.All, pq.Array(query.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)),
				managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit+1)
		}
		if listManagedRateLimitPoliciesErr != nil {
			return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, fmt.Errorf("list RateLimitPolicies: %w", listManagedRateLimitPoliciesErr)
		}
		items := make([]policymanagement.RateLimitPolicy, 0, query.Limit+1)
		for rows.Next() {
			item, scanErr := scanManagedRateLimitPolicy(rows)
			if scanErr != nil {
				_ = rows.Close()
				return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, fmt.Errorf("scan RateLimitPolicy page: %w", scanErr)
			}
			items = append(items, item)
		}
		if err := rows.Close(); err != nil {
			return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, fmt.Errorf("close RateLimitPolicy page: %w", err)
		}
		if err := rows.Err(); err != nil {
			return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, fmt.Errorf("read RateLimitPolicy page: %w", err)
		}
		page := trimManagedPage(items, query.Limit)
		ids := managedRatePolicyIDs(page.Items)
		rules, listManagedRateLimitPoliciesErr := loadManagedRateRules(ctx, tx, ids)
		if listManagedRateLimitPoliciesErr != nil {
			return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, listManagedRateLimitPoliciesErr
		}
		for index := range page.Items {
			page.Items[index].Rules = rules[page.Items[index].ID]
			if err := validateStoredManagedRatePolicy(page.Items[index]); err != nil {
				return policymanagement.RepositoryPage[policymanagement.RateLimitPolicy]{}, err
			}
		}
		return page, nil
	})
}

func loadManagedAccessGrants(
	ctx context.Context,
	tx *sql.Tx,
	policyIDs []string,
) (map[string][]policymanagement.AccessGrant, error) {
	result := make(map[string][]policymanagement.AccessGrant, len(policyIDs))
	if len(policyIDs) == 0 {
		return result, nil
	}
	rows, err := tx.QueryContext(ctx, listManagedAccessGrantsQuery, pq.Array(policyIDs))
	if err != nil {
		return nil, fmt.Errorf("list AccessPolicy grants: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var policyID string
		var grant policymanagement.AccessGrant
		if err := rows.Scan(&policyID, &grant.ResourceType, &grant.ResourceID, &grant.Permission, &grant.Effect); err != nil {
			return nil, fmt.Errorf("scan AccessPolicy grant: %w", err)
		}
		result[policyID] = append(result[policyID], grant)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read AccessPolicy grants: %w", err)
	}
	return result, nil
}

func loadManagedRateRules(
	ctx context.Context,
	tx *sql.Tx,
	policyIDs []string,
) (map[string][]policymanagement.RateLimitRule, error) {
	result := make(map[string][]policymanagement.RateLimitRule, len(policyIDs))
	if len(policyIDs) == 0 {
		return result, nil
	}
	rows, err := tx.QueryContext(ctx, listManagedRateRulesQuery, pq.Array(policyIDs))
	if err != nil {
		return nil, fmt.Errorf("list RateLimitPolicy rules: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		domain, err := scanRateLimitRule(rows)
		if err != nil {
			return nil, fmt.Errorf("scan RateLimitPolicy rule: %w", err)
		}
		policyID := string(domain.PolicyID)
		result[policyID] = append(result[policyID], managedRateRule(domain))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read RateLimitPolicy rules: %w", err)
	}
	return result, nil
}

func scanManagedAccessPolicy(scanner rowScanner) (policymanagement.AccessPolicy, error) {
	var policy policymanagement.AccessPolicy
	if err := scanner.Scan(&policy.ID, &policy.NamespaceID, &policy.Name, &policy.Description,
		&policy.Status, &policy.Revision, &policy.CreatedAt, &policy.UpdatedAt); err != nil {
		return policymanagement.AccessPolicy{}, err
	}
	policy.CreatedAt, policy.UpdatedAt = policy.CreatedAt.UTC(), policy.UpdatedAt.UTC()
	return policy, nil
}

func scanManagedRateLimitPolicy(scanner rowScanner) (policymanagement.RateLimitPolicy, error) {
	var policy policymanagement.RateLimitPolicy
	if err := scanner.Scan(&policy.ID, &policy.NamespaceID, &policy.Name, &policy.Description,
		&policy.Status, &policy.Revision, &policy.CreatedAt, &policy.UpdatedAt); err != nil {
		return policymanagement.RateLimitPolicy{}, err
	}
	policy.CreatedAt, policy.UpdatedAt = policy.CreatedAt.UTC(), policy.UpdatedAt.UTC()
	return policy, nil
}

func managedRateRule(rule accesscontrol.RateLimitRule) policymanagement.RateLimitRule {
	return policymanagement.RateLimitRule{
		ID: string(rule.ID), Metric: rule.Metric,
		Algorithm: rule.Algorithm, Limit: rule.Limit, Window: policymanagement.ISODuration(rule.Window),
		CalendarPeriod: rule.CalendarPeriod, Timezone: rule.Timezone,
		BucketCapacity: rule.BucketCapacity, RefillAmount: rule.RefillAmount,
		RefillPeriod:         policymanagement.ISODuration(rule.RefillPeriod),
		GCRAEmissionInterval: policymanagement.ISODuration(rule.GCRAEmissionInterval),
		GCRABurstTolerance:   cloneManagedInt64(rule.GCRABurstTolerance), Accounting: rule.Accounting,
		Enforcement: rule.Enforcement, Ordinal: rule.Ordinal,
	}
}

func managedRateRuleDomain(policyID string, rule policymanagement.RateLimitRule) accesscontrol.RateLimitRule {
	return accesscontrol.RateLimitRule{
		ID:       accesscontrol.RateLimitRuleID(rule.ID),
		PolicyID: accesscontrol.RateLimitPolicyID(policyID), Metric: rule.Metric,
		Algorithm: rule.Algorithm, Limit: rule.Limit, Window: rule.Window.Duration(),
		CalendarPeriod: rule.CalendarPeriod, Timezone: rule.Timezone,
		BucketCapacity: rule.BucketCapacity, RefillAmount: rule.RefillAmount,
		RefillPeriod: rule.RefillPeriod.Duration(), GCRAEmissionInterval: rule.GCRAEmissionInterval.Duration(),
		GCRABurstTolerance: cloneManagedInt64(rule.GCRABurstTolerance), Accounting: rule.Accounting,
		Enforcement: rule.Enforcement, Ordinal: rule.Ordinal,
	}
}

func managedAccessPolicyDomain(policy policymanagement.AccessPolicy) accesscontrol.AccessPolicy {
	grants := make([]accesscontrol.AccessPolicyGrant, len(policy.Grants))
	for index, grant := range policy.Grants {
		grants[index] = accesscontrol.AccessPolicyGrant{
			PolicyID:   accesscontrol.AccessPolicyID(policy.ID),
			Resource:   accesscontrol.GrantResource{Type: grant.ResourceType, ID: accesscontrol.ResourceID(grant.ResourceID)},
			Permission: grant.Permission, Effect: grant.Effect,
		}
	}
	return accesscontrol.AccessPolicy{
		ID:          accesscontrol.AccessPolicyID(policy.ID),
		NamespaceID: accesscontrol.NamespaceID(policy.NamespaceID), DisplayName: policy.Name,
		Status: policy.Status, Revision: accesscontrol.Revision(policy.Revision), Grants: grants,
		CreatedAt: policy.CreatedAt, UpdatedAt: policy.UpdatedAt,
	}
}

func managedRatePolicyDomain(policy policymanagement.RateLimitPolicy) accesscontrol.RateLimitPolicy {
	rules := make([]accesscontrol.RateLimitRule, len(policy.Rules))
	for index, rule := range policy.Rules {
		rules[index] = managedRateRuleDomain(policy.ID, rule)
	}
	return accesscontrol.RateLimitPolicy{
		ID:          accesscontrol.RateLimitPolicyID(policy.ID),
		NamespaceID: accesscontrol.NamespaceID(policy.NamespaceID), DisplayName: policy.Name,
		Status: policy.Status, Revision: accesscontrol.Revision(policy.Revision), Rules: rules,
		CreatedAt: policy.CreatedAt, UpdatedAt: policy.UpdatedAt,
	}
}

func validateStoredManagedAccessPolicy(policy policymanagement.AccessPolicy) error {
	if managedAccessPolicyDomain(policy).Validate() != nil || policy.Description != stringsTrimmed(policy.Description) {
		return errors.New("stored AccessPolicy violates its domain contract")
	}
	return nil
}

func validateStoredManagedRatePolicy(policy policymanagement.RateLimitPolicy) error {
	if managedRatePolicyDomain(policy).Validate() != nil || policy.Description != stringsTrimmed(policy.Description) {
		return errors.New("stored RateLimitPolicy violates its domain contract")
	}
	return nil
}

func managedAccessPolicyIDs(items []policymanagement.AccessPolicy) []string {
	ids := make([]string, len(items))
	for index := range items {
		ids[index] = items[index].ID
	}
	return ids
}

func managedRatePolicyIDs(items []policymanagement.RateLimitPolicy) []string {
	ids := make([]string, len(items))
	for index := range items {
		ids[index] = items[index].ID
	}
	return ids
}

func managedPolicyCursorArgs(cursor *policymanagement.Cursor) (any, any) {
	if cursor == nil {
		return nil, nil
	}
	return cursor.CreatedAt, cursor.ID
}

func trimManagedPage[T any](items []T, limit int) policymanagement.RepositoryPage[T] {
	page := policymanagement.RepositoryPage[T]{Items: items}
	if len(items) > limit {
		page.Items = items[:limit]
		page.HasMore = true
	}
	return page
}

func cloneManagedInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

var _ policymanagement.Repository = (*policyManagementRepositoryAdapter)(nil)

type policyManagementRepositoryAdapter struct{ store *Store }

func NewPolicyManagementRepository(store *Store) (policymanagement.Repository, error) {
	if store == nil || store.db == nil {
		return nil, policymanagement.ErrUnavailable
	}
	return &policyManagementRepositoryAdapter{store: store}, nil
}

func (adapter *policyManagementRepositoryAdapter) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	return adapter.store.ReadyPolicyManagement(ctx, codec)
}

func (adapter *policyManagementRepositoryAdapter) Replay(ctx context.Context, command managementcommand.Command) (policymanagement.MutationResult, bool, error) {
	return adapter.store.ReplayPolicyCommand(ctx, command)
}

func (adapter *policyManagementRepositoryAdapter) GetAccessPolicy(ctx context.Context, namespaceID, policyID string) (policymanagement.AccessPolicy, error) {
	return adapter.store.GetManagedAccessPolicy(ctx, namespaceID, policyID)
}

func (adapter *policyManagementRepositoryAdapter) ListAccessPolicies(ctx context.Context, query policymanagement.PolicyQuery) (policymanagement.RepositoryPage[policymanagement.AccessPolicy], error) {
	return adapter.store.ListManagedAccessPolicies(ctx, query)
}

func (adapter *policyManagementRepositoryAdapter) GetRateLimitPolicy(ctx context.Context, namespaceID, policyID string) (policymanagement.RateLimitPolicy, error) {
	return adapter.store.GetManagedRateLimitPolicy(ctx, namespaceID, policyID)
}

func (adapter *policyManagementRepositoryAdapter) ListRateLimitPolicies(ctx context.Context, query policymanagement.PolicyQuery) (policymanagement.RepositoryPage[policymanagement.RateLimitPolicy], error) {
	return adapter.store.ListManagedRateLimitPolicies(ctx, query)
}
