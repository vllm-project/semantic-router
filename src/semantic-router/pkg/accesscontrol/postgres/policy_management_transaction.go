package postgres

import (
	"context"
	"database/sql"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

// managedRateLimitOverride is the PostgreSQL orchestration seam used by API-key
// creation. Exactly one of PolicyID and InlinePolicy must be set. Binding owns
// its counter scope and is always materialized as a key allocation.
type managedRateLimitOverride struct {
	PolicyID     string
	InlinePolicy *policymanagement.RateLimitPolicy
	Binding      policymanagement.RateLimitBinding
}

// managedRateLimitOverrideResult is safe to place in an API-key command's
// encrypted idempotent response. Created reports whether this command created
// the ordinary reusable policy; a binding is always created with an override.
type managedRateLimitOverrideResult struct {
	Policy  policymanagement.RateLimitPolicy
	Binding policymanagement.RateLimitBinding
	Created bool
}

// materializeManagedAccessBinding performs every relational check and insert
// needed for an AccessPolicy binding without opening or committing a
// transaction. Explicit binding commands and aggregate workflows such as API-
// key creation therefore share one all-or-nothing write boundary.
func materializeManagedAccessBinding(
	ctx context.Context,
	tx *sql.Tx,
	binding policymanagement.AccessPolicyBinding,
) (policymanagement.AccessPolicyBinding, error) {
	if validateNewManagedAccessBinding(binding) != nil {
		return policymanagement.AccessPolicyBinding{}, policymanagement.ErrInvalidRequest
	}
	if err := lockManagedBindingReferences(ctx, tx, binding.NamespaceID,
		binding.PolicyID, binding.Subject, false); err != nil {
		return policymanagement.AccessPolicyBinding{}, err
	}
	return insertAndReadManagedAccessBinding(ctx, tx, binding)
}

// materializeManagedRateBinding has the same caller-owned transaction contract
// and additionally resolves the namespace quota partition before insertion.
func materializeManagedRateBinding(
	ctx context.Context,
	tx *sql.Tx,
	binding policymanagement.RateLimitBinding,
) (policymanagement.RateLimitBinding, error) {
	if validateNewManagedRateBinding(binding) != nil {
		return policymanagement.RateLimitBinding{}, policymanagement.ErrInvalidRequest
	}
	return createManagedRateBinding(ctx, tx, binding)
}

// materializeManagedInlineRateLimit creates a normal reusable policy and its
// binding in the caller's transaction. It deliberately emits no audit/outbox
// record and completes no command: the aggregate orchestrator appends all of
// those records once, together with the API key or explicit binding mutation.
func materializeManagedInlineRateLimit(
	ctx context.Context,
	tx *sql.Tx,
	policy policymanagement.RateLimitPolicy,
	binding policymanagement.RateLimitBinding,
) (managedRateLimitOverrideResult, error) {
	if validateNewManagedRatePolicy(policy) != nil || validateNewManagedRateBinding(binding) != nil ||
		binding.PolicyID != policy.ID || binding.NamespaceID != policy.NamespaceID {
		return managedRateLimitOverrideResult{}, policymanagement.ErrInvalidRequest
	}
	createdPolicy, err := insertManagedRatePolicy(ctx, tx, policy)
	if err != nil {
		return managedRateLimitOverrideResult{}, err
	}
	createdBinding, err := materializeManagedRateBinding(ctx, tx, binding)
	if err != nil {
		return managedRateLimitOverrideResult{}, err
	}
	return managedRateLimitOverrideResult{
		Policy: createdPolicy, Binding: createdBinding, Created: true,
	}, nil
}

// materializeManagedAPIKeyRateLimitOverride is the atomic API-key rate-policy
// composition seam. The API-key row and subject must already have been inserted
// by the same transaction. A caller either binds an existing policy or creates
// and binds one inline; invalid one-of or non-key/non-allocation inputs fail
// before a partial binding can exist.
func materializeManagedAPIKeyRateLimitOverride(
	ctx context.Context,
	tx *sql.Tx,
	override managedRateLimitOverride,
) (managedRateLimitOverrideResult, error) {
	hasPolicyID := override.PolicyID != ""
	hasInline := override.InlinePolicy != nil
	if hasPolicyID == hasInline || override.Binding.Subject.Type != accesscontrol.SubjectKindAPIKey ||
		override.Binding.Mode != accesscontrol.RateBindingAllocation {
		return managedRateLimitOverrideResult{}, policymanagement.ErrInvalidRequest
	}

	if hasInline {
		override.Binding.PolicyID = override.InlinePolicy.ID
		return materializeManagedInlineRateLimit(ctx, tx, *override.InlinePolicy, override.Binding)
	}

	override.Binding.PolicyID = override.PolicyID
	binding, err := materializeManagedRateBinding(ctx, tx, override.Binding)
	if err != nil {
		return managedRateLimitOverrideResult{}, err
	}
	policy, err := readManagedRateLimitPolicyTx(ctx, tx, binding.NamespaceID, binding.PolicyID)
	if err != nil {
		return managedRateLimitOverrideResult{}, err
	}
	return managedRateLimitOverrideResult{Policy: policy, Binding: binding}, nil
}

func readManagedRateLimitPolicyTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID, policyID string,
) (policymanagement.RateLimitPolicy, error) {
	policy, err := scanManagedRateLimitPolicy(tx.QueryRowContext(ctx,
		getManagedRatePolicyQuery, namespaceID, policyID))
	if err != nil {
		return policymanagement.RateLimitPolicy{}, mapManagedPolicyRead(err, "read RateLimitPolicy")
	}
	rules, err := loadManagedRateRules(ctx, tx, []string{policy.ID})
	if err != nil {
		return policymanagement.RateLimitPolicy{}, err
	}
	policy.Rules = rules[policy.ID]
	if err := validateStoredManagedRatePolicy(policy); err != nil {
		return policymanagement.RateLimitPolicy{}, err
	}
	return policy, nil
}
