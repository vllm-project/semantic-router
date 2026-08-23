package accesspublisher

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func loadAccessPolicies(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy, returnErr error) {
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT id, name, status, revision, created_at, updated_at
FROM access_policies WHERE namespace_id = $1`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list access policies: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy)
	for rows.Next() {
		var policy accesscontrol.AccessPolicy
		var revision int64
		policy.NamespaceID = namespaceID
		if err := rows.Scan(&policy.ID, &policy.DisplayName, &policy.Status, &revision, &policy.CreatedAt, &policy.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan access policy: %w", err)
		}
		if revision <= 0 {
			return nil, fmt.Errorf("access policy %s has invalid revision", policy.ID)
		}
		policy.Revision = accesscontrol.Revision(revision)
		result[policy.ID] = policy
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	grantRows, queryContextErr := tx.QueryContext(ctx, `SELECT g.policy_id, g.resource_type, g.resource_id, g.permission, g.effect
FROM access_policy_grants g JOIN access_policies p ON p.id = g.policy_id WHERE p.namespace_id = $1`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list access grants: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, grantRows.Close())
	}()
	for grantRows.Next() {
		var grant accesscontrol.AccessPolicyGrant
		if err := grantRows.Scan(&grant.PolicyID, &grant.Resource.Type, &grant.Resource.ID, &grant.Permission, &grant.Effect); err != nil {
			return nil, fmt.Errorf("scan access grant: %w", err)
		}
		policy, exists := result[grant.PolicyID]
		if !exists {
			return nil, fmt.Errorf("grant references missing access policy %s", grant.PolicyID)
		}
		policy.Grants = append(policy.Grants, grant)
		result[grant.PolicyID] = policy
	}
	return result, grantRows.Err()
}

func loadRatePolicies(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy, returnErr error) {
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT id, name, status, revision, created_at, updated_at
FROM rate_limit_policies WHERE namespace_id = $1`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list rate-limit policies: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy)
	for rows.Next() {
		var policy accesscontrol.RateLimitPolicy
		var revision int64
		policy.NamespaceID = namespaceID
		if err := rows.Scan(&policy.ID, &policy.DisplayName, &policy.Status, &revision, &policy.CreatedAt, &policy.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan rate-limit policy: %w", err)
		}
		if revision <= 0 {
			return nil, fmt.Errorf("rate-limit policy %s has invalid revision", policy.ID)
		}
		policy.Revision = accesscontrol.Revision(revision)
		result[policy.ID] = policy
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	ruleRows, queryContextErr := tx.QueryContext(ctx, `SELECT r.id, r.policy_id, r.metric, r.algorithm, r.limit_value,
r.window_seconds, r.calendar_period, r.timezone, r.bucket_capacity, r.refill_amount,
r.refill_period_milliseconds, r.gcra_emission_interval_microseconds, r.gcra_burst_tolerance,
r.accounting, r.enforcement, r.ordinal
FROM rate_limit_rules r JOIN rate_limit_policies p ON p.id = r.policy_id
WHERE p.namespace_id = $1 ORDER BY r.policy_id, r.ordinal, r.id`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list rate-limit rules: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, ruleRows.Close())
	}()
	for ruleRows.Next() {
		rule, err := scanDesiredRateRule(ruleRows)
		if err != nil {
			return nil, err
		}
		policy, exists := result[rule.PolicyID]
		if !exists {
			return nil, fmt.Errorf("rule references missing rate-limit policy %s", rule.PolicyID)
		}
		policy.Rules = append(policy.Rules, rule)
		result[rule.PolicyID] = policy
	}
	return result, ruleRows.Err()
}

func scanDesiredRateRule(scanner interface{ Scan(...any) error }) (accesscontrol.RateLimitRule, error) {
	var rule accesscontrol.RateLimitRule
	var limit, calendarPeriod, timezone, bucketCapacity, refillAmount sql.NullString
	var windowSeconds, refillMilliseconds, emissionMicroseconds, burst sql.NullInt64
	var ordinal int64
	if err := scanner.Scan(
		&rule.ID, &rule.PolicyID, &rule.Metric, &rule.Algorithm, &limit,
		&windowSeconds, &calendarPeriod, &timezone, &bucketCapacity, &refillAmount,
		&refillMilliseconds, &emissionMicroseconds, &burst, &rule.Accounting, &rule.Enforcement, &ordinal,
	); err != nil {
		return accesscontrol.RateLimitRule{}, fmt.Errorf("scan rate-limit rule: %w", err)
	}
	if ordinal < 0 || ordinal > math.MaxUint32 {
		return accesscontrol.RateLimitRule{}, fmt.Errorf("rate-limit rule ordinal is invalid")
	}
	rule.Ordinal = uint32(ordinal)
	if limit.Valid {
		value := limit.String
		if rule.Metric == accesscontrol.RateMetricCost {
			var err error
			value, err = unscaleCostLimit(value)
			if err != nil {
				return accesscontrol.RateLimitRule{}, err
			}
		}
		rule.Limit = accesscontrol.QuotaValue(value)
	}
	if bucketCapacity.Valid {
		rule.BucketCapacity = accesscontrol.QuotaValue(bucketCapacity.String)
	}
	if refillAmount.Valid {
		rule.RefillAmount = accesscontrol.QuotaValue(refillAmount.String)
	}
	if windowSeconds.Valid {
		rule.Window = time.Duration(windowSeconds.Int64) * time.Second
	}
	if refillMilliseconds.Valid {
		rule.RefillPeriod = time.Duration(refillMilliseconds.Int64) * time.Millisecond
	}
	if emissionMicroseconds.Valid {
		rule.GCRAEmissionInterval = time.Duration(emissionMicroseconds.Int64) * time.Microsecond
	}
	if burst.Valid {
		value := burst.Int64
		rule.GCRABurstTolerance = &value
	}
	if calendarPeriod.Valid {
		rule.CalendarPeriod = accesscontrol.CalendarPeriod(calendarPeriod.String)
	}
	if timezone.Valid {
		rule.Timezone = timezone.String
	}
	if err := rule.Validate(); err != nil {
		return accesscontrol.RateLimitRule{}, fmt.Errorf("validate stored rate-limit rule %s: %w", rule.ID, err)
	}
	return rule, nil
}

func unscaleCostLimit(value string) (string, error) {
	integer, ok := new(big.Int).SetString(value, 10)
	if !ok || integer.Sign() < 0 {
		return "", fmt.Errorf("invalid stored cost limit")
	}
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(15), nil)
	whole, remainder := new(big.Int), new(big.Int)
	whole.QuoRem(integer, scale, remainder)
	if remainder.Sign() == 0 {
		return whole.String(), nil
	}
	text := remainder.String()
	fraction := strings.TrimRight(strings.Repeat("0", 15-len(text))+text, "0")
	return whole.String() + "." + fraction, nil
}

func loadAccessBindings(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	subjects map[string]accesscontrol.SubjectRef,
) (_ map[string][]accesscontrol.AccessPolicyBinding, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, policy_id, subject_id, status, revision
FROM access_policy_bindings WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, fmt.Errorf("list access bindings: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string][]accesscontrol.AccessPolicyBinding)
	for rows.Next() {
		var binding accesscontrol.AccessPolicyBinding
		var subjectID string
		var revision int64
		binding.NamespaceID = namespaceID
		if err := rows.Scan(&binding.ID, &binding.PolicyID, &subjectID, &binding.Status, &revision); err != nil {
			return nil, fmt.Errorf("scan access binding: %w", err)
		}
		subject, exists := subjects[subjectID]
		if !exists || revision <= 0 {
			return nil, fmt.Errorf("access binding %s has invalid subject or revision", binding.ID)
		}
		binding.Subject, binding.Revision = subject, accesscontrol.Revision(revision)
		result[subjectID] = append(result[subjectID], binding)
	}
	return result, rows.Err()
}

func loadRateBindings(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	subjects map[string]accesscontrol.SubjectRef,
) (_ map[string][]accesscontrol.RateLimitBinding, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, policy_id, subject_id, binding_mode, quota_partition_id, status, revision
FROM rate_limit_bindings WHERE namespace_id = $1`, namespace.ID)
	if err != nil {
		return nil, fmt.Errorf("list rate-limit bindings: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string][]accesscontrol.RateLimitBinding)
	for rows.Next() {
		var binding accesscontrol.RateLimitBinding
		var subjectID string
		var revision int64
		binding.NamespaceID = namespace.ID
		if err := rows.Scan(&binding.ID, &binding.PolicyID, &subjectID, &binding.Mode,
			&binding.QuotaPartitionID, &binding.Status, &revision); err != nil {
			return nil, fmt.Errorf("scan rate-limit binding: %w", err)
		}
		subject, exists := subjects[subjectID]
		if !exists || revision <= 0 || binding.QuotaPartitionID != namespace.QuotaPartitionID {
			return nil, fmt.Errorf("rate-limit binding %s has invalid subject, revision, or partition", binding.ID)
		}
		binding.Subject, binding.Revision = subject, accesscontrol.Revision(revision)
		result[subjectID] = append(result[subjectID], binding)
	}
	return result, rows.Err()
}
