package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	getRateLimitPolicyQuery = `SELECT id, namespace_id, name, status, revision, created_at, updated_at
FROM rate_limit_policies
WHERE namespace_id = $1 AND id = $2`
	rateLimitRuleColumns = `id, policy_id, metric, algorithm, limit_value, window_seconds,
       calendar_period, timezone, bucket_capacity, refill_amount,
       refill_period_milliseconds, gcra_emission_interval_microseconds,
       gcra_burst_tolerance, accounting, enforcement, ordinal`
	listRateLimitRulesQuery = `SELECT ` + rateLimitRuleColumns + `
FROM rate_limit_rules
WHERE policy_id = $1
ORDER BY ordinal, id`
	insertRateLimitPolicyQuery = `INSERT INTO rate_limit_policies
  (id, namespace_id, name, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, 1, $5, $6)
RETURNING id, namespace_id, name, status, revision, created_at, updated_at`
	updateRateLimitPolicyQuery = `UPDATE rate_limit_policies
SET name = $4, status = $5, revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND revision = $3
RETURNING id, namespace_id, name, status, revision, created_at, updated_at`
	insertRateLimitRuleQuery = `INSERT INTO rate_limit_rules
  (id, policy_id, metric, algorithm, limit_value, window_seconds,
   calendar_period, timezone, bucket_capacity, refill_amount,
   refill_period_milliseconds, gcra_emission_interval_microseconds,
   gcra_burst_tolerance, accounting, enforcement, ordinal)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)`
	updateRateLimitRuleQuery = `UPDATE rate_limit_rules
SET metric = $3, algorithm = $4, limit_value = $5, window_seconds = $6,
    calendar_period = $7, timezone = $8, bucket_capacity = $9,
    refill_amount = $10, refill_period_milliseconds = $11,
    gcra_emission_interval_microseconds = $12, gcra_burst_tolerance = $13,
    accounting = $14, enforcement = $15, ordinal = $16
WHERE policy_id = $1 AND id = $2`
	setRateLimitRuleOrdinalQuery = `UPDATE rate_limit_rules
SET ordinal = $3
WHERE policy_id = $1 AND id = $2`
	deleteRateLimitRuleQuery = `DELETE FROM rate_limit_rules
WHERE policy_id = $1 AND id = $2`
)

func (s *Store) GetRateLimitPolicy(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	id accesscontrol.RateLimitPolicyID,
) (accesscontrol.RateLimitPolicy, error) {
	if err := validateIdentityIDs(namespaceID, string(id)); err != nil {
		return accesscontrol.RateLimitPolicy{}, err
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (accesscontrol.RateLimitPolicy, error) {
		policy, err := scanRateLimitPolicy(tx.QueryRowContext(ctx, getRateLimitPolicyQuery, namespaceID, id))
		if errors.Is(err, sql.ErrNoRows) {
			return accesscontrol.RateLimitPolicy{}, ErrNotFound
		}
		if err != nil {
			return accesscontrol.RateLimitPolicy{}, fmt.Errorf("get rate-limit policy: %w", err)
		}
		rules, err := listRateLimitRules(ctx, tx, id)
		if err != nil {
			return accesscontrol.RateLimitPolicy{}, err
		}
		policy.Rules = rules
		if err := policy.Validate(); err != nil {
			return accesscontrol.RateLimitPolicy{}, fmt.Errorf("validate stored rate-limit policy: %w", err)
		}
		return policy, nil
	})
}

func (s *Store) CreateRateLimitPolicy(
	ctx context.Context,
	policy accesscontrol.RateLimitPolicy,
	meta MutationMeta,
) (MutationResult[accesscontrol.RateLimitPolicy], error) {
	if err := validateRateLimitPolicyForWrite(policy, 1); err != nil {
		return MutationResult[accesscontrol.RateLimitPolicy]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.RateLimitPolicy]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.RateLimitPolicy], error) {
		created, createRateLimitPolicyErr := scanRateLimitPolicy(tx.QueryRowContext(ctx, insertRateLimitPolicyQuery,
			policy.ID, policy.NamespaceID, policy.DisplayName, policy.Status,
			policy.CreatedAt, policy.UpdatedAt))
		if createRateLimitPolicyErr != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, fmt.Errorf("insert rate-limit policy: %w", createRateLimitPolicyErr)
		}
		if err := insertRateLimitRules(ctx, tx, created.ID, policy.Rules); err != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, err
		}
		created.Rules = policy.Rules
		receipt, createRateLimitPolicyErr := appendMutationRecords(ctx, tx, policy.NamespaceID, outboxMutation{
			AggregateType: "rate_limit_policy", AggregateID: string(policy.ID),
			AggregateRevision: created.Revision, Operation: outboxCreated,
		}, meta)
		if createRateLimitPolicyErr != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, createRateLimitPolicyErr
		}
		return MutationResult[accesscontrol.RateLimitPolicy]{Value: created, Receipt: receipt}, nil
	})
}

func (s *Store) UpdateRateLimitPolicy(
	ctx context.Context,
	policy accesscontrol.RateLimitPolicy,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[accesscontrol.RateLimitPolicy], error) {
	if err := validateRateLimitPolicyForWrite(policy, expected); err != nil {
		return MutationResult[accesscontrol.RateLimitPolicy]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.RateLimitPolicy]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.RateLimitPolicy]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.RateLimitPolicy], error) {
		updated, updateRateLimitPolicyErr := scanRateLimitPolicy(tx.QueryRowContext(ctx, updateRateLimitPolicyQuery,
			policy.NamespaceID, policy.ID, expectedRevision, policy.DisplayName, policy.Status))
		if errors.Is(updateRateLimitPolicyErr, sql.ErrNoRows) {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, ErrRevisionConflict
		}
		if updateRateLimitPolicyErr != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, fmt.Errorf("update rate-limit policy: %w", updateRateLimitPolicyErr)
		}
		if err := syncRateLimitRules(ctx, tx, updated.ID, policy.Rules); err != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, err
		}
		updated.Rules = policy.Rules
		receipt, updateRateLimitPolicyErr := appendMutationRecords(ctx, tx, policy.NamespaceID, outboxMutation{
			AggregateType: "rate_limit_policy", AggregateID: string(policy.ID),
			AggregateRevision: updated.Revision, Operation: outboxUpdated,
		}, meta)
		if updateRateLimitPolicyErr != nil {
			return MutationResult[accesscontrol.RateLimitPolicy]{}, updateRateLimitPolicyErr
		}
		return MutationResult[accesscontrol.RateLimitPolicy]{Value: updated, Receipt: receipt}, nil
	})
}

func validateRateLimitPolicyForWrite(policy accesscontrol.RateLimitPolicy, expected accesscontrol.Revision) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if policy.Revision != expected {
		return fmt.Errorf("rate-limit policy revision must match expected revision")
	}
	if err := validateIdentityIDs(policy.NamespaceID, string(policy.ID)); err != nil {
		return err
	}
	for _, rule := range policy.Rules {
		if err := validateUUID("rate-limit rule id", string(rule.ID)); err != nil {
			return err
		}
		if _, err := encodeRateLimitRule(rule); err != nil {
			return err
		}
	}
	return nil
}

func insertRateLimitRules(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	rules []accesscontrol.RateLimitRule,
) error {
	for _, rule := range rules {
		if err := insertRateLimitRule(ctx, tx, policyID, rule); err != nil {
			return err
		}
	}
	return nil
}

func insertRateLimitRule(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	rule accesscontrol.RateLimitRule,
) error {
	stored, err := encodeRateLimitRule(rule)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, insertRateLimitRuleQuery,
		rule.ID, policyID, rule.Metric, rule.Algorithm,
		stored.limitValue, stored.windowSeconds, stored.calendarPeriod, stored.timezone,
		stored.bucketCapacity, stored.refillAmount, stored.refillPeriodMilliseconds,
		stored.gcraEmissionIntervalMicros, stored.gcraBurstTolerance,
		rule.Accounting, rule.Enforcement, int64(rule.Ordinal)); err != nil {
		return fmt.Errorf("insert rate-limit rule: %w", err)
	}
	return nil
}

func listRateLimitRules(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
) ([]accesscontrol.RateLimitRule, error) {
	rows, err := tx.QueryContext(ctx, listRateLimitRulesQuery, policyID)
	if err != nil {
		return nil, fmt.Errorf("list rate-limit rules: %w", err)
	}
	defer rows.Close()
	rules := make([]accesscontrol.RateLimitRule, 0)
	for rows.Next() {
		rule, err := scanRateLimitRule(rows)
		if err != nil {
			return nil, fmt.Errorf("scan rate-limit rule: %w", err)
		}
		rules = append(rules, rule)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate rate-limit rules: %w", err)
	}
	return rules, nil
}

func scanRateLimitPolicy(scanner rowScanner) (accesscontrol.RateLimitPolicy, error) {
	var policy accesscontrol.RateLimitPolicy
	var revision int64
	if err := scanner.Scan(
		&policy.ID, &policy.NamespaceID, &policy.DisplayName, &policy.Status,
		&revision, &policy.CreatedAt, &policy.UpdatedAt,
	); err != nil {
		return accesscontrol.RateLimitPolicy{}, err
	}
	parsedRevision, err := scanRevision(revision)
	if err != nil {
		return accesscontrol.RateLimitPolicy{}, err
	}
	policy.Revision = parsedRevision
	return policy, nil
}
