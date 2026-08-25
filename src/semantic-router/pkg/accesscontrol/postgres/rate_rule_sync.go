package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func syncRateLimitRules(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	desired []accesscontrol.RateLimitRule,
) error {
	existing, err := listRateLimitRules(ctx, tx, policyID)
	if err != nil {
		return err
	}
	currentByID := make(map[accesscontrol.RateLimitRuleID]accesscontrol.RateLimitRule, len(existing))
	for _, rule := range existing {
		currentByID[rule.ID] = rule
	}
	if err := validateRetainedRuleSemantics(currentByID, desired); err != nil {
		return err
	}
	if err := assignTemporaryRuleOrdinals(ctx, tx, policyID, existing, desired); err != nil {
		return err
	}
	if err := applyDesiredRules(ctx, tx, policyID, currentByID, desired); err != nil {
		return err
	}
	return deleteRemovedRules(ctx, tx, policyID, existing, desired)
}

func validateRetainedRuleSemantics(
	currentByID map[accesscontrol.RateLimitRuleID]accesscontrol.RateLimitRule,
	desired []accesscontrol.RateLimitRule,
) error {
	for _, rule := range desired {
		current, exists := currentByID[rule.ID]
		if exists && !sameCounterSemantics(current, rule) {
			return fmt.Errorf("rate-limit rule %s changes counter semantics; use a new rule id", rule.ID)
		}
	}
	return nil
}

func applyDesiredRules(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	currentByID map[accesscontrol.RateLimitRuleID]accesscontrol.RateLimitRule,
	desired []accesscontrol.RateLimitRule,
) error {
	for _, rule := range desired {
		if _, exists := currentByID[rule.ID]; exists {
			if err := updateRateLimitRule(ctx, tx, policyID, rule); err != nil {
				return err
			}
			continue
		}
		if err := insertRateLimitRule(ctx, tx, policyID, rule); err != nil {
			return err
		}
	}
	return nil
}

func deleteRemovedRules(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	existing []accesscontrol.RateLimitRule,
	desired []accesscontrol.RateLimitRule,
) error {
	desiredIDs := make(map[accesscontrol.RateLimitRuleID]struct{}, len(desired))
	for _, rule := range desired {
		desiredIDs[rule.ID] = struct{}{}
	}
	for _, rule := range existing {
		if _, retained := desiredIDs[rule.ID]; retained {
			continue
		}
		result, err := tx.ExecContext(ctx, deleteRateLimitRuleQuery, policyID, rule.ID)
		if err != nil {
			return fmt.Errorf("delete rate-limit rule: %w", err)
		}
		if err := requireOneRow(result, ErrRevisionConflict); err != nil {
			return err
		}
	}
	return nil
}

func updateRateLimitRule(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	rule accesscontrol.RateLimitRule,
) error {
	stored, err := encodeRateLimitRule(rule)
	if err != nil {
		return err
	}
	result, err := tx.ExecContext(ctx, updateRateLimitRuleQuery,
		policyID, rule.ID, rule.Metric, rule.Algorithm,
		stored.limitValue, stored.windowSeconds, stored.calendarPeriod, stored.timezone,
		stored.bucketCapacity, stored.refillAmount, stored.refillPeriodMilliseconds,
		stored.gcraEmissionIntervalMicros, stored.gcraBurstTolerance,
		rule.Accounting, rule.Enforcement, int64(rule.Ordinal))
	if err != nil {
		return fmt.Errorf("update rate-limit rule: %w", err)
	}
	return requireOneRow(result, ErrRevisionConflict)
}

func assignTemporaryRuleOrdinals(
	ctx context.Context,
	tx *sql.Tx,
	policyID accesscontrol.RateLimitPolicyID,
	existing []accesscontrol.RateLimitRule,
	desired []accesscontrol.RateLimitRule,
) error {
	excluded := make(map[uint32]struct{}, len(existing)+len(desired))
	for _, rule := range existing {
		excluded[rule.Ordinal] = struct{}{}
	}
	for _, rule := range desired {
		excluded[rule.Ordinal] = struct{}{}
	}
	candidate := int64(math.MaxInt32)
	for _, rule := range existing {
		for candidate >= 0 {
			ordinal, conversionErr := nonNegativeUint32(candidate, "temporary rate-limit ordinal")
			if conversionErr != nil {
				return conversionErr
			}
			if _, occupied := excluded[ordinal]; !occupied {
				break
			}
			candidate--
		}
		if candidate < 0 {
			return fmt.Errorf("no temporary rate-limit ordinal is available")
		}
		result, err := tx.ExecContext(ctx, setRateLimitRuleOrdinalQuery, policyID, rule.ID, candidate)
		if err != nil {
			return fmt.Errorf("stage rate-limit rule ordinal: %w", err)
		}
		if err := requireOneRow(result, ErrRevisionConflict); err != nil {
			return err
		}
		excluded[uint32(candidate)] = struct{}{}
		candidate--
	}
	return nil
}

func sameCounterSemantics(left, right accesscontrol.RateLimitRule) bool {
	return left.Metric == right.Metric &&
		left.Algorithm == right.Algorithm &&
		left.Window == right.Window &&
		left.CalendarPeriod == right.CalendarPeriod &&
		left.Timezone == right.Timezone &&
		left.BucketCapacity == right.BucketCapacity &&
		left.RefillAmount == right.RefillAmount &&
		left.RefillPeriod == right.RefillPeriod &&
		left.GCRAEmissionInterval == right.GCRAEmissionInterval &&
		equalOptionalInt64(left.GCRABurstTolerance, right.GCRABurstTolerance) &&
		left.Accounting == right.Accounting
}

func equalOptionalInt64(left, right *int64) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}
