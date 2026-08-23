package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

func lockSubjectTeamPolicySelection(
	ctx context.Context,
	tx *sql.Tx,
	mutation subjectmanagement.CreateTeamMutation,
) error {
	if mutation.UseDefaultAccessPolicy || mutation.UseDefaultRateLimitPolicy {
		defaults, err := scanTeamDefaults(tx.QueryRowContext(ctx, subjectLockTeamDefaultsQuery,
			mutation.Team.NamespaceID), mutation.Team.NamespaceID)
		if errors.Is(err, sql.ErrNoRows) {
			return subjectmanagement.ErrDefaultsUnavailable
		}
		if err != nil {
			return fmt.Errorf("lock Team defaults: %w", err)
		}
		if mutation.NamespaceDefaults == nil || defaults.SelfServiceRevision != mutation.NamespaceDefaults.SelfServiceRevision ||
			(mutation.UseDefaultAccessPolicy &&
				(defaults.AccessPolicyID != mutation.NamespaceDefaults.AccessPolicyID ||
					defaults.AccessPolicyRevision != mutation.NamespaceDefaults.AccessPolicyRevision)) ||
			(mutation.UseDefaultRateLimitPolicy &&
				(defaults.RateLimitPolicyID != mutation.NamespaceDefaults.RateLimitPolicyID ||
					defaults.RateLimitPolicyRevision != mutation.NamespaceDefaults.RateLimitPolicyRevision)) {
			return subjectmanagement.ErrDefaultsUnavailable
		}
	}

	policyIDs := make([]string, 0, len(mutation.AccessPolicyBindings))
	for _, binding := range mutation.AccessPolicyBindings {
		policyIDs = append(policyIDs, binding.PolicyID)
	}
	rows, lockSubjectTeamPolicySelectionErr := tx.QueryContext(ctx, subjectLockSelectedAccessPoliciesQuery,
		mutation.Team.NamespaceID, pq.Array(policyIDs))
	if lockSubjectTeamPolicySelectionErr != nil {
		return fmt.Errorf("lock Team AccessPolicies: %w", lockSubjectTeamPolicySelectionErr)
	}
	defer rows.Close()
	selected := make([]string, 0, len(policyIDs))
	for rows.Next() {
		var policyID string
		if err := rows.Scan(&policyID); err != nil {
			return fmt.Errorf("scan Team AccessPolicy: %w", err)
		}
		selected = append(selected, policyID)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("read Team AccessPolicies: %w", err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close Team AccessPolicies: %w", err)
	}
	if len(selected) != len(policyIDs) {
		return subjectmanagement.ErrPolicySelectionUnavailable
	}
	for index := range selected {
		if selected[index] != policyIDs[index] {
			return subjectmanagement.ErrPolicySelectionUnavailable
		}
	}

	var ratePolicyID string
	lockSubjectTeamPolicySelectionErr = tx.QueryRowContext(ctx, subjectLockSelectedRatePolicyQuery,
		mutation.Team.NamespaceID, mutation.RateLimitAllocation.PolicyID).Scan(&ratePolicyID)
	if errors.Is(lockSubjectTeamPolicySelectionErr, sql.ErrNoRows) {
		return subjectmanagement.ErrPolicySelectionUnavailable
	}
	if lockSubjectTeamPolicySelectionErr != nil {
		return fmt.Errorf("lock Team RateLimitPolicy: %w", lockSubjectTeamPolicySelectionErr)
	}
	if ratePolicyID != mutation.RateLimitAllocation.PolicyID {
		return subjectmanagement.ErrPolicySelectionUnavailable
	}
	return nil
}
