package accesscontrol

import (
	"context"
	"encoding/json"
)

type modelPolicyTiers struct {
	key  []string
	user []string
	team []string
}

// ResolveAPIKeyPolicies enriches a page of keys with effective policy in two
// bounded queries. List views therefore stay accurate without issuing queries
// per row.
func (s *Store) ResolveAPIKeyPolicies(ctx context.Context, items []APIKey) ([]APIKey, error) {
	if len(items) == 0 {
		return items, nil
	}
	ids, indexes := prepareAPIKeyPolicyResolution(items)
	if err := s.resolveAPIKeyModelPolicies(ctx, items, ids, indexes); err != nil {
		return nil, err
	}
	if err := s.resolveAPIKeyBudgetPolicies(ctx, items, ids, indexes); err != nil {
		return nil, err
	}
	return items, nil
}

func prepareAPIKeyPolicyResolution(items []APIKey) ([]string, map[string]int) {
	ids := make([]string, len(items))
	indexes := make(map[string]int, len(items))
	for index := range items {
		ids[index] = items[index].ID
		indexes[items[index].ID] = index
		items[index].ModelPatterns = nil
		items[index].ModelPolicySource = ""
		items[index].EffectiveBudgetID = ""
		items[index].BudgetPolicySource = ""
	}
	return ids, indexes
}

func (s *Store) resolveAPIKeyModelPolicies(
	ctx context.Context,
	items []APIKey,
	ids []string,
	indexes map[string]int,
) error {
	rows, err := s.pool.Query(ctx, `
SELECT k.id,b.subject_type,g.model_patterns
FROM access_api_keys k
JOIN access_group_bindings b ON
  (b.subject_type='key' AND b.subject_id=k.id) OR
  (b.subject_type='user' AND b.subject_id=k.user_id) OR
  (b.subject_type='team' AND b.subject_id=COALESCE(k.team_id,k.context_team_id))
JOIN access_groups g ON g.id=b.group_id
	WHERE k.id=ANY($1)`, ids)
	if err != nil {
		return err
	}
	tiers := make(map[string]*modelPolicyTiers, len(items))
	for rows.Next() {
		var keyID, subjectType string
		var raw []byte
		if err = rows.Scan(&keyID, &subjectType, &raw); err != nil {
			rows.Close()
			return err
		}
		var patterns []string
		if err = json.Unmarshal(raw, &patterns); err != nil {
			rows.Close()
			return err
		}
		policy := tiers[keyID]
		if policy == nil {
			policy = &modelPolicyTiers{}
			tiers[keyID] = policy
		}
		switch subjectType {
		case "key":
			policy.key = append(policy.key, patterns...)
		case "user":
			policy.user = append(policy.user, patterns...)
		case "team":
			policy.team = append(policy.team, patterns...)
		}
	}
	if err = rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()
	for keyID, policy := range tiers {
		index := indexes[keyID]
		switch {
		case len(policy.key) > 0:
			items[index].ModelPatterns = uniqueStrings(policy.key)
			items[index].ModelPolicySource = "key"
		case len(policy.user) > 0:
			items[index].ModelPatterns = uniqueStrings(policy.user)
			items[index].ModelPolicySource = "user"
		case len(policy.team) > 0:
			items[index].ModelPatterns = uniqueStrings(policy.team)
			items[index].ModelPolicySource = "team"
		}
	}
	return nil
}

func (s *Store) resolveAPIKeyBudgetPolicies(
	ctx context.Context,
	items []APIKey,
	ids []string,
	indexes map[string]int,
) error {
	budgetRows, err := s.pool.Query(ctx, `
SELECT k.id,COALESCE(k.budget_id,''),COALESCE(u.budget_id,''),COALESCE(t.budget_id,'')
FROM access_api_keys k
LEFT JOIN access_users u ON u.id=k.user_id
LEFT JOIN access_teams t ON t.id=COALESCE(k.team_id,k.context_team_id)
	WHERE k.id=ANY($1)`, ids)
	if err != nil {
		return err
	}
	defer budgetRows.Close()
	for budgetRows.Next() {
		var keyID, keyBudgetID, userBudgetID, teamBudgetID string
		if err = budgetRows.Scan(&keyID, &keyBudgetID, &userBudgetID, &teamBudgetID); err != nil {
			return err
		}
		index := indexes[keyID]
		items[index].EffectiveBudgetID = effectiveBudgetID(keyBudgetID, userBudgetID, teamBudgetID)
		switch items[index].EffectiveBudgetID {
		case keyBudgetID:
			if keyBudgetID != "" {
				items[index].BudgetPolicySource = "key"
			}
		case userBudgetID:
			if userBudgetID != "" {
				items[index].BudgetPolicySource = "user"
			}
		case teamBudgetID:
			if teamBudgetID != "" {
				items[index].BudgetPolicySource = "team"
			}
		}
	}
	if err = budgetRows.Err(); err != nil {
		return err
	}
	return nil
}
