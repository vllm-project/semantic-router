package accesscontrol

import (
	"context"
	"encoding/json"
	"time"

	"github.com/jackc/pgx/v5"
)

func (s *Store) PrincipalByDigest(ctx context.Context, digest string) (*Principal, error) {
	var keyID string
	if err := s.pool.QueryRow(ctx, `SELECT id FROM access_api_keys WHERE digest=$1`, digest).Scan(&keyID); err != nil {
		return nil, err
	}
	key, err := s.GetAPIKey(ctx, keyID)
	if err != nil || key.Status != StatusActive || (key.ExpiresAt != nil && key.ExpiresAt.Before(time.Now())) {
		return nil, pgx.ErrNoRows
	}
	principal := &Principal{Key: key}
	if key.UserID != "" {
		user, userErr := s.GetUser(ctx, key.UserID)
		if userErr != nil || user.Status != StatusActive {
			return nil, pgx.ErrNoRows
		}
		principal.User = &user
	}
	teamID := key.ContextTeamID
	if key.TeamID != "" {
		teamID = key.TeamID
	}
	if teamID != "" {
		team, teamErr := s.GetTeam(ctx, teamID)
		if teamErr != nil || team.Status != StatusActive {
			return nil, pgx.ErrNoRows
		}
		if key.UserID != "" {
			var member bool
			if memberErr := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM access_team_members WHERE team_id=$1 AND user_id=$2)`, teamID, key.UserID).Scan(&member); memberErr != nil || !member {
				return nil, pgx.ErrNoRows
			}
		}
		principal.Team = &team
	}
	principal.ModelPatterns, err = s.ModelPatternsForKey(ctx, key)
	if err != nil {
		return nil, err
	}
	userBudgetID, teamBudgetID := "", ""
	if principal.User != nil {
		userBudgetID = principal.User.BudgetID
	}
	if principal.Team != nil {
		teamBudgetID = principal.Team.BudgetID
	}
	if budgetID := effectiveBudgetID(key.BudgetID, userBudgetID, teamBudgetID); budgetID != "" {
		budget, budgetErr := s.GetBudget(ctx, budgetID)
		if budgetErr != nil {
			return nil, budgetErr
		}
		if budget.Enabled {
			principal.Budgets = []Budget{budget}
		}
	}
	_, _ = s.pool.Exec(ctx, `UPDATE access_api_keys SET last_used_at=NOW() WHERE id=$1`, key.ID)
	return principal, nil
}

// PrincipalForDashboardUser resolves the user's self-service API key so
// Playground requests follow the same grants, quota, and accounting path.
func (s *Store) PrincipalForDashboardUser(ctx context.Context, userID string) (*Principal, error) {
	if userID == "" {
		return nil, pgx.ErrNoRows
	}
	var digest string
	err := s.pool.QueryRow(ctx, `
SELECT digest FROM access_api_keys
WHERE user_id=$1 AND status='active' AND (expires_at IS NULL OR expires_at>NOW())
ORDER BY created_at ASC LIMIT 1`, userID).Scan(&digest)
	if err != nil {
		return nil, err
	}
	return s.PrincipalByDigest(ctx, digest)
}

// ModelPolicyForKey applies the policy precedence Key > User > Team and
// returns the tier that supplied the effective model grant.
func (s *Store) ModelPolicyForKey(ctx context.Context, key APIKey) ([]string, string, error) {
	teamID := key.ContextTeamID
	if key.TeamID != "" {
		teamID = key.TeamID
	}
	rows, err := s.pool.Query(ctx, `
SELECT DISTINCT b.subject_type,g.model_patterns FROM access_groups g
JOIN access_group_bindings b ON b.group_id=g.id
WHERE (b.subject_type='key' AND b.subject_id=$1)
   OR (b.subject_type='user' AND b.subject_id=$2)
   OR (b.subject_type='team' AND b.subject_id=$3)`, key.ID, key.UserID, teamID)
	if err != nil {
		return nil, "", err
	}
	defer rows.Close()
	var keyPatterns, userPatterns, teamPatterns []string
	for rows.Next() {
		var subjectType string
		var raw []byte
		var patterns []string
		if err := rows.Scan(&subjectType, &raw); err != nil {
			return nil, "", err
		}
		if err := json.Unmarshal(raw, &patterns); err != nil {
			return nil, "", err
		}
		switch subjectType {
		case "key":
			keyPatterns = append(keyPatterns, patterns...)
		case "user":
			userPatterns = append(userPatterns, patterns...)
		case "team":
			teamPatterns = append(teamPatterns, patterns...)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, "", err
	}
	if patterns := uniqueStrings(keyPatterns); len(patterns) > 0 {
		return patterns, "key", nil
	}
	if patterns := uniqueStrings(userPatterns); len(patterns) > 0 {
		return patterns, "user", nil
	}
	return uniqueStrings(teamPatterns), choose(len(teamPatterns) > 0, "team", ""), nil
}

func (s *Store) ModelPatternsForKey(ctx context.Context, key APIKey) ([]string, error) {
	patterns, _, err := s.ModelPolicyForKey(ctx, key)
	return patterns, err
}

func (s *Store) BudgetPolicyForKey(ctx context.Context, key APIKey) (string, string, error) {
	userBudgetID, teamBudgetID := "", ""
	if key.UserID != "" {
		if err := s.pool.QueryRow(ctx, `SELECT COALESCE(budget_id,'') FROM access_users WHERE id=$1`, key.UserID).Scan(&userBudgetID); err != nil {
			return "", "", err
		}
	}
	teamID := key.ContextTeamID
	if key.TeamID != "" {
		teamID = key.TeamID
	}
	if teamID != "" {
		if err := s.pool.QueryRow(ctx, `SELECT budget_id FROM access_teams WHERE id=$1`, teamID).Scan(&teamBudgetID); err != nil {
			return "", "", err
		}
	}
	budgetID := effectiveBudgetID(key.BudgetID, userBudgetID, teamBudgetID)
	if budgetID == "" {
		return "", "", nil
	}
	switch budgetID {
	case key.BudgetID:
		return budgetID, "key", nil
	case userBudgetID:
		return budgetID, "user", nil
	case teamBudgetID:
		return budgetID, "team", nil
	default:
		return "", "", nil
	}
}
