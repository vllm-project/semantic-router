package accesscontrol

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

func (s *Store) PrincipalByDigest(ctx context.Context, digest string) (*Principal, error) {
	var principal Principal
	err := s.pool.QueryRow(ctx, `
SELECT id,name,prefix,COALESCE(user_id,''),COALESCE(team_id,''),COALESCE(budget_id,''),status,expires_at,last_used_at,created_at,updated_at
FROM access_api_keys WHERE digest=$1`, digest).
		Scan(&principal.Key.ID, &principal.Key.Name, &principal.Key.Prefix, &principal.Key.UserID, &principal.Key.TeamID, &principal.Key.BudgetID,
			&principal.Key.Status, &principal.Key.ExpiresAt, &principal.Key.LastUsed, &principal.Key.CreatedAt, &principal.Key.UpdatedAt)
	if err != nil {
		return nil, err
	}
	if principal.Key.Status != StatusActive || (principal.Key.ExpiresAt != nil && principal.Key.ExpiresAt.Before(time.Now())) {
		return nil, pgx.ErrNoRows
	}
	if principal.Key.UserID != "" {
		var user User
		if userErr := s.pool.QueryRow(ctx, `SELECT id,email,name,status,created_at,updated_at FROM access_users WHERE id=$1`, principal.Key.UserID).
			Scan(&user.ID, &user.Email, &user.Name, &user.Status, &user.CreatedAt, &user.UpdatedAt); userErr != nil || user.Status != StatusActive {
			return nil, pgx.ErrNoRows
		}
		principal.User = &user
		if principal.Key.TeamID == "" {
			var team Team
			teamErr := s.pool.QueryRow(ctx, `
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at
FROM access_teams t JOIN access_team_members m ON m.team_id=t.id
WHERE m.user_id=$1 AND t.status='active'
ORDER BY t.created_at ASC LIMIT 1`, principal.Key.UserID).
				Scan(&team.ID, &team.Name, &team.Description, &team.Status, &team.CreatedAt, &team.UpdatedAt)
			if teamErr == nil {
				principal.Team = &team
				principal.Key.TeamID = team.ID
			} else if !errors.Is(teamErr, pgx.ErrNoRows) {
				return nil, teamErr
			}
		}
	}
	if principal.Key.TeamID != "" {
		var team Team
		if teamErr := s.pool.QueryRow(ctx, `SELECT id,name,description,status,created_at,updated_at FROM access_teams WHERE id=$1`, principal.Key.TeamID).
			Scan(&team.ID, &team.Name, &team.Description, &team.Status, &team.CreatedAt, &team.UpdatedAt); teamErr != nil || team.Status != StatusActive {
			return nil, pgx.ErrNoRows
		}
		principal.Team = &team
	}
	principal.ModelPatterns, err = s.ModelPatternsForKey(ctx, principal.Key)
	if err != nil {
		return nil, err
	}

	budgetRows, err := s.pool.Query(ctx, `
SELECT id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled,created_at,updated_at
FROM access_budgets WHERE enabled=TRUE AND (
 scope_type='global' OR (scope_type='key' AND scope_id=$1) OR
 (scope_type='user' AND scope_id=$2) OR (scope_type='team' AND scope_id=$3) OR id=NULLIF($4,''))`,
		principal.Key.ID, principal.Key.UserID, principal.Key.TeamID, principal.Key.BudgetID)
	if err != nil {
		return nil, err
	}
	for budgetRows.Next() {
		var budget Budget
		if err := budgetRows.Scan(&budget.ID, &budget.Name, &budget.ScopeType, &budget.ScopeID, &budget.RPM, &budget.TPM, &budget.DailyTokens, &budget.Enabled, &budget.CreatedAt, &budget.UpdatedAt); err != nil {
			budgetRows.Close()
			return nil, err
		}
		principal.Budgets = append(principal.Budgets, budget)
	}
	if err := budgetRows.Err(); err != nil {
		budgetRows.Close()
		return nil, err
	}
	budgetRows.Close()
	_, _ = s.pool.Exec(ctx, `UPDATE access_api_keys SET last_used_at=NOW() WHERE id=$1`, principal.Key.ID)
	return &principal, nil
}

// PrincipalForDashboardUser resolves the user's self-service API key so
// Playground requests follow the exact same grants, quotas, and accounting
// path as public API traffic. A user must create their key before Playground
// becomes available.
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

// ModelPatternsForKey resolves the model catalog visible to one key without
// exposing the access-group definitions themselves to a self-service user.
func (s *Store) ModelPatternsForKey(ctx context.Context, key APIKey) ([]string, error) {
	rows, err := s.pool.Query(ctx, `
SELECT DISTINCT b.subject_type,g.model_patterns FROM access_groups g
JOIN access_group_bindings b ON b.group_id=g.id
WHERE (b.subject_type='key' AND b.subject_id=$1)
   OR (b.subject_type='user' AND b.subject_id=$2)
   OR (b.subject_type='team' AND b.subject_id=$3)`, key.ID, key.UserID, key.TeamID)
	if err != nil {
		return nil, err
	}
	var keyPatterns, inheritedPatterns []string
	for rows.Next() {
		var subjectType string
		var raw []byte
		var patterns []string
		if scanErr := rows.Scan(&subjectType, &raw); scanErr != nil {
			rows.Close()
			return nil, scanErr
		}
		if decodeErr := json.Unmarshal(raw, &patterns); decodeErr != nil {
			rows.Close()
			return nil, decodeErr
		}
		if subjectType == "key" {
			keyPatterns = append(keyPatterns, patterns...)
		} else {
			inheritedPatterns = append(inheritedPatterns, patterns...)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return effectiveModelPatterns(keyPatterns, inheritedPatterns), nil
}
