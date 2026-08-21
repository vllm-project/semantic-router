package accesscontrol

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (s *Store) ListAPIKeys(ctx context.Context, filter ListFilter) ([]APIKey, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_api_keys
WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR prefix ILIKE '%' || $1 || '%')
  AND ($2='' OR user_id=$2) AND ($3='' OR team_id=$3) AND ($4='' OR id=$4)`, filter.Query, filter.UserID, filter.TeamID, filter.KeyID).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT k.id,k.name,k.prefix,COALESCE(k.user_id,''),COALESCE(k.team_id,''),
 COALESCE(k.team_id,(SELECT m.team_id FROM access_team_members m JOIN access_teams t ON t.id=m.team_id WHERE m.user_id=k.user_id AND t.status='active' LIMIT 1),''),
 COALESCE(k.budget_id,''),k.status,k.expires_at,k.last_used_at,k.created_at,k.updated_at,
 COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='key' AND b.subject_id=k.id),'[]'::jsonb),
 COALESCE((SELECT jsonb_build_object('rpm',q.rpm,'tpm',q.tpm,'dailyTokens',q.daily_tokens) FROM access_budgets q WHERE q.scope_type='key' AND q.scope_id=k.id),'null'::jsonb)
FROM access_api_keys k
WHERE ($1='' OR k.name ILIKE '%' || $1 || '%' OR k.prefix ILIKE '%' || $1 || '%')
  AND ($2='' OR k.user_id=$2) AND ($3='' OR k.team_id=$3) AND ($4='' OR k.id=$4)
ORDER BY k.created_at DESC LIMIT $5 OFFSET $6`, filter.Query, filter.UserID, filter.TeamID, filter.KeyID, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []APIKey{}
	for rows.Next() {
		item, scanErr := scanAPIKey(rows)
		if scanErr != nil {
			return nil, 0, scanErr
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetAPIKey(ctx context.Context, id string) (APIKey, error) {
	return scanAPIKey(s.pool.QueryRow(ctx, `
SELECT k.id,k.name,k.prefix,COALESCE(k.user_id,''),COALESCE(k.team_id,''),
 COALESCE(k.team_id,(SELECT m.team_id FROM access_team_members m JOIN access_teams t ON t.id=m.team_id WHERE m.user_id=k.user_id AND t.status='active' LIMIT 1),''),
 COALESCE(k.budget_id,''),k.status,k.expires_at,k.last_used_at,k.created_at,k.updated_at,
 COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='key' AND b.subject_id=k.id),'[]'::jsonb),
 COALESCE((SELECT jsonb_build_object('rpm',q.rpm,'tpm',q.tpm,'dailyTokens',q.daily_tokens) FROM access_budgets q WHERE q.scope_type='key' AND q.scope_id=k.id),'null'::jsonb)
FROM access_api_keys k WHERE k.id=$1`, id))
}

func scanAPIKey(row rowScanner) (APIKey, error) {
	var item APIKey
	var groupJSON, budgetJSON []byte
	if err := row.Scan(&item.ID, &item.Name, &item.Prefix, &item.UserID, &item.TeamID, &item.EffectiveTeamID, &item.BudgetID, &item.Status, &item.ExpiresAt, &item.LastUsed, &item.CreatedAt, &item.UpdatedAt, &groupJSON, &budgetJSON); err != nil {
		return APIKey{}, err
	}
	if err := json.Unmarshal(groupJSON, &item.AccessGroupIDs); err != nil {
		return APIKey{}, err
	}
	if string(budgetJSON) != "null" {
		item.Budget = &KeyBudget{}
		if err := json.Unmarshal(budgetJSON, item.Budget); err != nil {
			return APIKey{}, err
		}
	}
	return item, nil
}

func (s *Store) CreateAPIKey(ctx context.Context, item APIKey, digest, ciphertext string) (APIKey, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return APIKey{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	err = tx.QueryRow(ctx, `
	INSERT INTO access_api_keys(id,name,prefix,digest,secret_ciphertext,user_id,team_id,budget_id,status,expires_at)
	VALUES($1,$2,$3,$4,$5,NULLIF($6,''),NULLIF($7,''),NULLIF($8,''),$9,$10)
	RETURNING id,name,prefix,COALESCE(user_id,''),COALESCE(team_id,''),COALESCE(budget_id,''),status,expires_at,last_used_at,created_at,updated_at`,
		item.ID, item.Name, item.Prefix, digest, ciphertext, item.UserID, item.TeamID, item.BudgetID, item.Status, item.ExpiresAt).
		Scan(&item.ID, &item.Name, &item.Prefix, &item.UserID, &item.TeamID, &item.BudgetID, &item.Status, &item.ExpiresAt, &item.LastUsed, &item.CreatedAt, &item.UpdatedAt)
	if err != nil {
		return APIKey{}, err
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	for _, groupID := range item.AccessGroupIDs {
		result, bindErr := tx.Exec(ctx, `INSERT INTO access_group_bindings(group_id,subject_type,subject_id) VALUES($1,'key',$2) ON CONFLICT DO NOTHING`, groupID, item.ID)
		if bindErr != nil {
			return APIKey{}, bindErr
		}
		if result.RowsAffected() == 0 {
			var exists bool
			if checkErr := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM access_groups WHERE id=$1)`, groupID).Scan(&exists); checkErr != nil || !exists {
				return APIKey{}, fmt.Errorf("access group %s does not exist", groupID)
			}
		}
	}
	if item.Budget != nil {
		_, err = tx.Exec(ctx, `
INSERT INTO access_budgets(id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled)
VALUES($1,$2,'key',$3,$4,$5,$6,TRUE)`, "key-budget-"+item.ID, item.Name+" key limits", item.ID,
			item.Budget.RPM, item.Budget.TPM, item.Budget.DailyTokens)
		if err != nil {
			return APIKey{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	return item, nil
}

// CreateSelfAPIKey serializes by user across all Dashboard replicas so the
// one-key self-service rule cannot be bypassed by concurrent requests.
func (s *Store) CreateSelfAPIKey(ctx context.Context, item APIKey, digest, ciphertext string) (APIKey, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return APIKey{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, lockErr := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, item.UserID); lockErr != nil {
		return APIKey{}, lockErr
	}
	var count int
	if countErr := tx.QueryRow(ctx, `SELECT COUNT(*) FROM access_api_keys WHERE user_id=$1`, item.UserID).Scan(&count); countErr != nil {
		return APIKey{}, countErr
	}
	if count > 0 {
		return APIKey{}, ErrSelfAPIKeyExists
	}
	err = tx.QueryRow(ctx, `
INSERT INTO access_api_keys(id,name,prefix,digest,secret_ciphertext,user_id,status)
SELECT $1,$2,$3,$4,$5,u.id,'active' FROM access_users u
WHERE u.id=$6 AND u.status='active'
RETURNING id,name,prefix,COALESCE(user_id,''),COALESCE(team_id,''),COALESCE(budget_id,''),status,expires_at,last_used_at,created_at,updated_at`,
		item.ID, item.Name, item.Prefix, digest, ciphertext, item.UserID).
		Scan(&item.ID, &item.Name, &item.Prefix, &item.UserID, &item.TeamID, &item.BudgetID, &item.Status, &item.ExpiresAt, &item.LastUsed, &item.CreatedAt, &item.UpdatedAt)
	if err != nil {
		return APIKey{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	item.AccessGroupIDs = []string{}
	return item, nil
}

func (s *Store) APIKeyCiphertext(ctx context.Context, id string) (string, error) {
	var ciphertext string
	err := s.pool.QueryRow(ctx, `SELECT secret_ciphertext FROM access_api_keys WHERE id=$1`, id).Scan(&ciphertext)
	return ciphertext, err
}

func (s *Store) RotateAPIKeySecret(ctx context.Context, id, prefix, digest, ciphertext string) (APIKey, error) {
	result, err := s.pool.Exec(ctx, `
UPDATE access_api_keys SET prefix=$2,digest=$3,secret_ciphertext=$4,status='active',updated_at=NOW() WHERE id=$1`,
		id, prefix, digest, ciphertext)
	if err != nil {
		return APIKey{}, err
	}
	if result.RowsAffected() == 0 {
		return APIKey{}, pgx.ErrNoRows
	}
	return s.GetAPIKey(ctx, id)
}

func (s *Store) SetAPIKeyStatus(ctx context.Context, id, status string) (APIKey, error) {
	result, err := s.pool.Exec(ctx, `UPDATE access_api_keys SET status=$2,updated_at=NOW() WHERE id=$1`, id, status)
	if err != nil {
		return APIKey{}, err
	}
	if result.RowsAffected() == 0 {
		return APIKey{}, pgx.ErrNoRows
	}
	return s.GetAPIKey(ctx, id)
}

func (s *Store) UpdateAPIKey(ctx context.Context, item APIKey) (APIKey, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return APIKey{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	result, err := tx.Exec(ctx, `
UPDATE access_api_keys SET name=$2,user_id=NULLIF($3,''),team_id=NULLIF($4,''),budget_id=NULLIF($5,''),
 status=$6,expires_at=$7,updated_at=NOW() WHERE id=$1`,
		item.ID, item.Name, item.UserID, item.TeamID, item.BudgetID, item.Status, item.ExpiresAt)
	if err != nil {
		return APIKey{}, err
	}
	if result.RowsAffected() == 0 {
		return APIKey{}, pgx.ErrNoRows
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type='key' AND subject_id=$1`, item.ID); err != nil {
		return APIKey{}, err
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	for _, groupID := range item.AccessGroupIDs {
		if _, err = tx.Exec(ctx, `INSERT INTO access_group_bindings(group_id,subject_type,subject_id) VALUES($1,'key',$2)`, groupID, item.ID); err != nil {
			return APIKey{}, err
		}
	}
	directBudgetID := "key-budget-" + item.ID
	if item.Budget == nil {
		if _, err = tx.Exec(ctx, `DELETE FROM access_budgets WHERE id=$1`, directBudgetID); err != nil {
			return APIKey{}, err
		}
	} else {
		_, err = tx.Exec(ctx, `
INSERT INTO access_budgets(id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled)
VALUES($1,$2,'key',$3,$4,$5,$6,TRUE)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,rpm=EXCLUDED.rpm,tpm=EXCLUDED.tpm,
 daily_tokens=EXCLUDED.daily_tokens,enabled=TRUE,updated_at=NOW()`, directBudgetID, item.Name+" key limits", item.ID,
			item.Budget.RPM, item.Budget.TPM, item.Budget.DailyTokens)
		if err != nil {
			return APIKey{}, err
		}
	}
	if err = tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	return s.GetAPIKey(ctx, item.ID)
}
