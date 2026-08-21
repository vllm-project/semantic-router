package accesscontrol

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

func (s *Store) ListUsers(ctx context.Context, filter ListFilter) ([]User, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_users WHERE ($1='' OR email ILIKE '%' || $1 || '%' OR name ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT id,email,name,status,created_at,updated_at FROM access_users
WHERE ($1='' OR email ILIKE '%' || $1 || '%' OR name ILIKE '%' || $1 || '%')
ORDER BY created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []User{}
	for rows.Next() {
		var item User
		if err := rows.Scan(&item.ID, &item.Email, &item.Name, &item.Status, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, 0, err
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetUser(ctx context.Context, id string) (User, error) {
	var item User
	err := s.pool.QueryRow(ctx, `SELECT id,email,name,status,created_at,updated_at FROM access_users WHERE id=$1`, id).
		Scan(&item.ID, &item.Email, &item.Name, &item.Status, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (s *Store) SaveUser(ctx context.Context, item User) (User, error) {
	err := s.pool.QueryRow(ctx, `
INSERT INTO access_users(id,email,name,status) VALUES($1,$2,$3,$4)
ON CONFLICT(id) DO UPDATE SET email=EXCLUDED.email,name=EXCLUDED.name,status=EXCLUDED.status,updated_at=NOW()
RETURNING id,email,name,status,created_at,updated_at`, item.ID, item.Email, item.Name, item.Status).
		Scan(&item.ID, &item.Email, &item.Name, &item.Status, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (s *Store) DeleteUser(ctx context.Context, id string) error {
	result, err := s.pool.Exec(ctx, `DELETE FROM access_users WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}

// SetUserTeam makes invitation-time assignment deterministic. A user belongs
// to one Team so inherited grants and quota always resolve unambiguously.
func (s *Store) SetUserTeam(ctx context.Context, userID, teamID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = lockTeamPolicy(ctx, tx); err != nil {
		return err
	}

	var userExists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM access_users WHERE id=$1)`, userID).Scan(&userExists); err != nil {
		return err
	}
	if !userExists {
		return pgx.ErrNoRows
	}
	if _, err := tx.Exec(ctx, `DELETE FROM access_team_members WHERE user_id=$1`, userID); err != nil {
		return err
	}
	if teamID != "" {
		result, insertErr := tx.Exec(ctx, `
INSERT INTO access_team_members(team_id,user_id)
SELECT id,$2 FROM access_teams WHERE id=$1 AND status='active'`, teamID, userID)
		if insertErr != nil {
			return insertErr
		}
		if result.RowsAffected() == 0 {
			return pgx.ErrNoRows
		}
	}
	return tx.Commit(ctx)
}

func (s *Store) ListTeamsForUser(ctx context.Context, userID string) ([]Team, error) {
	rows, err := s.pool.Query(ctx, `
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at,
       jsonb_build_array($1::text),
       COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='team' AND b.subject_id=t.id),'[]'::jsonb),
       COALESCE((SELECT jsonb_build_object('rpm',q.rpm,'tpm',q.tpm,'dailyTokens',q.daily_tokens) FROM access_budgets q WHERE q.scope_type='team' AND q.scope_id=t.id),'null'::jsonb)
FROM access_teams t JOIN access_team_members m ON m.team_id=t.id
WHERE m.user_id=$1 ORDER BY t.created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []Team{}
	for rows.Next() {
		item, scanErr := scanTeam(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) ListTeams(ctx context.Context, filter ListFilter) ([]Team, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_teams WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at,
       COALESCE((SELECT jsonb_agg(m.user_id ORDER BY m.user_id) FROM access_team_members m WHERE m.team_id=t.id),'[]'::jsonb),
       COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='team' AND b.subject_id=t.id),'[]'::jsonb),
       COALESCE((SELECT jsonb_build_object('rpm',q.rpm,'tpm',q.tpm,'dailyTokens',q.daily_tokens) FROM access_budgets q WHERE q.scope_type='team' AND q.scope_id=t.id),'null'::jsonb)
FROM access_teams t
WHERE ($1='' OR t.name ILIKE '%' || $1 || '%' OR t.description ILIKE '%' || $1 || '%')
ORDER BY t.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []Team{}
	for rows.Next() {
		item, scanErr := scanTeam(rows)
		if scanErr != nil {
			return nil, 0, scanErr
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetTeam(ctx context.Context, id string) (Team, error) {
	return scanTeam(s.pool.QueryRow(ctx, `
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at,
       COALESCE((SELECT jsonb_agg(m.user_id ORDER BY m.user_id) FROM access_team_members m WHERE m.team_id=t.id),'[]'::jsonb),
       COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='team' AND b.subject_id=t.id),'[]'::jsonb),
       COALESCE((SELECT jsonb_build_object('rpm',q.rpm,'tpm',q.tpm,'dailyTokens',q.daily_tokens) FROM access_budgets q WHERE q.scope_type='team' AND q.scope_id=t.id),'null'::jsonb)
FROM access_teams t WHERE t.id=$1`, id))
}

func scanTeam(row rowScanner) (Team, error) {
	var item Team
	var membersJSON, groupsJSON, budgetJSON []byte
	if err := row.Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.CreatedAt, &item.UpdatedAt, &membersJSON, &groupsJSON, &budgetJSON); err != nil {
		return Team{}, err
	}
	if err := json.Unmarshal(membersJSON, &item.UserIDs); err != nil {
		return Team{}, err
	}
	if err := json.Unmarshal(groupsJSON, &item.AccessGroupIDs); err != nil {
		return Team{}, err
	}
	if string(budgetJSON) != "null" {
		item.Budget = &KeyBudget{}
		if err := json.Unmarshal(budgetJSON, item.Budget); err != nil {
			return Team{}, err
		}
	}
	return item, nil
}

func (s *Store) SaveTeam(ctx context.Context, item Team) (Team, error) {
	if item.Budget == nil {
		return Team{}, errors.New("team budget is required")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return Team{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = lockTeamPolicy(ctx, tx); err != nil {
		return Team{}, err
	}
	err = tx.QueryRow(ctx, `
INSERT INTO access_teams(id,name,description,status) VALUES($1,$2,$3,$4)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description,status=EXCLUDED.status,updated_at=NOW()
RETURNING id,name,description,status,created_at,updated_at`, item.ID, item.Name, item.Description, item.Status).
		Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.CreatedAt, &item.UpdatedAt)
	if err != nil {
		return Team{}, err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_team_members WHERE team_id=$1`, item.ID); err != nil {
		return Team{}, err
	}
	for _, userID := range uniqueStrings(item.UserIDs) {
		if _, err = tx.Exec(ctx, `DELETE FROM access_team_members WHERE user_id=$1 AND team_id<>$2`, userID, item.ID); err != nil {
			return Team{}, err
		}
		if _, err = tx.Exec(ctx, `INSERT INTO access_team_members(team_id,user_id) VALUES($1,$2)`, item.ID, userID); err != nil {
			return Team{}, err
		}
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type='team' AND subject_id=$1`, item.ID); err != nil {
		return Team{}, err
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	for _, groupID := range item.AccessGroupIDs {
		result, bindErr := tx.Exec(ctx, `
INSERT INTO access_group_bindings(group_id,subject_type,subject_id)
SELECT id,'team',$2 FROM access_groups WHERE id=$1`, groupID, item.ID)
		if bindErr != nil {
			return Team{}, bindErr
		}
		if result.RowsAffected() == 0 {
			return Team{}, fmt.Errorf("access group %s does not exist", groupID)
		}
	}
	_, err = tx.Exec(ctx, `
INSERT INTO access_budgets(id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled)
VALUES($1,$2,'team',$3,$4,$5,$6,TRUE)
ON CONFLICT(scope_type,scope_id) DO UPDATE SET name=EXCLUDED.name,rpm=EXCLUDED.rpm,tpm=EXCLUDED.tpm,
 daily_tokens=EXCLUDED.daily_tokens,enabled=TRUE,updated_at=NOW()`, "team-budget-"+item.ID, item.Name+" team limits", item.ID,
		item.Budget.RPM, item.Budget.TPM, item.Budget.DailyTokens)
	if err != nil {
		return Team{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return Team{}, err
	}
	item.UserIDs = uniqueStrings(item.UserIDs)
	return item, nil
}

func (s *Store) DeleteTeam(ctx context.Context, id string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = lockTeamPolicy(ctx, tx); err != nil {
		return err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type='team' AND subject_id=$1`, id); err != nil {
		return err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_budgets WHERE scope_type='team' AND scope_id=$1`, id); err != nil {
		return err
	}
	result, err := tx.Exec(ctx, `DELETE FROM access_teams WHERE id=$1`, id)
	if err != nil {
		return err
	}
	if result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return tx.Commit(ctx)
}
