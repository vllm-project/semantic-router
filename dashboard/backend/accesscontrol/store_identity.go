package accesscontrol

import (
	"context"
	"encoding/json"

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

// SetUserTeam makes the invitation-time Team assignment deterministic. The
// product currently exposes one primary Team per user, while the relational
// schema remains ready for broader membership in the future.
func (s *Store) SetUserTeam(ctx context.Context, userID, teamID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()

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
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at
FROM access_teams t JOIN access_team_members m ON m.team_id=t.id
WHERE m.user_id=$1 ORDER BY t.created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []Team{}
	for rows.Next() {
		var item Team
		if err := rows.Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		item.UserIDs = []string{userID}
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
       COALESCE(jsonb_agg(m.user_id) FILTER (WHERE m.user_id IS NOT NULL),'[]'::jsonb)
FROM access_teams t LEFT JOIN access_team_members m ON m.team_id=t.id
WHERE ($1='' OR t.name ILIKE '%' || $1 || '%' OR t.description ILIKE '%' || $1 || '%')
GROUP BY t.id ORDER BY t.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []Team{}
	for rows.Next() {
		var item Team
		var memberJSON []byte
		if err := rows.Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.CreatedAt, &item.UpdatedAt, &memberJSON); err != nil {
			return nil, 0, err
		}
		if err := json.Unmarshal(memberJSON, &item.UserIDs); err != nil {
			return nil, 0, err
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetTeam(ctx context.Context, id string) (Team, error) {
	var item Team
	var memberJSON []byte
	err := s.pool.QueryRow(ctx, `
SELECT t.id,t.name,t.description,t.status,t.created_at,t.updated_at,
       COALESCE(jsonb_agg(m.user_id) FILTER (WHERE m.user_id IS NOT NULL),'[]'::jsonb)
FROM access_teams t LEFT JOIN access_team_members m ON m.team_id=t.id
WHERE t.id=$1 GROUP BY t.id`, id).
		Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.CreatedAt, &item.UpdatedAt, &memberJSON)
	if err == nil {
		err = json.Unmarshal(memberJSON, &item.UserIDs)
	}
	return item, err
}

func (s *Store) SaveTeam(ctx context.Context, item Team) (Team, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return Team{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
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
		if _, err = tx.Exec(ctx, `INSERT INTO access_team_members(team_id,user_id) VALUES($1,$2)`, item.ID, userID); err != nil {
			return Team{}, err
		}
	}
	if err = tx.Commit(ctx); err != nil {
		return Team{}, err
	}
	item.UserIDs = uniqueStrings(item.UserIDs)
	return item, nil
}

func (s *Store) DeleteTeam(ctx context.Context, id string) error {
	result, err := s.pool.Exec(ctx, `DELETE FROM access_teams WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}
