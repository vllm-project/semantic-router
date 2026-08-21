package accesscontrol

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
)

const userSelect = `
SELECT u.id,u.email,u.name,u.status,COALESCE(u.budget_id,''),u.created_at,u.updated_at,
 COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='user' AND b.subject_id=u.id),'[]'::jsonb),
 COALESCE((SELECT jsonb_agg(jsonb_build_object('teamId',m.team_id,'userId',m.user_id,'role',m.role) ORDER BY m.created_at) FROM access_team_members m WHERE m.user_id=u.id),'[]'::jsonb)
FROM access_users u`

const teamSelect = `
SELECT t.id,t.name,t.description,t.status,t.budget_id,t.created_at,t.updated_at,
 COALESCE((SELECT jsonb_agg(jsonb_build_object('teamId',m.team_id,'userId',m.user_id,'role',m.role) ORDER BY m.created_at) FROM access_team_members m WHERE m.team_id=t.id),'[]'::jsonb),
 COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='team' AND b.subject_id=t.id),'[]'::jsonb)
FROM access_teams t`

func (s *Store) ListUsers(ctx context.Context, filter ListFilter) ([]User, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_users WHERE ($1='' OR email ILIKE '%' || $1 || '%' OR name ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, userSelect+`
WHERE ($1='' OR u.email ILIKE '%' || $1 || '%' OR u.name ILIKE '%' || $1 || '%')
ORDER BY u.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []User{}
	for rows.Next() {
		item, scanErr := scanUser(rows)
		if scanErr != nil {
			return nil, 0, scanErr
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetUser(ctx context.Context, id string) (User, error) {
	return scanUser(s.pool.QueryRow(ctx, userSelect+` WHERE u.id=$1`, id))
}

func scanUser(row rowScanner) (User, error) {
	var item User
	var groupsJSON, membershipsJSON []byte
	if err := row.Scan(&item.ID, &item.Email, &item.Name, &item.Status, &item.BudgetID, &item.CreatedAt, &item.UpdatedAt, &groupsJSON, &membershipsJSON); err != nil {
		return User{}, err
	}
	if err := json.Unmarshal(groupsJSON, &item.AccessGroupIDs); err != nil {
		return User{}, err
	}
	if err := json.Unmarshal(membershipsJSON, &item.Memberships); err != nil {
		return User{}, err
	}
	return item, nil
}

func (s *Store) SaveUser(ctx context.Context, item User) (User, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return User{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = tx.QueryRow(ctx, `
INSERT INTO access_users(id,email,name,status,budget_id) VALUES($1,$2,$3,$4,NULLIF($5,''))
ON CONFLICT(id) DO UPDATE SET email=EXCLUDED.email,name=EXCLUDED.name,status=EXCLUDED.status,budget_id=EXCLUDED.budget_id,updated_at=NOW()
RETURNING id`, item.ID, item.Email, item.Name, item.Status, item.BudgetID).Scan(&item.ID); err != nil {
		return User{}, err
	}
	if err = replaceGroupBindings(ctx, tx, "user", item.ID, item.AccessGroupIDs); err != nil {
		return User{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return User{}, err
	}
	return s.GetUser(ctx, item.ID)
}

func (s *Store) DeleteUser(ctx context.Context, id string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type='user' AND subject_id=$1`, id); err != nil {
		return err
	}
	result, err := tx.Exec(ctx, `DELETE FROM access_users WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// SetUserTeamMembership adds or updates one explicit Team membership. Users
// may join multiple Teams; the API key chooses which Team provides context.
func (s *Store) SetUserTeamMembership(ctx context.Context, userID, teamID, role string) error {
	if teamID == "" {
		return nil
	}
	result, err := s.pool.Exec(ctx, `
INSERT INTO access_team_members(team_id,user_id,role)
SELECT t.id,u.id,$3 FROM access_teams t CROSS JOIN access_users u
WHERE t.id=$1 AND t.status='active' AND u.id=$2 AND u.status='active'
ON CONFLICT(team_id,user_id) DO UPDATE SET role=EXCLUDED.role`, teamID, userID, role)
	if err != nil {
		return err
	}
	if result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return nil
}

func (s *Store) ListTeamsForUser(ctx context.Context, userID string) ([]Team, error) {
	rows, err := s.pool.Query(ctx, teamSelect+`
JOIN access_team_members mine ON mine.team_id=t.id
WHERE mine.user_id=$1 ORDER BY t.created_at DESC`, userID)
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

func (s *Store) ListTeamIDsForUser(ctx context.Context, userID string) ([]string, error) {
	rows, err := s.pool.Query(ctx, `
SELECT team_id FROM access_team_members WHERE user_id=$1 ORDER BY team_id`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	teamIDs := make([]string, 0)
	for rows.Next() {
		var teamID string
		if err = rows.Scan(&teamID); err != nil {
			return nil, err
		}
		teamIDs = append(teamIDs, teamID)
	}
	return teamIDs, rows.Err()
}

func (s *Store) IsTeamAdmin(ctx context.Context, userID, teamID string) (bool, error) {
	var allowed bool
	err := s.pool.QueryRow(ctx, `
SELECT EXISTS(
  SELECT 1 FROM access_team_members
  WHERE user_id=$1 AND team_id=$2 AND role='admin'
)`, userID, teamID).Scan(&allowed)
	return allowed, err
}

func (s *Store) GetTeamForUser(ctx context.Context, userID, teamID string) (Team, error) {
	return scanTeam(s.pool.QueryRow(ctx, teamSelect+`
WHERE t.id=$2 AND EXISTS(
  SELECT 1 FROM access_team_members mine WHERE mine.team_id=t.id AND mine.user_id=$1
)`, userID, teamID))
}

// ListUsersSharingTeam returns only identities visible through the caller's
// Team memberships. Personal grants and budgets stay private.
func (s *Store) ListUsersSharingTeam(ctx context.Context, userID string) ([]TeamMemberIdentity, error) {
	rows, err := s.pool.Query(ctx, `
SELECT DISTINCT u.id,u.email,u.name,u.status
FROM access_users u
JOIN access_team_members peer ON peer.user_id=u.id
JOIN access_team_members mine ON mine.team_id=peer.team_id
WHERE mine.user_id=$1
ORDER BY u.name,u.email`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []TeamMemberIdentity{}
	for rows.Next() {
		var item TeamMemberIdentity
		if err = rows.Scan(&item.ID, &item.Email, &item.Name, &item.Status); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) ListPoliciesForUserTeams(ctx context.Context, userID string) ([]AccessGroup, []Budget, error) {
	groupRows, err := s.pool.Query(ctx, accessGroupSelect+`
WHERE g.id IN (
  SELECT binding.group_id FROM access_group_bindings binding
  JOIN access_team_members mine ON mine.team_id=binding.subject_id
  WHERE binding.subject_type='team' AND mine.user_id=$1
)
GROUP BY g.id ORDER BY g.name`, userID)
	if err != nil {
		return nil, nil, err
	}
	groups := []AccessGroup{}
	for groupRows.Next() {
		item, scanErr := scanAccessGroup(groupRows)
		if scanErr != nil {
			groupRows.Close()
			return nil, nil, scanErr
		}
		groups = append(groups, item)
	}
	if err = groupRows.Err(); err != nil {
		groupRows.Close()
		return nil, nil, err
	}
	groupRows.Close()

	budgetRows, err := s.pool.Query(ctx, budgetSelect+`
WHERE q.id IN (
  SELECT t.budget_id FROM access_teams t
  JOIN access_team_members mine ON mine.team_id=t.id
  WHERE mine.user_id=$1
)
ORDER BY q.name`, userID)
	if err != nil {
		return nil, nil, err
	}
	defer budgetRows.Close()
	budgets := []Budget{}
	for budgetRows.Next() {
		item, scanErr := scanBudget(budgetRows)
		if scanErr != nil {
			return nil, nil, scanErr
		}
		budgets = append(budgets, item)
	}
	return groups, budgets, budgetRows.Err()
}

func (s *Store) ListTeams(ctx context.Context, filter ListFilter) ([]Team, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_teams WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, teamSelect+`
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
	return scanTeam(s.pool.QueryRow(ctx, teamSelect+` WHERE t.id=$1`, id))
}

func scanTeam(row rowScanner) (Team, error) {
	var item Team
	var membersJSON, groupsJSON []byte
	if err := row.Scan(&item.ID, &item.Name, &item.Description, &item.Status, &item.BudgetID, &item.CreatedAt, &item.UpdatedAt, &membersJSON, &groupsJSON); err != nil {
		return Team{}, err
	}
	if err := json.Unmarshal(membersJSON, &item.Members); err != nil {
		return Team{}, err
	}
	if err := json.Unmarshal(groupsJSON, &item.AccessGroupIDs); err != nil {
		return Team{}, err
	}
	return item, nil
}

func (s *Store) SaveTeam(ctx context.Context, item Team) (Team, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return Team{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = lockTeamPolicy(ctx, tx); err != nil {
		return Team{}, err
	}
	if err = tx.QueryRow(ctx, `
INSERT INTO access_teams(id,name,description,status,budget_id) VALUES($1,$2,$3,$4,$5)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description,status=EXCLUDED.status,budget_id=EXCLUDED.budget_id,updated_at=NOW()
RETURNING id`, item.ID, item.Name, item.Description, item.Status, item.BudgetID).Scan(&item.ID); err != nil {
		return Team{}, err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_team_members WHERE team_id=$1`, item.ID); err != nil {
		return Team{}, err
	}
	seenMembers := map[string]struct{}{}
	for _, member := range item.Members {
		if _, exists := seenMembers[member.UserID]; exists || member.UserID == "" {
			continue
		}
		seenMembers[member.UserID] = struct{}{}
		if member.Role != TeamRoleAdmin && member.Role != TeamRoleMember {
			return Team{}, validationErrorf("invalid Team role for user %s", member.UserID)
		}
		result, insertErr := tx.Exec(ctx, `INSERT INTO access_team_members(team_id,user_id,role) SELECT $1,id,$3 FROM access_users WHERE id=$2`, item.ID, member.UserID, member.Role)
		if insertErr != nil {
			return Team{}, insertErr
		}
		if result.RowsAffected() == 0 {
			return Team{}, validationErrorf("user %s does not exist", member.UserID)
		}
	}
	if err = replaceGroupBindings(ctx, tx, "team", item.ID, item.AccessGroupIDs); err != nil {
		return Team{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return Team{}, err
	}
	return s.GetTeam(ctx, item.ID)
}

func (s *Store) DeleteTeam(ctx context.Context, id string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type='team' AND subject_id=$1`, id); err != nil {
		return err
	}
	result, err := tx.Exec(ctx, `DELETE FROM access_teams WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}
