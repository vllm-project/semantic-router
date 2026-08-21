package accesscontrol

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
)

const accessGroupSelect = `
SELECT g.id,g.name,g.description,g.model_patterns,g.created_at,g.updated_at,COUNT(b.subject_id)
FROM access_groups g LEFT JOIN access_group_bindings b ON b.group_id=g.id`

func (s *Store) ListAccessGroups(ctx context.Context, filter ListFilter) ([]AccessGroup, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_groups WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, accessGroupSelect+`
WHERE ($1='' OR g.name ILIKE '%' || $1 || '%' OR g.description ILIKE '%' || $1 || '%')
GROUP BY g.id ORDER BY g.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []AccessGroup{}
	for rows.Next() {
		item, scanErr := scanAccessGroup(rows)
		if scanErr != nil {
			return nil, 0, scanErr
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetAccessGroup(ctx context.Context, id string) (AccessGroup, error) {
	return scanAccessGroup(s.pool.QueryRow(ctx, accessGroupSelect+` WHERE g.id=$1 GROUP BY g.id`, id))
}

func scanAccessGroup(row rowScanner) (AccessGroup, error) {
	var item AccessGroup
	var patterns []byte
	if err := row.Scan(&item.ID, &item.Name, &item.Description, &patterns, &item.CreatedAt, &item.UpdatedAt, &item.AssignmentCount); err != nil {
		return AccessGroup{}, err
	}
	if err := json.Unmarshal(patterns, &item.ModelPatterns); err != nil {
		return AccessGroup{}, err
	}
	return item, nil
}

func (s *Store) SaveAccessGroup(ctx context.Context, item AccessGroup) (AccessGroup, error) {
	patterns, _ := json.Marshal(uniqueStrings(item.ModelPatterns))
	if err := s.pool.QueryRow(ctx, `
INSERT INTO access_groups(id,name,description,model_patterns) VALUES($1,$2,$3,$4)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description,model_patterns=EXCLUDED.model_patterns,updated_at=NOW()
RETURNING id`, item.ID, item.Name, item.Description, patterns).Scan(&item.ID); err != nil {
		return AccessGroup{}, err
	}
	return s.GetAccessGroup(ctx, item.ID)
}

func (s *Store) DeleteAccessGroup(ctx context.Context, id string) error {
	var assignments int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_group_bindings WHERE group_id=$1`, id).Scan(&assignments); err != nil {
		return err
	}
	if assignments > 0 {
		return validationError("remove this access group from its assignments before deleting it")
	}
	result, err := s.pool.Exec(ctx, `DELETE FROM access_groups WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}

const budgetSelect = `
SELECT q.id,q.name,q.description,q.rpm,q.tpm,q.daily_tokens,q.enabled,q.created_at,q.updated_at,
 ((SELECT COUNT(*) FROM access_users u WHERE u.budget_id=q.id) +
  (SELECT COUNT(*) FROM access_teams t WHERE t.budget_id=q.id) +
  (SELECT COUNT(*) FROM access_api_keys k WHERE k.budget_id=q.id)) AS assignments
FROM access_budgets q`

func (s *Store) ListBudgets(ctx context.Context, filter ListFilter) ([]Budget, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_budgets WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, budgetSelect+`
WHERE ($1='' OR q.name ILIKE '%' || $1 || '%' OR q.description ILIKE '%' || $1 || '%')
ORDER BY q.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []Budget{}
	for rows.Next() {
		item, scanErr := scanBudget(rows)
		if scanErr != nil {
			return nil, 0, scanErr
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetBudget(ctx context.Context, id string) (Budget, error) {
	return scanBudget(s.pool.QueryRow(ctx, budgetSelect+` WHERE q.id=$1`, id))
}

func scanBudget(row rowScanner) (Budget, error) {
	var item Budget
	err := row.Scan(&item.ID, &item.Name, &item.Description, &item.RPM, &item.TPM, &item.DailyTokens, &item.Enabled, &item.CreatedAt, &item.UpdatedAt, &item.AssignmentCount)
	return item, err
}

func (s *Store) SaveBudget(ctx context.Context, item Budget) (Budget, error) {
	if err := s.pool.QueryRow(ctx, `
INSERT INTO access_budgets(id,name,description,rpm,tpm,daily_tokens,enabled)
VALUES($1,$2,$3,$4,$5,$6,$7)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description,rpm=EXCLUDED.rpm,tpm=EXCLUDED.tpm,daily_tokens=EXCLUDED.daily_tokens,enabled=EXCLUDED.enabled,updated_at=NOW()
RETURNING id`, item.ID, item.Name, item.Description, item.RPM, item.TPM, item.DailyTokens, item.Enabled).Scan(&item.ID); err != nil {
		return Budget{}, err
	}
	return s.GetBudget(ctx, item.ID)
}

func (s *Store) DeleteBudget(ctx context.Context, id string) error {
	budget, err := s.GetBudget(ctx, id)
	if err != nil {
		return err
	}
	if budget.AssignmentCount > 0 {
		return validationError("remove this budget from its assignments before deleting it")
	}
	result, err := s.pool.Exec(ctx, `DELETE FROM access_budgets WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}
