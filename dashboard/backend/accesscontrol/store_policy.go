package accesscontrol

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
)

func (s *Store) ListAccessGroups(ctx context.Context, filter ListFilter) ([]AccessGroup, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_groups WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT g.id,g.name,g.description,g.model_patterns,g.created_at,g.updated_at,
       COALESCE(jsonb_agg(jsonb_build_object('subjectType',b.subject_type,'subjectId',b.subject_id))
         FILTER (WHERE b.subject_id IS NOT NULL),'[]'::jsonb)
FROM access_groups g LEFT JOIN access_group_bindings b ON b.group_id=g.id
WHERE ($1='' OR g.name ILIKE '%' || $1 || '%' OR g.description ILIKE '%' || $1 || '%')
GROUP BY g.id ORDER BY g.created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []AccessGroup{}
	for rows.Next() {
		var item AccessGroup
		var patterns, bindings []byte
		if err := rows.Scan(&item.ID, &item.Name, &item.Description, &patterns, &item.CreatedAt, &item.UpdatedAt, &bindings); err != nil {
			return nil, 0, err
		}
		if err := json.Unmarshal(patterns, &item.ModelPatterns); err != nil {
			return nil, 0, err
		}
		if err := json.Unmarshal(bindings, &item.Bindings); err != nil {
			return nil, 0, err
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetAccessGroup(ctx context.Context, id string) (AccessGroup, error) {
	var item AccessGroup
	var patterns, bindings []byte
	err := s.pool.QueryRow(ctx, `
SELECT g.id,g.name,g.description,g.model_patterns,g.created_at,g.updated_at,
       COALESCE(jsonb_agg(jsonb_build_object('subjectType',b.subject_type,'subjectId',b.subject_id))
         FILTER (WHERE b.subject_id IS NOT NULL),'[]'::jsonb)
FROM access_groups g LEFT JOIN access_group_bindings b ON b.group_id=g.id
WHERE g.id=$1 GROUP BY g.id`, id).
		Scan(&item.ID, &item.Name, &item.Description, &patterns, &item.CreatedAt, &item.UpdatedAt, &bindings)
	if err != nil {
		return AccessGroup{}, err
	}
	if err = json.Unmarshal(patterns, &item.ModelPatterns); err == nil {
		err = json.Unmarshal(bindings, &item.Bindings)
	}
	return item, err
}

func (s *Store) SaveAccessGroup(ctx context.Context, item AccessGroup) (AccessGroup, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return AccessGroup{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	patterns, _ := json.Marshal(uniqueStrings(item.ModelPatterns))
	err = tx.QueryRow(ctx, `
INSERT INTO access_groups(id,name,description,model_patterns) VALUES($1,$2,$3,$4)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description,model_patterns=EXCLUDED.model_patterns,updated_at=NOW()
RETURNING id,name,description,created_at,updated_at`, item.ID, item.Name, item.Description, patterns).
		Scan(&item.ID, &item.Name, &item.Description, &item.CreatedAt, &item.UpdatedAt)
	if err != nil {
		return AccessGroup{}, err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE group_id=$1`, item.ID); err != nil {
		return AccessGroup{}, err
	}
	for _, binding := range uniqueBindings(item.Bindings) {
		if _, err = tx.Exec(ctx, `INSERT INTO access_group_bindings(group_id,subject_type,subject_id) VALUES($1,$2,$3)`, item.ID, binding.SubjectType, binding.SubjectID); err != nil {
			return AccessGroup{}, err
		}
	}
	if err = tx.Commit(ctx); err != nil {
		return AccessGroup{}, err
	}
	item.ModelPatterns = uniqueStrings(item.ModelPatterns)
	item.Bindings = uniqueBindings(item.Bindings)
	return item, nil
}

func (s *Store) DeleteAccessGroup(ctx context.Context, id string) error {
	result, err := s.pool.Exec(ctx, `DELETE FROM access_groups WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}

func (s *Store) ListBudgets(ctx context.Context, filter ListFilter) ([]Budget, int64, error) {
	filter = normalizeFilter(filter)
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_budgets WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR scope_type ILIKE '%' || $1 || '%' OR scope_id ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, `
SELECT id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled,created_at,updated_at
FROM access_budgets
WHERE ($1='' OR name ILIKE '%' || $1 || '%' OR scope_type ILIKE '%' || $1 || '%' OR scope_id ILIKE '%' || $1 || '%')
ORDER BY created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []Budget{}
	for rows.Next() {
		var item Budget
		if err := rows.Scan(&item.ID, &item.Name, &item.ScopeType, &item.ScopeID, &item.RPM, &item.TPM, &item.DailyTokens, &item.Enabled, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, 0, err
		}
		items = append(items, item)
	}
	return items, total, rows.Err()
}

func (s *Store) GetBudget(ctx context.Context, id string) (Budget, error) {
	var item Budget
	err := s.pool.QueryRow(ctx, `SELECT id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled,created_at,updated_at FROM access_budgets WHERE id=$1`, id).
		Scan(&item.ID, &item.Name, &item.ScopeType, &item.ScopeID, &item.RPM, &item.TPM, &item.DailyTokens, &item.Enabled, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (s *Store) SaveBudget(ctx context.Context, item Budget) (Budget, error) {
	err := s.pool.QueryRow(ctx, `
INSERT INTO access_budgets(id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled)
VALUES($1,$2,$3,$4,$5,$6,$7,$8)
ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name,scope_type=EXCLUDED.scope_type,scope_id=EXCLUDED.scope_id,
 rpm=EXCLUDED.rpm,tpm=EXCLUDED.tpm,daily_tokens=EXCLUDED.daily_tokens,enabled=EXCLUDED.enabled,updated_at=NOW()
RETURNING id,name,scope_type,scope_id,rpm,tpm,daily_tokens,enabled,created_at,updated_at`,
		item.ID, item.Name, item.ScopeType, item.ScopeID, item.RPM, item.TPM, item.DailyTokens, item.Enabled).
		Scan(&item.ID, &item.Name, &item.ScopeType, &item.ScopeID, &item.RPM, &item.TPM, &item.DailyTokens, &item.Enabled, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (s *Store) DeleteBudget(ctx context.Context, id string) error {
	result, err := s.pool.Exec(ctx, `DELETE FROM access_budgets WHERE id=$1`, id)
	if err == nil && result.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	return err
}
