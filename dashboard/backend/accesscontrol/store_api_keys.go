package accesscontrol

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
)

const subjectRowsSQL = `
SELECT k.id,k.name,k.prefix,COALESCE(k.user_id,''),COALESCE(k.team_id,''),COALESCE(k.context_team_id,''),
 COALESCE(k.budget_id,''),k.status,k.expires_at,k.last_used_at,k.created_at,k.updated_at,
 COALESCE((SELECT jsonb_agg(b.group_id ORDER BY b.group_id) FROM access_group_bindings b WHERE b.subject_type='key' AND b.subject_id=k.id),'[]'::jsonb)
FROM access_api_keys k`

func (s *Store) ListAPIKeys(ctx context.Context, filter ListFilter) ([]APIKey, int64, error) {
	filter = normalizeFilter(filter)
	where := ` WHERE ($1='' OR k.name ILIKE '%' || $1 || '%' OR k.prefix ILIKE '%' || $1 || '%')
  AND ($2='' OR k.user_id=$2) AND ($3='' OR k.team_id=$3 OR k.context_team_id=$3) AND ($4='' OR k.id=$4)`
	var total int64
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM access_api_keys k`+where, filter.Query, filter.UserID, filter.TeamID, filter.KeyID).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.pool.Query(ctx, subjectRowsSQL+where+` ORDER BY k.created_at DESC LIMIT $5 OFFSET $6`, filter.Query, filter.UserID, filter.TeamID, filter.KeyID, filter.Limit, filter.Offset)
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

// ListAPIKeysForUser returns the caller's personal key plus Team-owned keys
// for Teams they administer. Membership alone never exposes a shared secret.
func (s *Store) ListAPIKeysForUser(ctx context.Context, userID string) ([]APIKey, error) {
	rows, err := s.pool.Query(ctx, subjectRowsSQL+`
WHERE k.user_id=$1 OR EXISTS(
  SELECT 1 FROM access_team_members membership
  WHERE membership.user_id=$1 AND membership.team_id=k.team_id AND membership.role='admin'
)
ORDER BY k.created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []APIKey{}
	for rows.Next() {
		item, scanErr := scanAPIKey(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) GetAPIKey(ctx context.Context, id string) (APIKey, error) {
	return scanAPIKey(s.pool.QueryRow(ctx, subjectRowsSQL+` WHERE k.id=$1`, id))
}

func scanAPIKey(row rowScanner) (APIKey, error) {
	var item APIKey
	var groupsJSON []byte
	if err := row.Scan(&item.ID, &item.Name, &item.Prefix, &item.UserID, &item.TeamID, &item.ContextTeamID, &item.BudgetID, &item.Status, &item.ExpiresAt, &item.LastUsed, &item.CreatedAt, &item.UpdatedAt, &groupsJSON); err != nil {
		return APIKey{}, err
	}
	if err := json.Unmarshal(groupsJSON, &item.AccessGroupIDs); err != nil {
		return APIKey{}, err
	}
	if item.UserID != "" {
		item.OwnerType, item.OwnerID = "user", item.UserID
	} else {
		item.OwnerType, item.OwnerID = "team", item.TeamID
		item.ContextTeamID = item.TeamID
	}
	return item, nil
}

func (s *Store) CreateAPIKey(ctx context.Context, item APIKey, digest, ciphertext string) (APIKey, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return APIKey{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err = tx.QueryRow(ctx, `
INSERT INTO access_api_keys(id,name,prefix,digest,secret_ciphertext,user_id,team_id,context_team_id,budget_id,status,expires_at)
VALUES($1,$2,$3,$4,$5,NULLIF($6,''),NULLIF($7,''),NULLIF($8,''),NULLIF($9,''),$10,$11)
RETURNING id`, item.ID, item.Name, item.Prefix, digest, ciphertext, item.UserID, item.TeamID, item.ContextTeamID, item.BudgetID, item.Status, item.ExpiresAt).Scan(&item.ID); err != nil {
		return APIKey{}, err
	}
	if err = replaceGroupBindings(ctx, tx, "key", item.ID, item.AccessGroupIDs); err != nil {
		return APIKey{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	return s.GetAPIKey(ctx, item.ID)
}

// CreateSelfAPIKey serializes by user across all Dashboard replicas so the
// one-key self-service rule cannot be bypassed by concurrent requests.
func (s *Store) CreateSelfAPIKey(ctx context.Context, item APIKey, digest, ciphertext string) (APIKey, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return APIKey{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err = tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, item.UserID); err != nil {
		return APIKey{}, err
	}
	var count int
	if err = tx.QueryRow(ctx, `SELECT COUNT(*) FROM access_api_keys WHERE user_id=$1`, item.UserID).Scan(&count); err != nil {
		return APIKey{}, err
	}
	if count > 0 {
		return APIKey{}, ErrSelfAPIKeyExists
	}
	if err = tx.QueryRow(ctx, `
INSERT INTO access_api_keys(id,name,prefix,digest,secret_ciphertext,user_id,context_team_id,status)
SELECT $1,$2,$3,$4,$5,u.id,NULLIF($7,''),'active' FROM access_users u
WHERE u.id=$6 AND u.status='active'
RETURNING id`, item.ID, item.Name, item.Prefix, digest, ciphertext, item.UserID, item.ContextTeamID).Scan(&item.ID); err != nil {
		return APIKey{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	return s.GetAPIKey(ctx, item.ID)
}

func (s *Store) APIKeyCiphertext(ctx context.Context, id string) (string, error) {
	var ciphertext string
	err := s.pool.QueryRow(ctx, `SELECT secret_ciphertext FROM access_api_keys WHERE id=$1`, id).Scan(&ciphertext)
	return ciphertext, err
}

func (s *Store) RotateAPIKeySecret(ctx context.Context, id, prefix, digest, ciphertext string) (APIKey, error) {
	result, err := s.pool.Exec(ctx, `UPDATE access_api_keys SET prefix=$2,digest=$3,secret_ciphertext=$4,status='active',updated_at=NOW() WHERE id=$1`, id, prefix, digest, ciphertext)
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
UPDATE access_api_keys SET name=$2,user_id=NULLIF($3,''),team_id=NULLIF($4,''),context_team_id=NULLIF($5,''),budget_id=NULLIF($6,''),status=$7,expires_at=$8,updated_at=NOW() WHERE id=$1`,
		item.ID, item.Name, item.UserID, item.TeamID, item.ContextTeamID, item.BudgetID, item.Status, item.ExpiresAt)
	if err != nil {
		return APIKey{}, err
	}
	if result.RowsAffected() == 0 {
		return APIKey{}, pgx.ErrNoRows
	}
	if err = replaceGroupBindings(ctx, tx, "key", item.ID, item.AccessGroupIDs); err != nil {
		return APIKey{}, err
	}
	if err = tx.Commit(ctx); err != nil {
		return APIKey{}, err
	}
	return s.GetAPIKey(ctx, item.ID)
}
