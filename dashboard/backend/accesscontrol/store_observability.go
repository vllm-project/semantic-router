package accesscontrol

import (
	"context"
	"encoding/json"
)

func (s *Store) InsertAudit(ctx context.Context, event AuditEvent) error {
	details, _ := json.Marshal(event.Details)
	_, err := s.pool.Exec(ctx, `
INSERT INTO access_audit_events(id,actor_id,actor_email,action,resource_type,resource_id,details,created_at)
VALUES($1,$2,$3,$4,$5,$6,$7,$8)`, event.ID, event.ActorID, event.ActorEmail, event.Action,
		event.ResourceType, event.ResourceID, details, event.CreatedAt)
	return err
}

func (s *Store) ListAudit(ctx context.Context, filter ListFilter) ([]AuditEvent, error) {
	filter = normalizeFilter(filter)
	rows, err := s.pool.Query(ctx, `
SELECT id,actor_id,actor_email,action,resource_type,resource_id,details,created_at
FROM access_audit_events
WHERE ($1='' OR actor_email ILIKE '%' || $1 || '%' OR action ILIKE '%' || $1 || '%' OR resource_type ILIKE '%' || $1 || '%')
ORDER BY created_at DESC LIMIT $2 OFFSET $3`, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []AuditEvent{}
	for rows.Next() {
		var item AuditEvent
		var details []byte
		if err := rows.Scan(&item.ID, &item.ActorID, &item.ActorEmail, &item.Action, &item.ResourceType, &item.ResourceID, &details, &item.CreatedAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(details, &item.Details)
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) CountAudit(ctx context.Context, filter ListFilter) (int64, error) {
	var total int64
	err := s.pool.QueryRow(ctx, `
SELECT COUNT(*) FROM access_audit_events
WHERE ($1='' OR actor_email ILIKE '%' || $1 || '%' OR action ILIKE '%' || $1 || '%' OR resource_type ILIKE '%' || $1 || '%')`, filter.Query).Scan(&total)
	return total, err
}

func (s *Store) Overview(ctx context.Context) (Overview, error) {
	var overview Overview
	err := s.pool.QueryRow(ctx, `SELECT
 (SELECT COUNT(*) FROM access_users),
 (SELECT COUNT(*) FROM access_teams),
 (SELECT COUNT(*) FROM access_api_keys WHERE status='active' AND (expires_at IS NULL OR expires_at > NOW())),
 (SELECT COUNT(*) FROM access_api_keys WHERE status='active' AND expires_at BETWEEN NOW() AND NOW() + INTERVAL '7 days'),
 (SELECT COUNT(*) FROM access_groups),
 (SELECT COUNT(*) FROM access_budgets WHERE enabled=TRUE),
 (SELECT COUNT(*) FROM access_usage_events WHERE created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
 (SELECT COUNT(*) FROM access_usage_events WHERE status_code BETWEEN 200 AND 399 AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
 (SELECT COALESCE(SUM(total_tokens),0) FROM access_usage_events WHERE created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
 (SELECT COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),0)::BIGINT FROM access_usage_events WHERE created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC'))`).
		Scan(&overview.Users, &overview.Teams, &overview.ActiveKeys, &overview.ExpiringKeys, &overview.AccessGroups,
			&overview.EnabledBudgets, &overview.RequestsToday, &overview.SuccessfulToday, &overview.TokensToday, &overview.P95LatencyMS)
	return overview, err
}

func (s *Store) OverviewForUser(ctx context.Context, userID string) (Overview, error) {
	var overview Overview
	err := s.pool.QueryRow(ctx, `SELECT
  (SELECT COUNT(*) FROM access_users WHERE id=$1),
  (SELECT COUNT(*) FROM access_team_members WHERE user_id=$1),
  (SELECT COUNT(*) FROM access_api_keys WHERE user_id=$1 AND status='active' AND (expires_at IS NULL OR expires_at > NOW())),
  (SELECT COUNT(*) FROM access_api_keys WHERE user_id=$1 AND status='active' AND expires_at BETWEEN NOW() AND NOW() + INTERVAL '7 days'),
  (SELECT COUNT(DISTINCT b.group_id) FROM access_group_bindings b
     WHERE (b.subject_type='user' AND b.subject_id=$1)
        OR (b.subject_type='key' AND b.subject_id IN (SELECT id FROM access_api_keys WHERE user_id=$1))),
  (SELECT COUNT(*) FROM access_budgets q
     WHERE q.enabled=TRUE AND ((q.scope_type='user' AND q.scope_id=$1)
        OR (q.scope_type='key' AND q.scope_id IN (SELECT id FROM access_api_keys WHERE user_id=$1)))),
  (SELECT COUNT(*) FROM access_usage_events WHERE user_id=$1 AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
  (SELECT COUNT(*) FROM access_usage_events WHERE user_id=$1 AND status_code BETWEEN 200 AND 399 AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
  (SELECT COALESCE(SUM(total_tokens),0) FROM access_usage_events WHERE user_id=$1 AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC')),
  (SELECT COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),0)::BIGINT FROM access_usage_events WHERE user_id=$1 AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC'))`, userID).
		Scan(&overview.Users, &overview.Teams, &overview.ActiveKeys, &overview.ExpiringKeys, &overview.AccessGroups,
			&overview.EnabledBudgets, &overview.RequestsToday, &overview.SuccessfulToday, &overview.TokensToday, &overview.P95LatencyMS)
	return overview, err
}
