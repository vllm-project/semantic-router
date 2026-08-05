package auth

import (
	"context"
)

type AuditLog struct {
	ID         int64  `json:"id"`
	UserID     string `json:"userId"`
	Action     string `json:"action"`
	Resource   string `json:"resource"`
	Method     string `json:"method"`
	Path       string `json:"path"`
	IP         string `json:"ip"`
	UserAgent  string `json:"userAgent"`
	StatusCode int    `json:"statusCode"`
	CreatedAt  int64  `json:"createdAt"`
	ExtraJSON  string `json:"extraJson,omitempty"`
}

func (s *Store) AddAuditLog(ctx context.Context, logRow AuditLog) error {
	createdAt := logRow.CreatedAt
	if createdAt <= 0 {
		createdAt = nowUnix()
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO user_audit_logs(user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		nilOrString(logRow.UserID), logRow.Action, logRow.Resource, logRow.Method, logRow.Path, logRow.IP, logRow.UserAgent, logRow.StatusCode, createdAt, logRow.ExtraJSON)
	return err
}

func (s *Store) ListAuditLogs(ctx context.Context, userID, action, resource string, limit, offset int) ([]AuditLog, error) {
	if limit <= 0 || limit > 200 {
		limit = defaultPageSize
	}
	logs, _, err := s.QueryAuditLogs(ctx, AuditLogListOptions{
		UserID:   userID,
		Action:   action,
		Resource: resource,
		Sort:     "id",
		Order:    "desc",
		Limit:    limit,
		Offset:   offset,
	})
	return logs, err
}

func nilOrString(v string) interface{} {
	if v == "" {
		return nil
	}
	return v
}
