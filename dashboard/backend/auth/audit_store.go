package auth

import (
	"context"
	"database/sql"
	"strings"
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
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO user_audit_logs(user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		nilOrString(logRow.UserID), logRow.Action, logRow.Resource, logRow.Method, logRow.Path, logRow.IP, logRow.UserAgent, logRow.StatusCode, nowUnix(), logRow.ExtraJSON)
	return err
}

func (s *Store) ListAuditLogs(ctx context.Context, userID, action, resource string, limit, offset int) ([]AuditLog, error) {
	if limit <= 0 || limit > 200 {
		limit = defaultPageSize
	}
	q := `SELECT id, user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json FROM user_audit_logs`
	args := []interface{}{}
	predicates := []string{}
	if userID != "" {
		predicates = append(predicates, "user_id = ?")
		args = append(args, userID)
	}
	if action != "" {
		predicates = append(predicates, "action = ?")
		args = append(args, action)
	}
	if resource != "" {
		predicates = append(predicates, "resource = ?")
		args = append(args, resource)
	}
	if len(predicates) > 0 {
		q += " WHERE " + strings.Join(predicates, " AND ")
	}
	q += " ORDER BY id DESC LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	var out []AuditLog
	for rows.Next() {
		var row AuditLog
		var uid sql.NullString
		if err := rows.Scan(&row.ID, &uid, &row.Action, &row.Resource, &row.Method, &row.Path, &row.IP, &row.UserAgent, &row.StatusCode, &row.CreatedAt, &row.ExtraJSON); err != nil {
			return nil, err
		}
		if uid.Valid {
			row.UserID = uid.String
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func nilOrString(v string) interface{} {
	if v == "" {
		return nil
	}
	return v
}
