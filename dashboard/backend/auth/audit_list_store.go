package auth

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const defaultAuditPageSize = 25

var ErrInvalidAuditLogFilter = errors.New("invalid audit log filter")

type AuditLogListOptions struct {
	Query    string
	UserID   string
	Action   string
	Resource string
	Status   string
	Sort     string
	Order    string
	From     int64
	To       int64
	Limit    int
	Offset   int
}

func normalizeAuditLogListOptions(options AuditLogListOptions) AuditLogListOptions {
	if options.Limit <= 0 || options.Limit > 200 {
		options.Limit = defaultAuditPageSize
	}
	if options.Offset < 0 {
		options.Offset = 0
	}

	allowedSorts := map[string]string{
		"id":         "id",
		"userid":     "user_id",
		"action":     "action",
		"resource":   "resource",
		"method":     "method",
		"statuscode": "status_code",
		"createdat":  "created_at",
	}
	options.Sort = allowedSorts[strings.ToLower(strings.TrimSpace(options.Sort))]
	if options.Sort == "" {
		options.Sort = "created_at"
	}
	if strings.EqualFold(strings.TrimSpace(options.Order), "asc") {
		options.Order = "ASC"
	} else {
		options.Order = "DESC"
	}

	options.Query = strings.ToLower(strings.TrimSpace(options.Query))
	options.UserID = strings.TrimSpace(options.UserID)
	options.Action = strings.TrimSpace(options.Action)
	options.Resource = strings.TrimSpace(options.Resource)
	options.Status = strings.ToLower(strings.TrimSpace(options.Status))
	return options
}

func auditLogListPredicate(options AuditLogListOptions) (string, []interface{}, error) {
	if options.From > 0 && options.To > 0 && options.From > options.To {
		return "", nil, fmt.Errorf("%w: from must be before to", ErrInvalidAuditLogFilter)
	}

	predicates := make([]string, 0, 7)
	args := make([]interface{}, 0, 16)
	if options.Query != "" {
		pattern := auditLogLikePattern(options.Query)
		predicates = append(predicates, `(
			LOWER(COALESCE(user_id, '')) LIKE ? ESCAPE '\' OR
			LOWER(action) LIKE ? ESCAPE '\' OR
			LOWER(resource) LIKE ? ESCAPE '\' OR
			LOWER(method) LIKE ? ESCAPE '\' OR
			LOWER(path) LIKE ? ESCAPE '\' OR
			LOWER(ip) LIKE ? ESCAPE '\' OR
			LOWER(user_agent) LIKE ? ESCAPE '\' OR
			LOWER(extra_json) LIKE ? ESCAPE '\' OR
			CAST(status_code AS TEXT) LIKE ? ESCAPE '\'
		)`)
		for range 9 {
			args = append(args, pattern)
		}
	}
	for _, filter := range []struct{ column, values string }{
		{"user_id", options.UserID},
		{"action", options.Action},
	} {
		if values := listFilterValues(filter.values); len(values) > 0 {
			predicate, valueArgs := sqlMatchAny(filter.column, values)
			predicates = append(predicates, predicate)
			args = append(args, valueArgs...)
		}
	}
	if options.Resource != "" {
		predicates = append(predicates, "resource = ?")
		args = append(args, options.Resource)
	}
	if options.From > 0 {
		predicates = append(predicates, "created_at >= ?")
		args = append(args, options.From)
	}
	if options.To > 0 {
		predicates = append(predicates, "created_at <= ?")
		args = append(args, options.To)
	}

	statusPredicate, statusArgs, err := auditLogStatusPredicate(options.Status)
	if err != nil {
		return "", nil, err
	}
	if statusPredicate != "" {
		predicates = append(predicates, statusPredicate)
		args = append(args, statusArgs...)
	}

	if len(predicates) == 0 {
		return "", args, nil
	}
	return " WHERE " + strings.Join(predicates, " AND "), args, nil
}

// listFilterValues splits a comma-separated list filter into its trimmed values.
func listFilterValues(raw string) []string {
	values := make([]string, 0, 1)
	for _, value := range strings.Split(raw, ",") {
		if value = strings.TrimSpace(value); value != "" {
			values = append(values, value)
		}
	}
	return values
}

// sqlMatchAny matches a column against any of the values; column is a fixed SQL identifier.
func sqlMatchAny(column string, values []string) (string, []interface{}) {
	args := make([]interface{}, len(values))
	for index, value := range values {
		args[index] = value
	}
	if len(values) == 1 {
		return column + " = ?", args
	}
	return column + " IN (" + strings.TrimSuffix(strings.Repeat("?, ", len(values)), ", ") + ")", args
}

// auditLogStatusPredicate accepts one status or a comma-separated list that
// matches any of them; "all" anywhere in the list removes the filter.
func auditLogStatusPredicate(status string) (string, []interface{}, error) {
	values := listFilterValues(status)
	if len(values) <= 1 {
		return auditLogSingleStatusPredicate(status)
	}
	parts := make([]string, 0, len(values))
	args := make([]interface{}, 0, len(values)*2)
	for _, value := range values {
		predicate, valueArgs, err := auditLogSingleStatusPredicate(value)
		if err != nil {
			return "", nil, err
		}
		if predicate == "" {
			return "", nil, nil
		}
		parts = append(parts, "("+predicate+")")
		args = append(args, valueArgs...)
	}
	return "(" + strings.Join(parts, " OR ") + ")", args, nil
}

func auditLogSingleStatusPredicate(status string) (string, []interface{}, error) {
	switch status {
	case "", "all":
		return "", nil, nil
	case "success":
		return "status_code >= ? AND status_code < ?", []interface{}{200, 400}, nil
	case "failure", "error":
		return "status_code >= ?", []interface{}{400}, nil
	case "client_error", "4xx":
		return "status_code >= ? AND status_code < ?", []interface{}{400, 500}, nil
	case "server_error", "5xx":
		return "status_code >= ? AND status_code < ?", []interface{}{500, 600}, nil
	case "2xx", "3xx":
		class, _ := strconv.Atoi(status[:1])
		return "status_code >= ? AND status_code < ?", []interface{}{class * 100, (class + 1) * 100}, nil
	default:
		code, err := strconv.Atoi(status)
		if err != nil || code < 100 || code > 599 {
			return "", nil, fmt.Errorf("%w: unsupported status %q", ErrInvalidAuditLogFilter, status)
		}
		return "status_code = ?", []interface{}{code}, nil
	}
}

func auditLogLikePattern(value string) string {
	replacer := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return "%" + replacer.Replace(value) + "%"
}

func (s *Store) QueryAuditLogs(ctx context.Context, options AuditLogListOptions) ([]AuditLog, int, error) {
	options = normalizeAuditLogListOptions(options)
	where, args, err := auditLogListPredicate(options)
	if err != nil {
		return nil, 0, err
	}

	var total int
	countErr := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM user_audit_logs`+where, args...).Scan(&total)
	if countErr != nil {
		return nil, 0, countErr
	}

	// #nosec G202 -- Sort and Order are replaced by fixed allowlisted SQL tokens in
	// normalizeAuditLogListOptions; user-provided text only reaches bound parameters.
	query := `SELECT id, user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json
		FROM user_audit_logs` + where + fmt.Sprintf(" ORDER BY %s %s", options.Sort, options.Order)
	if options.Sort != "id" {
		query += fmt.Sprintf(", id %s", options.Order)
	}
	query += " LIMIT ? OFFSET ?"
	queryArgs := append(append([]interface{}{}, args...), options.Limit, options.Offset)

	rows, err := s.db.QueryContext(ctx, query, queryArgs...)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = rows.Close() }()

	logs := make([]AuditLog, 0, min(total, options.Limit))
	for rows.Next() {
		var logRow AuditLog
		var userID, method, path, ip, userAgent, extraJSON sql.NullString
		var statusCode sql.NullInt64
		if err := rows.Scan(
			&logRow.ID,
			&userID,
			&logRow.Action,
			&logRow.Resource,
			&method,
			&path,
			&ip,
			&userAgent,
			&statusCode,
			&logRow.CreatedAt,
			&extraJSON,
		); err != nil {
			return nil, 0, err
		}
		if userID.Valid {
			logRow.UserID = userID.String
		}
		logRow.Method = method.String
		logRow.Path = path.String
		logRow.IP = ip.String
		logRow.UserAgent = userAgent.String
		if statusCode.Valid {
			logRow.StatusCode = int(statusCode.Int64)
		}
		logRow.ExtraJSON = extraJSON.String
		logs = append(logs, logRow)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, err
	}
	return logs, total, nil
}

// AuditLogFacet is one value of a filterable audit field and how many entries carry it.
type AuditLogFacet struct {
	Value string `json:"value"`
	Count int    `json:"count"`
}

// AuditLogFacets lists the most frequent actors and actions, for filter menus.
type AuditLogFacets struct {
	Users   []AuditLogFacet `json:"users"`
	Actions []AuditLogFacet `json:"actions"`
}

const auditLogFacetLimit = 50

func (s *Store) AuditLogFacets(ctx context.Context) (AuditLogFacets, error) {
	users, err := s.auditLogFacet(ctx, "user_id")
	if err != nil {
		return AuditLogFacets{}, err
	}
	actions, err := s.auditLogFacet(ctx, "action")
	if err != nil {
		return AuditLogFacets{}, err
	}
	return AuditLogFacets{Users: users, Actions: actions}, nil
}

func (s *Store) auditLogFacet(ctx context.Context, column string) ([]AuditLogFacet, error) {
	// #nosec G201 -- column is one of the fixed identifiers passed by AuditLogFacets.
	query := fmt.Sprintf(
		"SELECT %[1]s, COUNT(*) FROM user_audit_logs WHERE %[1]s IS NOT NULL AND %[1]s <> '' GROUP BY %[1]s ORDER BY COUNT(*) DESC, %[1]s LIMIT ?",
		column,
	)
	rows, err := s.db.QueryContext(ctx, query, auditLogFacetLimit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	facets := make([]AuditLogFacet, 0, auditLogFacetLimit)
	for rows.Next() {
		var facet AuditLogFacet
		if err := rows.Scan(&facet.Value, &facet.Count); err != nil {
			return nil, err
		}
		facets = append(facets, facet)
	}
	return facets, rows.Err()
}
