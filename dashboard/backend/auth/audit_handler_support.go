package auth

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const legacyAuditLogPageSize = 100

type AuditLogPageResponse struct {
	Logs  []AuditLog `json:"logs"`
	Total int        `json:"total"`
	Page  int        `json:"page"`
	Limit int        `json:"limit"`
}

// usesLegacyAuditLogResponse identifies the query surface supported before the
// paginated audit-log API was introduced. Keep those callers on the top-level
// array response while enterprise filters opt into the paginated envelope.
func usesLegacyAuditLogResponse(r *http.Request) bool {
	query := r.URL.Query()
	for _, key := range []string{"page", "q", "user", "status", "from", "to", "sort", "order"} {
		if query.Has(key) {
			return false
		}
	}
	return true
}

func applyLegacyAuditLogDefaults(r *http.Request, options *AuditLogListOptions) {
	if strings.TrimSpace(r.URL.Query().Get("limit")) == "" {
		options.Limit = legacyAuditLogPageSize
	}
	options.Offset = 0
	options.Sort = "id"
	options.Order = "desc"
}

func auditLogOptionsFromRequest(r *http.Request) (AuditLogListOptions, int, error) {
	query := r.URL.Query()
	page, err := auditPositiveQueryInt(query.Get("page"), 1, 1_000_000, "page")
	if err != nil {
		return AuditLogListOptions{}, 0, err
	}
	limit, err := auditPositiveQueryInt(query.Get("limit"), defaultAuditPageSize, 200, "limit")
	if err != nil {
		return AuditLogListOptions{}, 0, err
	}

	sortField := strings.TrimSpace(query.Get("sort"))
	if !auditLogSortAllowed(sortField) {
		return AuditLogListOptions{}, 0, fmt.Errorf("sort must be one of id, userId, action, resource, method, statusCode, createdAt")
	}
	order := strings.ToLower(strings.TrimSpace(query.Get("order")))
	if order != "" && order != "asc" && order != "desc" {
		return AuditLogListOptions{}, 0, fmt.Errorf("order must be asc or desc")
	}

	from, err := parseAuditLogTimestamp(query.Get("from"), false)
	if err != nil {
		return AuditLogListOptions{}, 0, fmt.Errorf("invalid from: %w", err)
	}
	to, err := parseAuditLogTimestamp(query.Get("to"), true)
	if err != nil {
		return AuditLogListOptions{}, 0, fmt.Errorf("invalid to: %w", err)
	}
	if from > 0 && to > 0 && from > to {
		return AuditLogListOptions{}, 0, fmt.Errorf("from must be before to")
	}

	userID := strings.TrimSpace(query.Get("user"))
	if userID == "" {
		userID = strings.TrimSpace(query.Get("userId"))
	}
	return AuditLogListOptions{
		Query:    query.Get("q"),
		UserID:   userID,
		Action:   query.Get("action"),
		Resource: query.Get("resource"),
		Status:   query.Get("status"),
		Sort:     sortField,
		Order:    order,
		From:     from,
		To:       to,
		Limit:    limit,
		Offset:   (page - 1) * limit,
	}, page, nil
}

func auditPositiveQueryInt(raw string, fallback, maximum int, name string) (int, error) {
	if strings.TrimSpace(raw) == "" {
		return fallback, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 || value > maximum {
		return 0, fmt.Errorf("%s must be between 1 and %d", name, maximum)
	}
	return value, nil
}

func auditLogSortAllowed(sortField string) bool {
	switch strings.ToLower(strings.TrimSpace(sortField)) {
	case "", "id", "userid", "action", "resource", "method", "statuscode", "createdat":
		return true
	default:
		return false
	}
}

func parseAuditLogTimestamp(raw string, endOfDay bool) (int64, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return 0, nil
	}
	if unixSeconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if unixSeconds < 0 {
			return 0, fmt.Errorf("timestamp cannot be negative")
		}
		return unixSeconds, nil
	}
	if parsed, err := time.Parse(time.RFC3339, value); err == nil {
		return parsed.Unix(), nil
	}
	parsed, err := time.Parse("2006-01-02", value)
	if err != nil {
		return 0, fmt.Errorf("use Unix seconds, RFC3339, or YYYY-MM-DD")
	}
	if endOfDay {
		parsed = parsed.Add(24*time.Hour - time.Second)
	}
	return parsed.Unix(), nil
}
