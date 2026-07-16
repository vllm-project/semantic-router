package auth

import (
	"context"
	"fmt"
	"strings"
)

type UserListOptions struct {
	Status string
	Query  string
	Sort   string
	Order  string
	Limit  int
	Offset int
}

type UserDirectoryStats struct {
	Active     int `json:"active"`
	Privileged int `json:"privileged"`
}

func normalizeUserListOptions(options UserListOptions) UserListOptions {
	if options.Limit <= 0 || options.Limit > 200 {
		options.Limit = defaultPageSize
	}
	if options.Offset < 0 {
		options.Offset = 0
	}

	allowedSorts := map[string]string{
		"email":       "email",
		"name":        "name",
		"role":        "role",
		"status":      "status",
		"createdat":   "created_at",
		"updatedat":   "updated_at",
		"lastloginat": "last_login_at",
	}
	options.Sort = allowedSorts[strings.ToLower(strings.TrimSpace(options.Sort))]
	if options.Sort == "" {
		options.Sort = "created_at"
	}
	if strings.EqualFold(options.Order, "asc") {
		options.Order = "ASC"
	} else {
		options.Order = "DESC"
	}
	options.Status = strings.TrimSpace(options.Status)
	options.Query = strings.ToLower(strings.TrimSpace(options.Query))
	return options
}

func userListPredicate(options UserListOptions) (string, []interface{}) {
	predicates := make([]string, 0, 2)
	args := make([]interface{}, 0, 4)
	if options.Status != "" && options.Status != "all" {
		predicates = append(predicates, "status = ?")
		args = append(args, options.Status)
	}
	if options.Query != "" {
		pattern := "%" + options.Query + "%"
		predicates = append(predicates, "(LOWER(email) LIKE ? OR LOWER(name) LIKE ? OR LOWER(role) LIKE ?)")
		args = append(args, pattern, pattern, pattern)
	}
	if len(predicates) == 0 {
		return "", args
	}
	return " WHERE " + strings.Join(predicates, " AND "), args
}

func (s *Store) ListUsers(ctx context.Context, options UserListOptions) ([]*User, error) {
	options = normalizeUserListOptions(options)
	where, args := userListPredicate(options)
	// #nosec G202 -- Sort and Order are replaced by fixed allowlisted SQL tokens in
	// normalizeUserListOptions; user-provided text only reaches bound parameters.
	query := `SELECT id, email, name, role, status, created_at, updated_at, last_login_at FROM users` +
		where + fmt.Sprintf(" ORDER BY %s %s LIMIT ? OFFSET ?", options.Sort, options.Order)
	args = append(args, options.Limit, options.Offset)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var users []*User
	for rows.Next() {
		user, scanErr := scanUserRows(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		users = append(users, user)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return users, nil
}

func (s *Store) CountFilteredUsers(ctx context.Context, options UserListOptions) (int, error) {
	options = normalizeUserListOptions(options)
	where, args := userListPredicate(options)
	var total int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`+where, args...).Scan(&total); err != nil {
		return 0, err
	}
	return total, nil
}

func (s *Store) UserDirectoryStats(ctx context.Context) (UserDirectoryStats, error) {
	var stats UserDirectoryStats
	err := s.db.QueryRowContext(ctx, `
		SELECT
			COALESCE(SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN role = ? THEN 1 ELSE 0 END), 0)
		FROM users`, RoleAdmin).Scan(&stats.Active, &stats.Privileged)
	return stats, err
}
