package accesscontrol

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	pool *pgxpool.Pool
}

type rowScanner interface {
	Scan(dest ...any) error
}

func OpenStore(ctx context.Context, databaseURL string) (*Store, error) {
	if strings.TrimSpace(databaseURL) == "" {
		return nil, errors.New("ACCESS_CONTROL_DATABASE_URL is required")
	}
	poolConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse access-control database URL: %w", err)
	}
	poolConfig.MaxConns = 20
	poolConfig.MinConns = 2
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("open access-control database: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping access-control database: %w", err)
	}
	if _, err := pool.Exec(ctx, schema); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate access-control database: %w", err)
	}
	return &Store{pool: pool}, nil
}

func (s *Store) Close() { s.pool.Close() }

func normalizeFilter(filter ListFilter) ListFilter {
	if filter.Limit <= 0 || filter.Limit > 100 {
		filter.Limit = 100
	}
	if filter.Offset < 0 {
		filter.Offset = 0
	}
	return filter
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

// effectiveModelPatterns makes direct key grants authoritative. A key with no
// direct binding inherits its user/team grants; once a direct binding is
// present it becomes that key's explicit model visibility. This lets two keys
// owned by the same user intentionally expose different model catalogs.
func effectiveModelPatterns(keyPatterns, inheritedPatterns []string) []string {
	if patterns := uniqueStrings(keyPatterns); len(patterns) > 0 {
		return patterns
	}
	return uniqueStrings(inheritedPatterns)
}

func uniqueBindings(values []Binding) []Binding {
	seen := make(map[string]struct{}, len(values))
	result := make([]Binding, 0, len(values))
	for _, value := range values {
		value.SubjectType = strings.TrimSpace(value.SubjectType)
		value.SubjectID = strings.TrimSpace(value.SubjectID)
		key := value.SubjectType + ":" + value.SubjectID
		if value.SubjectID == "" || (value.SubjectType != "user" && value.SubjectType != "team" && value.SubjectType != "key") {
			continue
		}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, value)
	}
	return result
}
