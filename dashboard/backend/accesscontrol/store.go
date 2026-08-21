package accesscontrol

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	pool *pgxpool.Pool
}

type rowScanner interface {
	Scan(dest ...any) error
}

func lockTeamPolicy(ctx context.Context, tx pgx.Tx) error {
	_, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended('access-team-policy',0))`)
	return err
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

// effectiveModelPatterns selects the most specific grant tier. A key inherits
// from its user only when it has no direct grants, and from its Team only when
// neither the key nor user has grants.
func effectiveModelPatterns(keyPatterns, userPatterns, teamPatterns []string) []string {
	if patterns := uniqueStrings(keyPatterns); len(patterns) > 0 {
		return patterns
	}
	if patterns := uniqueStrings(userPatterns); len(patterns) > 0 {
		return patterns
	}
	return uniqueStrings(teamPatterns)
}

// effectiveBudgetID selects one reusable quota policy. Key overrides User,
// and User overrides the explicit Team context.
func effectiveBudgetID(keyBudgetID, userBudgetID, teamBudgetID string) string {
	for _, candidate := range []string{keyBudgetID, userBudgetID, teamBudgetID} {
		if candidate = strings.TrimSpace(candidate); candidate != "" {
			return candidate
		}
	}
	return ""
}
