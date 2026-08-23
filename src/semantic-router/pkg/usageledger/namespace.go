package usageledger

import (
	"context"
	"database/sql"
	"fmt"
)

// ActiveNamespace identifies one live usage stream. The quota partition is
// deliberately discovered from PostgreSQL rather than derived from the
// namespace ID: it is part of the published quota-runtime contract.
type ActiveNamespace struct {
	ID               string
	QuotaPartitionID string
}

// NamespaceSource supplies the complete active namespace set. Implementations
// must return a point-in-time snapshot; Supervisor computes additions and
// removals from successive snapshots.
type NamespaceSource interface {
	ListActiveNamespaces(context.Context) ([]ActiveNamespace, error)
}

// PostgresNamespaceSource discovers active usage partitions from the managed
// desired-state authority.
type PostgresNamespaceSource struct {
	db *sql.DB
}

func NewPostgresNamespaceSource(db *sql.DB) (*PostgresNamespaceSource, error) {
	if db == nil {
		return nil, fmt.Errorf("usage namespace database is required")
	}
	return &PostgresNamespaceSource{db: db}, nil
}

func (source *PostgresNamespaceSource) ListActiveNamespaces(ctx context.Context) ([]ActiveNamespace, error) {
	if source == nil || source.db == nil {
		return nil, fmt.Errorf("usage namespace source is unavailable")
	}
	rows, err := source.db.QueryContext(ctx, `SELECT id::text, quota_partition_id
FROM access_namespaces
WHERE status = 'active'
ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("discover active usage namespaces: %w", err)
	}
	defer rows.Close()

	namespaces := make([]ActiveNamespace, 0)
	for rows.Next() {
		var namespace ActiveNamespace
		if err := rows.Scan(&namespace.ID, &namespace.QuotaPartitionID); err != nil {
			return nil, fmt.Errorf("scan active usage namespace: %w", err)
		}
		namespaces = append(namespaces, namespace)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate active usage namespaces: %w", err)
	}
	return namespaces, nil
}
