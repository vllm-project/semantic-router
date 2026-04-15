package store

import (
	"context"
	"database/sql"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

func newPostgresRuntimeConfig(cfg *PostgresConfig) (postgres.RuntimeConfig, error) {
	return postgres.NewRuntimeConfig(cfg, DefaultPostgresTableName)
}

func openConfiguredPostgresDB(ctx context.Context, cfg postgres.RuntimeConfig) (*sql.DB, error) {
	return postgres.OpenDB(ctx, cfg)
}
