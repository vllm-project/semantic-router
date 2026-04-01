package store

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

type postgresRuntimeConfig struct {
	host            string
	port            int
	database        string
	user            string
	password        string
	sslMode         string
	tableName       string
	maxOpenConns    int
	maxIdleConns    int
	connMaxLifetime int
}

func newPostgresRuntimeConfig(cfg *PostgresConfig) (postgresRuntimeConfig, error) {
	if cfg == nil {
		return postgresRuntimeConfig{}, fmt.Errorf("postgres config is required")
	}

	runtimeCfg := postgresRuntimeConfig{
		host:            defaultString(cfg.Host, "localhost"),
		port:            defaultPositiveInt(cfg.Port, 5432),
		database:        cfg.Database,
		user:            cfg.User,
		password:        cfg.Password,
		sslMode:         defaultString(cfg.SSLMode, "disable"),
		tableName:       defaultString(cfg.TableName, DefaultPostgresTableName),
		maxOpenConns:    defaultPositiveInt(cfg.MaxOpenConns, DefaultPostgresMaxOpenConns),
		maxIdleConns:    defaultPositiveInt(cfg.MaxIdleConns, DefaultPostgresMaxIdleConns),
		connMaxLifetime: defaultPositiveInt(cfg.ConnMaxLifetime, DefaultPostgresConnMaxLifetime),
	}

	if err := requirePostgresField("database name", runtimeCfg.database); err != nil {
		return postgresRuntimeConfig{}, err
	}
	if err := requirePostgresField("user", runtimeCfg.user); err != nil {
		return postgresRuntimeConfig{}, err
	}
	if err := validatePostgresIdentifier(runtimeCfg.tableName); err != nil {
		return postgresRuntimeConfig{}, err
	}
	return runtimeCfg, nil
}

func defaultString(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func defaultPositiveInt(value int, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}

func requirePostgresField(name string, value string) error {
	if value == "" {
		return fmt.Errorf("postgres %s is required", name)
	}
	return nil
}

func openConfiguredPostgresDB(ctx context.Context, cfg postgresRuntimeConfig) (*sql.DB, error) {
	db, err := sql.Open("postgres", postgresConnectionString(cfg))
	if err != nil {
		return nil, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	configurePostgresDB(db, cfg)
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to ping postgres: %w", err)
	}
	return db, nil
}

func postgresConnectionString(cfg postgresRuntimeConfig) string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.host, cfg.port, cfg.user, cfg.password, cfg.database, cfg.sslMode,
	)
}

func configurePostgresDB(db *sql.DB, cfg postgresRuntimeConfig) {
	db.SetMaxOpenConns(cfg.maxOpenConns)
	db.SetMaxIdleConns(cfg.maxIdleConns)
	db.SetConnMaxLifetime(time.Duration(cfg.connMaxLifetime) * time.Second)
}
