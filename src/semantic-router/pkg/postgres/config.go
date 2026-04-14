package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"time"

	_ "github.com/lib/pq"
)

const (
	DefaultMaxOpenConns    = 25
	DefaultMaxIdleConns    = 5
	DefaultConnMaxLifetime = 300 // seconds
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Config holds PostgreSQL connection and pool parameters shared across
// subsystems (router replay, vector store registry, etc.).
type Config struct {
	Host     string `json:"host" yaml:"host"`
	Port     int    `json:"port" yaml:"port"`
	Database string `json:"database" yaml:"database"`
	User     string `json:"user" yaml:"user"`
	Password string `json:"password" yaml:"password"`
	SSLMode  string `json:"ssl_mode,omitempty" yaml:"ssl_mode,omitempty"`
	// Connection pool settings
	MaxOpenConns    int    `json:"max_open_conns,omitempty" yaml:"max_open_conns,omitempty"`
	MaxIdleConns    int    `json:"max_idle_conns,omitempty" yaml:"max_idle_conns,omitempty"`
	ConnMaxLifetime int    `json:"conn_max_lifetime,omitempty" yaml:"conn_max_lifetime,omitempty"` // seconds
	TableName       string `json:"table_name,omitempty" yaml:"table_name,omitempty"`
}

// RuntimeConfig holds validated, defaulted connection parameters ready for use.
type RuntimeConfig struct {
	Host            string
	Port            int
	Database        string
	User            string
	Password        string
	SSLMode         string
	TableName       string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime int
}

// NewRuntimeConfig validates and applies defaults to a Config, producing
// a RuntimeConfig suitable for opening a connection.
func NewRuntimeConfig(cfg *Config, defaultTableName string) (RuntimeConfig, error) {
	if cfg == nil {
		return RuntimeConfig{}, fmt.Errorf("postgres config is required")
	}

	rc := RuntimeConfig{
		Host:            defaultString(cfg.Host, "localhost"),
		Port:            defaultPositiveInt(cfg.Port, 5432),
		Database:        cfg.Database,
		User:            cfg.User,
		Password:        cfg.Password,
		SSLMode:         defaultString(cfg.SSLMode, "disable"),
		TableName:       defaultString(cfg.TableName, defaultTableName),
		MaxOpenConns:    defaultPositiveInt(cfg.MaxOpenConns, DefaultMaxOpenConns),
		MaxIdleConns:    defaultPositiveInt(cfg.MaxIdleConns, DefaultMaxIdleConns),
		ConnMaxLifetime: defaultPositiveInt(cfg.ConnMaxLifetime, DefaultConnMaxLifetime),
	}

	if rc.Database == "" {
		return RuntimeConfig{}, fmt.Errorf("postgres database name is required")
	}
	if rc.User == "" {
		return RuntimeConfig{}, fmt.Errorf("postgres user is required")
	}
	if err := ValidateIdentifier(rc.TableName); err != nil {
		return RuntimeConfig{}, err
	}
	return rc, nil
}

// OpenDB opens a *sql.DB using the given RuntimeConfig, configures the pool,
// and pings to verify connectivity.
func OpenDB(ctx context.Context, rc RuntimeConfig) (*sql.DB, error) {
	connStr := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		rc.Host, rc.Port, rc.User, rc.Password, rc.Database, rc.SSLMode,
	)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	db.SetMaxOpenConns(rc.MaxOpenConns)
	db.SetMaxIdleConns(rc.MaxIdleConns)
	db.SetConnMaxLifetime(time.Duration(rc.ConnMaxLifetime) * time.Second)

	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to ping postgres: %w", err)
	}
	return db, nil
}

// ValidateIdentifier checks that a SQL identifier (table name, index name)
// contains only safe characters.
func ValidateIdentifier(name string) error {
	if !identifierPattern.MatchString(name) {
		return fmt.Errorf("invalid postgres identifier %q", name)
	}
	return nil
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func defaultPositiveInt(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	return value
}
