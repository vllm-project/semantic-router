package postgres

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const migrationLockID int64 = 0x5653524D474D5431 // "VSRMGMT1"

//go:embed migrations/*.sql
var migrationFiles embed.FS

// Migration is one immutable, forward-only durable Management schema step.
type Migration struct {
	Version int64
	Name    string
	SQL     string
}

// Migrations returns the embedded migration set in application order.
func Migrations() ([]Migration, error) {
	entries, err := fs.ReadDir(migrationFiles, "migrations")
	if err != nil {
		return nil, fmt.Errorf("read embedded Management migrations: %w", err)
	}
	migrations := make([]Migration, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".sql" {
			continue
		}
		version, err := migrationVersion(entry.Name())
		if err != nil {
			return nil, err
		}
		body, err := migrationFiles.ReadFile("migrations/" + entry.Name())
		if err != nil {
			return nil, fmt.Errorf("read migration %s: %w", entry.Name(), err)
		}
		if strings.TrimSpace(string(body)) == "" {
			return nil, fmt.Errorf("migration %s is empty", entry.Name())
		}
		migrations = append(migrations, Migration{Version: version, Name: entry.Name(), SQL: string(body)})
	}
	sort.Slice(migrations, func(i, j int) bool { return migrations[i].Version < migrations[j].Version })
	for i := range migrations {
		if i > 0 && migrations[i-1].Version == migrations[i].Version {
			return nil, fmt.Errorf("duplicate Management migration version %d", migrations[i].Version)
		}
	}
	return migrations, nil
}

func migrationVersion(name string) (int64, error) {
	prefix, _, ok := strings.Cut(name, "_")
	if !ok || prefix == "" {
		return 0, fmt.Errorf("migration %q must start with a numeric version and underscore", name)
	}
	version, err := strconv.ParseInt(prefix, 10, 64)
	if err != nil || version <= 0 {
		return 0, fmt.Errorf("migration %q has invalid version", name)
	}
	return version, nil
}

// Migrator applies embedded migrations under a PostgreSQL session advisory lock.
// It is intended for the explicit migration command or Job, never Router startup.
type Migrator struct {
	DB *sql.DB
}

// Apply advances the schema to the latest embedded version.
func (m Migrator) Apply(ctx context.Context) (returnErr error) {
	if m.DB == nil {
		return fmt.Errorf("management migration database is required")
	}
	migrations, applyErr := Migrations()
	if applyErr != nil {
		return applyErr
	}
	conn, applyErr := m.DB.Conn(ctx)
	if applyErr != nil {
		return fmt.Errorf("acquire migration connection: %w", applyErr)
	}
	defer func() {
		if closeErr := conn.Close(); closeErr != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("close migration connection: %w", closeErr))
		}
	}()
	if _, err := conn.ExecContext(ctx, `SELECT pg_advisory_lock($1)`, migrationLockID); err != nil {
		return fmt.Errorf("lock Management migrations: %w", err)
	}
	defer func() {
		_, unlockErr := conn.ExecContext(context.Background(), `SELECT pg_advisory_unlock($1)`, migrationLockID)
		if unlockErr != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("unlock Management migrations: %w", unlockErr))
		}
	}()

	if _, err := conn.ExecContext(ctx, `
CREATE TABLE IF NOT EXISTS router_management_schema_migrations (
  version BIGINT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
)`); err != nil {
		return fmt.Errorf("create migration ledger: %w", err)
	}

	applied, applyErr := appliedVersions(ctx, conn)
	if applyErr != nil {
		return applyErr
	}
	if err := validateAppliedMigrations(migrations, applied); err != nil {
		return err
	}
	return applyPendingMigrations(ctx, conn, migrations, applied)
}

func applyPendingMigrations(
	ctx context.Context,
	conn *sql.Conn,
	migrations []Migration,
	applied map[int64]string,
) error {
	for _, migration := range migrations {
		if _, exists := applied[migration.Version]; exists {
			continue
		}
		tx, err := conn.BeginTx(ctx, &sql.TxOptions{})
		if err != nil {
			return fmt.Errorf("begin migration %s: %w", migration.Name, err)
		}
		if _, err = tx.ExecContext(ctx, migration.SQL); err == nil {
			_, err = tx.ExecContext(ctx,
				`INSERT INTO router_management_schema_migrations(version, name) VALUES ($1, $2)`,
				migration.Version, migration.Name,
			)
		}
		if err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("apply migration %s: %w", migration.Name, err)
		}
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit migration %s: %w", migration.Name, err)
		}
	}
	return nil
}

func appliedVersions(ctx context.Context, conn *sql.Conn) (_ map[int64]string, returnErr error) {
	rows, err := conn.QueryContext(ctx, `SELECT version,name FROM router_management_schema_migrations ORDER BY version`)
	if err != nil {
		return nil, fmt.Errorf("read applied migrations: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	versions := make(map[int64]string)
	for rows.Next() {
		var version int64
		var name string
		if err := rows.Scan(&version, &name); err != nil {
			return nil, fmt.Errorf("scan applied migration: %w", err)
		}
		versions[version] = name
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate applied migrations: %w", err)
	}
	return versions, nil
}

func validateAppliedMigrations(migrations []Migration, applied map[int64]string) error {
	embedded := make(map[int64]string, len(migrations))
	for _, migration := range migrations {
		embedded[migration.Version] = migration.Name
	}
	for version, name := range applied {
		expected, exists := embedded[version]
		if !exists {
			return fmt.Errorf("database contains unknown control-plane migration %d (%s)", version, name)
		}
		if name != expected {
			return fmt.Errorf("control-plane migration %d name is %q, want %q", version, name, expected)
		}
	}
	missing := false
	for _, migration := range migrations {
		_, exists := applied[migration.Version]
		if !exists {
			missing = true
			continue
		}
		if missing {
			return fmt.Errorf("applied control-plane migrations are not a contiguous prefix at version %d", migration.Version)
		}
	}
	return nil
}
