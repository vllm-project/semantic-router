package auth

import (
	"context"
	"database/sql"
	"fmt"
)

const addUserAuthGenerationColumn = `ALTER TABLE users
ADD COLUMN auth_generation INTEGER NOT NULL DEFAULT 0 CHECK(auth_generation >= 0)`

// ensureUserAuthGenerationColumn upgrades auth databases created before the
// credential-generation fence existed. CREATE TABLE IF NOT EXISTS does not add
// columns to an existing SQLite table, so the migration must be explicit.
func ensureUserAuthGenerationColumn(ctx context.Context, db *sql.DB) error {
	exists, err := userTableHasColumn(ctx, db, "auth_generation")
	if err != nil {
		return err
	}
	if exists {
		return nil
	}
	if _, err := db.ExecContext(ctx, addUserAuthGenerationColumn); err == nil {
		return nil
	} else {
		// A second opener may have completed the additive migration after the
		// inspection. Recheck the schema before surfacing the ALTER error.
		if migrated, inspectErr := userTableHasColumn(ctx, db, "auth_generation"); inspectErr == nil && migrated {
			return nil
		}
		return fmt.Errorf("add users.auth_generation: %w", err)
	}
}

func userTableHasColumn(ctx context.Context, db *sql.DB, wanted string) (bool, error) {
	rows, err := db.QueryContext(ctx, `PRAGMA table_info(users)`)
	if err != nil {
		return false, fmt.Errorf("inspect users schema: %w", err)
	}
	defer func() { _ = rows.Close() }()

	for rows.Next() {
		var (
			columnID     int
			name         string
			columnType   string
			notNull      int
			defaultValue sql.NullString
			primaryKey   int
		)
		if err := rows.Scan(
			&columnID,
			&name,
			&columnType,
			&notNull,
			&defaultValue,
			&primaryKey,
		); err != nil {
			return false, fmt.Errorf("read users schema: %w", err)
		}
		if name == wanted {
			return true, nil
		}
	}
	if err := rows.Err(); err != nil {
		return false, fmt.Errorf("read users schema: %w", err)
	}
	return false, nil
}
