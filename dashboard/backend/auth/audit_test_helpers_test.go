package auth

import (
	"context"
	"testing"
)

func seedAuditFixtureUsers(t *testing.T, store *Store, userIDs ...string) {
	t.Helper()
	for _, userID := range userIDs {
		if _, err := store.db.ExecContext(
			context.Background(),
			`INSERT INTO users(
				id, email, name, password_hash, role, status, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			userID,
			userID+"@example.test",
			"Audit Fixture",
			"test-only-hash",
			RoleRead,
			defaultUserStatusActive,
			1,
			1,
		); err != nil {
			t.Fatalf("seed audit fixture user %q: %v", userID, err)
		}
	}
}
