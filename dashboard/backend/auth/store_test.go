package auth

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
)

func TestNewStoreSyncsCanonicalRolePermissions(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "auth.db")
	store, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}

	if _, execErr := store.db.ExecContext(
		context.Background(),
		`INSERT INTO role_permissions(role, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(role, permission_key) DO UPDATE SET allowed = 1`,
		RoleRead,
		PermConfigWrite,
	); execErr != nil {
		t.Fatalf("insert stale read permission error = %v", execErr)
	}
	if closeErr := store.Close(); closeErr != nil {
		t.Fatalf("Close() error = %v", closeErr)
	}

	reopened, err := NewStore(path)
	if err != nil {
		t.Fatalf("reopen NewStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = reopened.Close()
	})

	perms, err := reopened.ListRolePermissions(context.Background())
	if err != nil {
		t.Fatalf("ListRolePermissions() error = %v", err)
	}
	if slices.Contains(perms[RoleRead], PermConfigWrite) {
		t.Fatalf("read role should not keep %q after sync", PermConfigWrite)
	}
}
