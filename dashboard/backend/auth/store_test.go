package auth

import (
	"context"
	"path/filepath"
	"slices"
	"testing"
)

func TestNewStoreNormalizesLegacyRolesAndSyncsDefaultPermissions(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "auth.db")
	store, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}

	admin := newTestUser(t, NewService(store, "test-secret", 1), "admin@example.com", RoleAdmin, "active")
	reader := newTestUser(t, NewService(store, "test-secret", 1), "reader@example.com", RoleRead, "active")

	if _, execErr := store.db.ExecContext(
		context.Background(),
		`UPDATE users SET role = ? WHERE id = ?`,
		legacyRoleSuperAdmin,
		admin.ID,
	); execErr != nil {
		t.Fatalf("downgrade admin role fixture error = %v", execErr)
	}
	if _, execErr := store.db.ExecContext(
		context.Background(),
		`UPDATE users SET role = ? WHERE id = ?`,
		legacyRoleReadonly,
		reader.ID,
	); execErr != nil {
		t.Fatalf("downgrade read role fixture error = %v", execErr)
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
	if _, execErr := store.db.ExecContext(
		context.Background(),
		`INSERT INTO role_permissions(role, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(role, permission_key) DO UPDATE SET allowed = 1`,
		legacyRoleReadonly,
		PermConfigRead,
	); execErr != nil {
		t.Fatalf("insert legacy role permission error = %v", execErr)
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

	adminUser, err := reopened.GetUserByID(context.Background(), admin.ID)
	if err != nil {
		t.Fatalf("GetUserByID(admin) error = %v", err)
	}
	if adminUser.Role != RoleAdmin {
		t.Fatalf("admin role = %q, want %q", adminUser.Role, RoleAdmin)
	}

	readUser, err := reopened.GetUserByID(context.Background(), reader.ID)
	if err != nil {
		t.Fatalf("GetUserByID(reader) error = %v", err)
	}
	if readUser.Role != RoleRead {
		t.Fatalf("reader role = %q, want %q", readUser.Role, RoleRead)
	}

	perms, err := reopened.ListRolePermissions(context.Background())
	if err != nil {
		t.Fatalf("ListRolePermissions() error = %v", err)
	}
	if slices.Contains(perms[RoleRead], PermConfigWrite) {
		t.Fatalf("read role should not keep %q after sync", PermConfigWrite)
	}
	if _, ok := perms[legacyRoleReadonly]; ok {
		t.Fatalf("legacy readonly role permissions should be removed")
	}
}
