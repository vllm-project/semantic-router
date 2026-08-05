package auth

import (
	"context"
	"path/filepath"
	"testing"
)

func TestUserListSupportsServerSearchPagingAndStats(t *testing.T) {
	t.Parallel()

	store, err := NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()

	fixtures := []struct {
		email  string
		name   string
		role   string
		status string
	}{
		{email: "zeta@example.com", name: "Zeta Operator", role: RoleWrite, status: "active"},
		{email: "alpha@example.com", name: "Alpha Admin", role: RoleAdmin, status: "active"},
		{email: "reader@example.com", name: "Archived Reader", role: RoleRead, status: "inactive"},
	}
	for _, fixture := range fixtures {
		if _, createErr := store.CreateUser(ctx, fixture.email, fixture.name, "hash", fixture.role, fixture.status); createErr != nil {
			t.Fatalf("CreateUser(%q) error = %v", fixture.email, createErr)
		}
	}

	options := UserListOptions{Status: "active", Query: "a", Sort: "email", Order: "asc", Limit: 1}
	users, err := store.ListUsers(ctx, options)
	if err != nil {
		t.Fatalf("ListUsers() error = %v", err)
	}
	if len(users) != 1 || users[0].Email != "alpha@example.com" {
		t.Fatalf("ListUsers() = %#v, want first sorted active match", users)
	}
	total, err := store.CountFilteredUsers(ctx, options)
	if err != nil || total != 2 {
		t.Fatalf("CountFilteredUsers() = %d, %v; want 2", total, err)
	}
	stats, err := store.UserDirectoryStats(ctx)
	if err != nil {
		t.Fatalf("UserDirectoryStats() error = %v", err)
	}
	if stats.Active != 2 || stats.Privileged != 1 {
		t.Fatalf("UserDirectoryStats() = %#v, want active=2 privileged=1", stats)
	}
}
