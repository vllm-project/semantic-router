package recipe

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestManagementCredentialHasNarrowReplayReadAndDetailPermissions(t *testing.T) {
	permissions := make(map[string]bool)
	for _, permission := range ManagementCredentialPermissions() {
		permissions[permission] = true
	}
	for _, required := range []string{"replay.read", "replay.detail"} {
		if !permissions[required] {
			t.Fatalf("managed Dashboard service role missing %q", required)
		}
	}
	for _, forbidden := range []string{"*", "secret_view", "data.write"} {
		if permissions[forbidden] {
			t.Fatalf("managed Dashboard service role must not receive broad permission %q", forbidden)
		}
	}
}

func TestManagementCredentialComesFromTheEnvironmentAndIsNeverWritten(t *testing.T) {
	token := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	t.Setenv(ManagementCredentialEnv, token)
	store := NewStore(StoreOptions{Root: t.TempDir()})
	if err := store.ensureLayout(); err != nil {
		t.Fatalf("ensure store layout: %v", err)
	}

	got, err := store.ManagementCredential()
	if err != nil || got != token || !store.HasManagementCredential() {
		t.Fatalf("ManagementCredential() = %q, %v; want the runtime token", got, err)
	}
	if _, err := os.Lstat(filepath.Join(store.Root(), "credentials")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the store keeps a credentials directory: %v", err)
	}
}

func TestManagementCredentialIgnoresATokenAnEarlierDashboardWrote(t *testing.T) {
	t.Setenv(ManagementCredentialEnv, "")
	store := NewStore(StoreOptions{Root: t.TempDir()})
	stale := filepath.Join(store.Root(), "credentials", "router-management.token")
	if err := os.MkdirAll(filepath.Dir(stale), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stale, []byte("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := store.ManagementCredential(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ManagementCredential() error = %v, want os.ErrNotExist", err)
	}
	if store.HasManagementCredential() {
		t.Fatal("a token on disk counted as the management credential")
	}
}

func TestManagementCredentialRejectsAMalformedEnvironmentValue(t *testing.T) {
	t.Setenv(ManagementCredentialEnv, "not-a-management-token")
	store := NewStore(StoreOptions{Root: t.TempDir()})

	if _, err := store.ManagementCredential(); err == nil || errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ManagementCredential() error = %v, want a validation error", err)
	}
}
