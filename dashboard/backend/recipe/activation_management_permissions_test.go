package recipe

import (
	"errors"
	"os"
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

func TestManagementCredentialUsesExplicitRuntimeEnvironmentWithoutPersistingIt(t *testing.T) {
	token := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	t.Setenv(ManagementCredentialEnv, token)
	store := NewStore(StoreOptions{Root: t.TempDir()})

	got, err := store.ManagementCredential()
	if err != nil {
		t.Fatalf("ManagementCredential() error = %v", err)
	}
	if got != token {
		t.Fatalf("ManagementCredential() = %q, want runtime token", got)
	}
	if _, err := store.readManagementCredentialLocked(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("runtime token must not be persisted, read error = %v", err)
	}
}

func TestManagementCredentialPrefersExplicitRuntimeEnvironmentOverPersistedActivationToken(t *testing.T) {
	runtimeToken := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	persistedToken := "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	t.Setenv(ManagementCredentialEnv, runtimeToken)
	store := NewStore(StoreOptions{Root: t.TempDir()})
	if err := store.ensureLayout(); err != nil {
		t.Fatalf("ensure store layout: %v", err)
	}
	if err := writeFileAtomically(
		store.managementCredentialPath(), []byte(persistedToken+"\n"), 0o600,
	); err != nil {
		t.Fatalf("persist stale activation token: %v", err)
	}

	got, err := store.ManagementCredential()
	if err != nil {
		t.Fatalf("ManagementCredential() error = %v", err)
	}
	if got != runtimeToken {
		t.Fatalf("ManagementCredential() = %q, want explicit runtime token", got)
	}
}

func TestEnsureManagementCredentialUsesExplicitRuntimeEnvironmentWithoutReplacingPersistedToken(t *testing.T) {
	runtimeToken := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	persistedToken := "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	t.Setenv(ManagementCredentialEnv, runtimeToken)
	store := NewStore(StoreOptions{Root: t.TempDir()})
	if err := store.ensureLayout(); err != nil {
		t.Fatalf("ensure store layout: %v", err)
	}
	if err := writeFileAtomically(
		store.managementCredentialPath(), []byte(persistedToken+"\n"), 0o600,
	); err != nil {
		t.Fatalf("persist stale activation token: %v", err)
	}

	got, err := store.EnsureManagementCredential()
	if err != nil {
		t.Fatalf("EnsureManagementCredential() error = %v", err)
	}
	if got != runtimeToken {
		t.Fatalf("EnsureManagementCredential() = %q, want explicit runtime token", got)
	}
	persisted, err := store.readManagementCredentialLocked()
	if err != nil {
		t.Fatalf("read persisted activation token: %v", err)
	}
	if persisted != persistedToken {
		t.Fatalf("persisted activation token = %q, want unchanged stale token", persisted)
	}
}
