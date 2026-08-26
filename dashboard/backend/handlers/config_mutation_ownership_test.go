package handlers

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteConfigAtomicallyRejectsDatabaseOwnedResources(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	original := []byte("version: v0.3\nrouting: {}\n")
	if err := os.WriteFile(configPath, original, 0o600); err != nil {
		t.Fatalf("write original config: %v", err)
	}

	blocked := []byte(`version: v0.3
routing: {}
enterprise:
  tenant_grants: [{tenant: acme, role: admin}]
  budgetPolicy: {daily: 1}
  virtual-keys: [{name: hidden}]
  audit_policy: {retention: disabled}
`)
	err := writeConfigAtomically(configPath, blocked)
	if !errors.Is(err, ErrDatabaseOwnedConfigMutation) {
		t.Fatalf("write error = %v, want %v", err, ErrDatabaseOwnedConfigMutation)
	}

	after, readErr := os.ReadFile(configPath)
	if readErr != nil {
		t.Fatalf("read config after rejected write: %v", readErr)
	}
	if string(after) != string(original) {
		t.Fatalf("config changed after rejected enterprise mutation:\n%s", after)
	}
}

func TestWriteConfigAtomicallyAllowsRouterOwnedConfig(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configData := []byte("version: v0.3\nrouting: {}\n")
	if err := writeConfigAtomically(configPath, configData); err != nil {
		t.Fatalf("write router config: %v", err)
	}
}
