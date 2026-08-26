package config

import (
	"path/filepath"
	"testing"
)

func TestResolveConfigPathsHonorsDashboardBaseDirectoryOverride(t *testing.T) {
	configRoot := t.TempDir()
	baseDirectory := t.TempDir()
	t.Setenv("DASHBOARD_CONFIG_DIR", baseDirectory)
	cfg := &Config{ConfigFile: filepath.Join(configRoot, ".vllm-sr", "runtime-config.stack.yaml")}
	if err := resolveConfigPaths(cfg); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(baseDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ConfigDir != want {
		t.Fatalf("ConfigDir = %q, want %q", cfg.ConfigDir, want)
	}
	if cfg.AbsConfigPath == "" || filepath.Base(cfg.AbsConfigPath) != "runtime-config.stack.yaml" {
		t.Fatalf("AbsConfigPath = %q", cfg.AbsConfigPath)
	}
}

func TestStatusDatabaseDefaultsBesideAuthDatabase(t *testing.T) {
	dataDir := t.TempDir()
	cfg := &Config{AuthDBPath: filepath.Join(dataDir, "auth.db")}
	resolveDashboardStatePaths(cfg)
	if want := filepath.Join(dataDir, "status.sqlite"); cfg.StatusDBPath != want {
		t.Fatalf("StatusDBPath = %q, want %q", cfg.StatusDBPath, want)
	}

	explicit := filepath.Join(t.TempDir(), "service-history.sqlite")
	cfg.StatusDBPath = explicit
	resolveDashboardStatePaths(cfg)
	if cfg.StatusDBPath != explicit {
		t.Fatalf("explicit StatusDBPath = %q, want %q", cfg.StatusDBPath, explicit)
	}
}
