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
