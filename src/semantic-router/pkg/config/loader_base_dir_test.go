package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseUsesAbsoluteConfigBaseDirOverride(t *testing.T) {
	root := t.TempDir()
	stateDir := filepath.Join(root, ".vllm-sr")
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(stateDir, "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte(entrypointRulesYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ConfigBaseDirEnv, root)

	cfg, err := testAuthoringParser(t).Parse(configPath)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if cfg.ConfigBaseDir != root {
		t.Fatalf("ConfigBaseDir = %q, want %q", cfg.ConfigBaseDir, root)
	}
}

func TestParseRejectsRelativeConfigBaseDirOverride(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(entrypointRulesYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ConfigBaseDirEnv, "relative-assets")

	_, err := testAuthoringParser(t).Parse(configPath)
	if err == nil || !strings.Contains(err.Error(), "must be an absolute directory") {
		t.Fatalf("Parse() error = %v, want absolute-directory error", err)
	}
}

func TestParseYAMLBytesDoesNotConsultConfigBaseDirOverride(t *testing.T) {
	t.Setenv(ConfigBaseDirEnv, "/path/that/does/not/exist")

	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML))
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes() error = %v", err)
	}
	if cfg.ConfigBaseDir != "" {
		t.Fatalf("ConfigBaseDir = %q, want empty for in-memory parsing", cfg.ConfigBaseDir)
	}
}
