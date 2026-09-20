package router

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// writeConfig lays out a checkout-shaped tree: <root>/config/config.yaml plus
// the tools database where the repository actually ships it, and returns the
// dashboard config the router would be built with.
func writeConfig(t *testing.T, body string) (*config.Config, string) {
	t.Helper()
	root := t.TempDir()
	configDir := filepath.Join(root, "config")
	dbDir := filepath.Join(configDir, "runtime", "tools")
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dbDir, "tools_db.json"), []byte("[]"), 0o644); err != nil {
		t.Fatalf("write db: %v", err)
	}
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return &config.Config{ConfigDir: configDir, AbsConfigPath: configPath, ConfigBaseDir: root}, root
}

const toolsConfig = `global:
  integrations:
    tools:
      enabled: true
      tools_db_path: config/runtime/tools/tools_db.json
`

// A configured tools_db_path is relative to the project root, which is how
// config/config.yaml spells it. Returning it unchanged left it to resolve
// against the process working directory — dashboard/backend for the documented
// dev launch — where the file does not exist, so /api/tools-db answered 404 in
// a default checkout.
func TestResolveToolsDBPathResolvesRelativeConfiguredPathAgainstProjectRoot(t *testing.T) {
	t.Parallel()

	cfg, root := writeConfig(t, toolsConfig)
	got := resolveToolsDBPath(cfg)

	want := filepath.Join(root, "config", "runtime", "tools", "tools_db.json")
	if got != want {
		t.Errorf("resolveToolsDBPath() = %q, want %q", got, want)
	}
	if !filepath.IsAbs(got) {
		t.Errorf("resolveToolsDBPath() = %q, want an absolute path", got)
	}
	if _, err := os.Stat(got); err != nil {
		t.Errorf("resolved path does not exist: %v", err)
	}
}

// With tools_db_path absent the router's canonical default applies, so the
// value reaching this function is the relative "config/tools_db.json" rather
// than an empty string. It must still come back anchored to the project root
// and must not repeat the config directory.
func TestResolveToolsDBPathDoesNotDoubleTheConfigDirectory(t *testing.T) {
	t.Parallel()

	cfg, root := writeConfig(t, "global:\n  integrations:\n    tools:\n      enabled: true\n")
	got := resolveToolsDBPath(cfg)

	if doubled := filepath.Join(cfg.ConfigDir, "config"); strings.HasPrefix(got, doubled) {
		t.Errorf("resolveToolsDBPath() = %q, which repeats the config directory %q", got, doubled)
	}
	if want := filepath.Join(root, "config", "tools_db.json"); got != want {
		t.Errorf("resolveToolsDBPath() = %q, want %q", got, want)
	}
}

// An unparsable config takes the error branch. It must yield the router's own
// default made absolute, not the doubled path — that branch is precisely what a
// user sees when their config is broken, so it should not compound the problem.
func TestResolveToolsDBPathFallbackIsUsedWhenTheConfigWillNotParse(t *testing.T) {
	t.Parallel()

	cfg, root := writeConfig(t, "global: [this is not a mapping\n")
	got := resolveToolsDBPath(cfg)

	if want := filepath.Join(root, defaultToolsDBPath); got != want {
		t.Errorf("resolveToolsDBPath() = %q, want %q", got, want)
	}
	if doubled := filepath.Join(cfg.ConfigDir, "config"); strings.HasPrefix(got, doubled) {
		t.Errorf("resolveToolsDBPath() = %q, which repeats the config directory", got)
	}
}

// An absolute configured path is already unambiguous. Rewriting it would break
// a deployment that points the database outside the checkout.
func TestResolveToolsDBPathLeavesAnAbsoluteConfiguredPathAlone(t *testing.T) {
	t.Parallel()

	outside := filepath.Join(t.TempDir(), "elsewhere", "tools_db.json")
	if err := os.MkdirAll(filepath.Dir(outside), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(outside, []byte("[]"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	cfg, _ := writeConfig(t, "global:\n  integrations:\n    tools:\n      enabled: true\n      tools_db_path: "+outside+"\n")

	if got := resolveToolsDBPath(cfg); got != outside {
		t.Errorf("resolveToolsDBPath() = %q, want the configured absolute path %q", got, outside)
	}
}
