package startupstatus

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStatusPathFromConfigPathUsesConfigDirWhenWritable(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")

	got := StatusPathFromConfigPath(configPath)
	want := filepath.Join(configDir, "router-runtime.json")
	if got != want {
		t.Fatalf("StatusPathFromConfigPath() = %q, want %q", got, want)
	}
}

func TestStatusPathFromConfigPathFallsBackForReadOnlyDir(t *testing.T) {
	configDir := filepath.Join(t.TempDir(), "readonly")
	if err := os.MkdirAll(configDir, 0o555); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chmod(configDir, 0o755)
	})

	configPath := filepath.Join(configDir, "config.yaml")
	got := StatusPathFromConfigPath(configPath)
	wantSibling := filepath.Join(configDir, "router-runtime.json")
	if got == wantSibling {
		t.Fatalf("expected read-only config dir to use fallback path, got sibling %q", got)
	}
	if !strings.Contains(got, filepath.Join("vllm-sr", "runtime-status")) {
		t.Fatalf("expected fallback runtime status path under tempdir, got %q", got)
	}
}

func TestStatusPathFromConfigPathHonorsOverride(t *testing.T) {
	overrideDir := filepath.Join(t.TempDir(), "status")
	if err := os.Setenv("VLLM_SR_RUNTIME_STATUS_DIR", overrideDir); err != nil {
		t.Fatalf("Setenv() error = %v", err)
	}
	defer func() {
		_ = os.Unsetenv("VLLM_SR_RUNTIME_STATUS_DIR")
	}()

	got := StatusPathFromConfigPath("/app/config/config.yaml")
	want := filepath.Join(overrideDir, "router-runtime.json")
	if got != want {
		t.Fatalf("StatusPathFromConfigPath() with override = %q, want %q", got, want)
	}
}
