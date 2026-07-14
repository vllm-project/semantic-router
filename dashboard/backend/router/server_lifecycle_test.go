package router

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestServerCloseReleasesAuthAndOtherOwnedStores(t *testing.T) {
	t.Parallel()

	cfg := lifecycleTestConfig(t)
	server, err := Setup(cfg)
	if err != nil {
		t.Fatalf("Setup() error = %v", err)
	}
	if _, err := server.auth.CanBootstrap(context.Background()); err != nil {
		t.Fatalf("auth store unavailable before Close(): %v", err)
	}
	if err := server.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := server.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
	if _, err := server.auth.CanBootstrap(context.Background()); err == nil {
		t.Fatal("auth store remained usable after Server.Close()")
	}
}

func TestSetupReturnsWorkflowInitializationErrorWithoutExiting(t *testing.T) {
	t.Parallel()

	cfg := lifecycleTestConfig(t)
	cfg.WorkflowDBPath = t.TempDir() // A directory cannot be opened as SQLite.
	server, err := Setup(cfg)
	if server != nil {
		t.Fatal("Setup() returned a server after workflow initialization failed")
	}
	if err == nil {
		t.Fatal("Setup() did not return the workflow initialization failure")
	}
}

func lifecycleTestConfig(t *testing.T) *config.Config {
	t.Helper()
	dir := t.TempDir()
	staticDir := filepath.Join(dir, "static")
	if err := os.MkdirAll(staticDir, 0o755); err != nil {
		t.Fatalf("create static dir: %v", err)
	}
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return &config.Config{
		AuthDBPath:             filepath.Join(dir, "auth", "auth.db"),
		JWTSecret:              "0123456789abcdef0123456789abcdef",
		JWTExpiryHours:         1,
		WorkflowDBPath:         filepath.Join(dir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(dir, "projection.sqlite"),
		OpenClawDataDir:        filepath.Join(dir, "openclaw"),
		StaticDir:              staticDir,
		ConfigFile:             configPath,
		AbsConfigPath:          configPath,
		ConfigDir:              dir,
		RouterAPIURL:           "http://127.0.0.1:8080",
		RouterMetrics:          "http://127.0.0.1:9190/metrics",
	}
}
