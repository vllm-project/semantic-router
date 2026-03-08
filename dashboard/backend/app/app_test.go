package app

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestNewInitializesConsoleStores(t *testing.T) {
	cfg := &config.Config{
		ConsoleStoreBackend: "sqlite",
		ConsoleDBPath:       filepath.Join(t.TempDir(), "console.db"),
	}

	application, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer func() {
		_ = application.Close()
	}()

	if application.Config != cfg {
		t.Fatalf("expected app to retain config pointer")
	}
	if application.Console == nil || application.Console.Users == nil || application.Console.Revisions == nil {
		t.Fatalf("expected console stores to be initialized")
	}
	if application.ConfigLifecycle == nil {
		t.Fatalf("expected config lifecycle service to be initialized")
	}
}

func TestNewRejectsNilConfig(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Fatal("expected nil config to be rejected")
	}
}
