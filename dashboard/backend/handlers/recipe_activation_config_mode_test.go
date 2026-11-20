package handlers

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// `vllm-sr serve` reads the config an activation publishes as its own user,
// not the Dashboard's, so the file carries the mode of every config the
// Dashboard saves.
func TestActivatedConfigIsReadableByTheServeUser(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX file modes")
	}
	path := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := writeActivationConfig(path, []byte("version: v0.5\n")); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o644 {
		t.Fatalf("active config mode = %o, want 644", got)
	}
}

// An activation that replaces a private config keeps it private.
func TestActivationKeepsTheConfigMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX file modes")
	}
	path := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(path, []byte("version: v0.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeActivationConfig(path, []byte("version: v0.5\nrecipes: []\n")); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("active config mode = %o, want 600", got)
	}
}
