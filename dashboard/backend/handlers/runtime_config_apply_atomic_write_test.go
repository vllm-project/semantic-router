package handlers

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteConfigAtomicallySucceeds(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")

	if err := writeConfigAtomically(configPath, []byte("routing: {}\n")); err != nil {
		t.Fatalf("writeConfigAtomically: %v", err)
	}

	got, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read configPath: %v", err)
	}
	if string(got) != "routing: {}\n" {
		t.Fatalf("configPath content = %q, want %q", got, "routing: {}\n")
	}
	if _, err := os.Stat(configPath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("expected .tmp file to be removed after a successful write, stat err = %v", err)
	}
}

// TestWriteConfigAtomicallyRenameFailureLeavesExistingConfigUntouched pins the
// invariant a failed rename must never fall back to a non-atomic direct
// write: it forces the rename step to fail (standing in for the EBUSY/EROFS
// a K8s ConfigMap subPath mount returns) and asserts configPath keeps its
// original content rather than being overwritten by a partial or full
// direct write.
func TestWriteConfigAtomicallyRenameFailureLeavesExistingConfigUntouched(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	original := []byte("routing: {original: true}\n")
	if err := os.WriteFile(configPath, original, 0o644); err != nil {
		t.Fatalf("seed configPath: %v", err)
	}

	renameErr := errors.New("simulated rename failure: device or resource busy")
	old := atomicRename
	atomicRename = func(string, string) error { return renameErr }
	t.Cleanup(func() { atomicRename = old })

	err := writeConfigAtomically(configPath, []byte("routing: {new: true}\n"))
	if !errors.Is(err, renameErr) {
		t.Fatalf("writeConfigAtomically error = %v, want %v", err, renameErr)
	}

	got, readErr := os.ReadFile(configPath)
	if readErr != nil {
		t.Fatalf("read configPath after failed write: %v", readErr)
	}
	if string(got) != string(original) {
		t.Fatalf("configPath was modified by a failed atomic write: got %q, want unchanged %q", got, original)
	}
	if _, statErr := os.Stat(configPath + ".tmp"); !os.IsNotExist(statErr) {
		t.Fatalf("expected .tmp file to be cleaned up after a failed rename, stat err = %v", statErr)
	}
}

func TestWriteConfigAtomicallyTempWriteFailureCleansUp(t *testing.T) {
	dir := t.TempDir()
	// A directory in place of configPath's parent forces the initial
	// temp-file open to fail (ENOTDIR/EISDIR on the ".tmp" path).
	badParent := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(badParent, []byte("x"), 0o644); err != nil {
		t.Fatalf("seed badParent: %v", err)
	}
	configPath := filepath.Join(badParent, "config.yaml")

	if err := writeConfigAtomically(configPath, []byte("routing: {}\n")); err == nil {
		t.Fatal("expected writeConfigAtomically to fail when the temp file cannot be created")
	}
}
