package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestConfigBackupAndDSLArchiveUsePrivateModes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits are required")
	}
	configDir := t.TempDir()
	version := createConfigBackup(configDir, []byte("version: private\n"))
	backupDir := configBackupDir(configDir)
	backupFile := filepath.Join(backupDir, "config."+version+".yaml")
	archiveDeployDSL(configDir, "ROUTE private {}")
	dslFile := filepath.Join(configDir, ".vllm-sr", "config.dsl")

	assertFileMode(t, backupDir, 0o700)
	assertFileMode(t, backupFile, 0o600)
	assertFileMode(t, dslFile, 0o600)
}

func TestPrivateConfigStorageRejectsSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic-link behavior differs on Windows")
	}
	t.Run("backup directory", func(t *testing.T) {
		configDir := t.TempDir()
		sharedDir := filepath.Join(configDir, ".vllm-sr")
		if err := os.MkdirAll(sharedDir, 0o755); err != nil {
			t.Fatal(err)
		}
		external := t.TempDir()
		if err := os.Symlink(external, filepath.Join(sharedDir, "config-backups")); err != nil {
			t.Fatal(err)
		}
		createConfigBackup(configDir, []byte("must-not-follow: true\n"))
		entries, err := os.ReadDir(external)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("backup write followed symlink: %v", entries)
		}
	})

	t.Run("backup parent", func(t *testing.T) {
		configDir := t.TempDir()
		external := t.TempDir()
		if err := os.Mkdir(filepath.Join(external, "config-backups"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(external, filepath.Join(configDir, ".vllm-sr")); err != nil {
			t.Fatal(err)
		}
		createConfigBackup(configDir, []byte("must-not-follow-parent: true\n"))
		entries, err := os.ReadDir(external)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 || entries[0].Name() != "config-backups" {
			t.Fatalf("backup write followed parent symlink: %v", entries)
		}
		backupEntries, err := os.ReadDir(filepath.Join(external, "config-backups"))
		if err != nil {
			t.Fatal(err)
		}
		if len(backupEntries) != 0 {
			t.Fatalf("backup write followed parent symlink into existing directory: %v", backupEntries)
		}
	})

	t.Run("dsl target", func(t *testing.T) {
		configDir := t.TempDir()
		sharedDir := filepath.Join(configDir, ".vllm-sr")
		if err := os.MkdirAll(sharedDir, 0o755); err != nil {
			t.Fatal(err)
		}
		external := filepath.Join(t.TempDir(), "external.dsl")
		if err := os.WriteFile(external, []byte("original"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(external, filepath.Join(sharedDir, "config.dsl")); err != nil {
			t.Fatal(err)
		}
		archiveDeployDSL(configDir, "replacement")
		got, err := os.ReadFile(external)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != "original" {
			t.Fatalf("DSL archive followed symlink, external content = %q", got)
		}
		if archived := readArchivedDSL(configDir); archived != "" {
			t.Fatalf("DSL read followed symlink and returned %q", archived)
		}
	})
}

func TestRollbackRejectsTraversalVersionBeforeFilesystemAccess(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	original := []byte("version: original\n")
	if err := os.WriteFile(configPath, original, 0o600); err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]string{"version": "../../sentinel"})
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(body))
	RollbackHandler(configPath, false, configDir)(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	got, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, original) {
		t.Fatalf("invalid rollback mutated config: %q", got)
	}
}

func assertFileMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if got := info.Mode().Perm(); got != want {
		t.Fatalf("mode(%s) = %o, want %o", path, got, want)
	}
}
