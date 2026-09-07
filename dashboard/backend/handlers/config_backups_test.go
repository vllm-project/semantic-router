package handlers

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// config.yaml can hold plaintext provider credentials, so no snapshot the
// Dashboard writes may be readable by another user.

func skipWithoutPOSIXModes(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("file mode bits are not enforced on Windows")
	}
}

func filePerm(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return info.Mode().Perm()
}

func soleBackupPath(t *testing.T, backupDir string) string {
	t.Helper()
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		t.Fatalf("read backup dir: %v", err)
	}
	var found []string
	for _, entry := range entries {
		if isConfigBackupEntry(entry) {
			found = append(found, filepath.Join(backupDir, entry.Name()))
		}
	}
	if len(found) != 1 {
		t.Fatalf("expected exactly one backup, found %d: %v", len(found), found)
	}
	return found[0]
}

func TestCreateConfigBackupWritesOwnerOnly(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()

	createConfigBackup(configDir, []byte("providers:\n  models: []\n"))

	backupDir := configBackupDir(configDir)
	if got := filePerm(t, backupDir); got != configSnapshotDirMode {
		t.Fatalf("backup dir mode = %04o, want %04o", got, configSnapshotDirMode)
	}
	if got := filePerm(t, soleBackupPath(t, backupDir)); got != configSnapshotFileMode {
		t.Fatalf("backup file mode = %04o, want %04o", got, configSnapshotFileMode)
	}
}

func TestSnapshotBeforeRollbackWritesOwnerOnly(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	snapshotCurrentConfigBeforeRollback(configPath, configDir)

	backupDir := configBackupDir(configDir)
	if got := filePerm(t, backupDir); got != configSnapshotDirMode {
		t.Fatalf("backup dir mode = %04o, want %04o", got, configSnapshotDirMode)
	}
	if got := filePerm(t, soleBackupPath(t, backupDir)); got != configSnapshotFileMode {
		t.Fatalf("backup file mode = %04o, want %04o", got, configSnapshotFileMode)
	}
}

func TestBackupCurrentConfigWritesOwnerOnly(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	if err := backupCurrentConfig(configPath, configDir); err != nil {
		t.Fatalf("backupCurrentConfig: %v", err)
	}

	backupDir := configBackupDir(configDir)
	if got := filePerm(t, backupDir); got != configSnapshotDirMode {
		t.Fatalf("backup dir mode = %04o, want %04o", got, configSnapshotDirMode)
	}
	if got := filePerm(t, soleBackupPath(t, backupDir)); got != configSnapshotFileMode {
		t.Fatalf("backup file mode = %04o, want %04o", got, configSnapshotFileMode)
	}
}

func TestArchiveDeployDSLWritesOwnerOnly(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()

	archiveDeployDSL(configDir, "route \"x\" {}\n")

	dslFile := filepath.Join(configDir, ".vllm-sr", "config.dsl")
	if got := filePerm(t, dslFile); got != configSnapshotFileMode {
		t.Fatalf("archived DSL mode = %04o, want %04o", got, configSnapshotFileMode)
	}
}

// An upgrade already has world-readable snapshots on disk; writing the next one
// must repair them too.
func TestCreateConfigBackupRepairsExistingSnapshots(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	backupDir := configBackupDir(configDir)
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("seed backup dir: %v", err)
	}
	legacy := filepath.Join(backupDir, "config.20200101-000000.yaml")
	if err := os.WriteFile(legacy, []byte("api_key: leaked\n"), 0o644); err != nil {
		t.Fatalf("seed legacy backup: %v", err)
	}
	unrelated := filepath.Join(backupDir, "notes.txt")
	if err := os.WriteFile(unrelated, []byte("keep my mode\n"), 0o644); err != nil {
		t.Fatalf("seed unrelated file: %v", err)
	}

	createConfigBackup(configDir, []byte("version: v0.3\n"))

	if got := filePerm(t, backupDir); got != configSnapshotDirMode {
		t.Fatalf("pre-existing dir mode = %04o, want %04o", got, configSnapshotDirMode)
	}
	if got := filePerm(t, legacy); got != configSnapshotFileMode {
		t.Fatalf("pre-existing backup mode = %04o, want %04o", got, configSnapshotFileMode)
	}
	if got := filePerm(t, unrelated); got != 0o644 {
		t.Fatalf("unrelated file mode = %04o, want it untouched at 0644", got)
	}
}

// A same-second rewrite reuses the path and must not inherit its old mode.
func TestWriteConfigSnapshotTightensExistingPath(t *testing.T) {
	skipWithoutPOSIXModes(t)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("old\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := writeConfigSnapshot(path, []byte("new\n")); err != nil {
		t.Fatalf("writeConfigSnapshot: %v", err)
	}

	if got := filePerm(t, path); got != configSnapshotFileMode {
		t.Fatalf("mode = %04o, want %04o", got, configSnapshotFileMode)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(data) != "new\n" {
		t.Fatalf("content = %q, want %q", data, "new\n")
	}
}

// .vllm-sr holds Envoy config and runtime state other containers read.
func TestEnsureConfigSnapshotDirLeavesParentAlone(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	parent := filepath.Join(configDir, ".vllm-sr")
	if err := os.MkdirAll(parent, 0o755); err != nil {
		t.Fatalf("seed parent: %v", err)
	}

	if err := ensureConfigSnapshotDir(configBackupDir(configDir)); err != nil {
		t.Fatalf("ensureConfigSnapshotDir: %v", err)
	}

	if got := filePerm(t, parent); got != 0o755 {
		t.Fatalf("parent mode = %04o, want it untouched at 0755", got)
	}
}
