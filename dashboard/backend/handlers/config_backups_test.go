package handlers

import (
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

	if _, err := createConfigBackup(configDir, []byte("providers:\n  models: []\n")); err != nil {
		t.Fatalf("createConfigBackup: %v", err)
	}

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

	if _, err := snapshotCurrentConfigBeforeRollback(configPath, configDir); err != nil {
		t.Fatalf("snapshotCurrentConfigBeforeRollback: %v", err)
	}

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

	if _, err := createConfigBackup(configDir, []byte("version: v0.3\n")); err != nil {
		t.Fatalf("createConfigBackup: %v", err)
	}

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

// The review case: a snapshot must not be written through the existing inode.
// Anything holding the old file open must keep seeing the old bytes, and the
// new bytes must land on a fresh 0600 inode.
func TestWriteConfigSnapshotReplacesInodeAndNeverExposesNewBytes(t *testing.T) {
	skipWithoutPOSIXModes(t)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("api_key: old\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	beforeInfo, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	// A reader that opened the world-readable file before the write.
	stale, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer stale.Close()

	if writeErr := writeConfigSnapshot(path, []byte("api_key: new-secret\n")); writeErr != nil {
		t.Fatalf("writeConfigSnapshot: %v", writeErr)
	}

	staleBytes, err := io.ReadAll(stale)
	if err != nil {
		t.Fatalf("read stale descriptor: %v", err)
	}
	if strings.Contains(string(staleBytes), "new-secret") {
		t.Fatalf("descriptor opened before the write saw the new credential: %q", staleBytes)
	}

	afterInfo, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat after: %v", err)
	}
	if os.SameFile(beforeInfo, afterInfo) {
		t.Fatal("snapshot reused the original inode instead of replacing it")
	}
	if got := filePerm(t, path); got != configSnapshotFileMode {
		t.Fatalf("mode = %04o, want %04o", got, configSnapshotFileMode)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(data) != "api_key: new-secret\n" {
		t.Fatalf("content = %q", data)
	}
	if leftovers := tempSnapshotLeftovers(t, filepath.Dir(path)); len(leftovers) != 0 {
		t.Fatalf("temp files left behind: %v", leftovers)
	}
}

func tempSnapshotLeftovers(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	var found []string
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".tmp") {
			found = append(found, entry.Name())
		}
	}
	return found
}

// A planted symlink must not redirect credentials out of the snapshot directory.
func TestWriteConfigSnapshotRejectsSymlinkTarget(t *testing.T) {
	skipWithoutPOSIXModes(t)
	dir := t.TempDir()
	outside := filepath.Join(dir, "outside.yaml")
	if err := os.WriteFile(outside, []byte("untouched\n"), 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}
	link := filepath.Join(dir, "config.20200101-000000.yaml")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if err := writeConfigSnapshot(link, []byte("api_key: secret\n")); err == nil {
		t.Fatal("expected a symlink target to be refused")
	}

	data, err := os.ReadFile(outside)
	if err != nil {
		t.Fatalf("read outside: %v", err)
	}
	if string(data) != "untouched\n" {
		t.Fatalf("symlink target was written through: %q", data)
	}
}

func TestWriteConfigSnapshotRejectsNonRegularTarget(t *testing.T) {
	skipWithoutPOSIXModes(t)
	dir := t.TempDir()
	target := filepath.Join(dir, "config.20200101-000000.yaml")
	if err := os.Mkdir(target, 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := writeConfigSnapshot(target, []byte("api_key: secret\n")); err == nil {
		t.Fatal("expected a directory target to be refused")
	}
}

func TestEnsureConfigSnapshotDirRejectsSymlink(t *testing.T) {
	skipWithoutPOSIXModes(t)
	dir := t.TempDir()
	real := filepath.Join(dir, "real")
	if err := os.Mkdir(real, 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if err := ensureConfigSnapshotDir(link); err == nil {
		t.Fatal("expected a symlinked snapshot directory to be refused")
	}
}

// Fail closed: an unsafe backup directory must abort before any snapshot lands.
func TestCreateConfigBackupFailsClosedOnSymlinkedDir(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	elsewhere := t.TempDir()
	if err := os.MkdirAll(filepath.Join(configDir, ".vllm-sr"), 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.Symlink(elsewhere, configBackupDir(configDir)); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	version, err := createConfigBackup(configDir, []byte("api_key: secret\n"))
	if err == nil {
		t.Fatal("expected createConfigBackup to fail closed")
	}
	if version != "" {
		t.Fatalf("version = %q, want empty on failure", version)
	}

	entries, readErr := os.ReadDir(elsewhere)
	if readErr != nil {
		t.Fatalf("read symlink destination: %v", readErr)
	}
	if len(entries) != 0 {
		t.Fatalf("wrote %d file(s) through the symlinked directory", len(entries))
	}
}

// A backup directory this process cannot write must abort, not proceed.
func TestCreateConfigBackupFailsClosedOnUnwritableDir(t *testing.T) {
	skipWithoutPOSIXModes(t)
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory permission checks")
	}
	configDir := t.TempDir()
	parent := filepath.Join(configDir, ".vllm-sr")
	if err := os.MkdirAll(parent, 0o500); err != nil {
		t.Fatalf("seed: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(parent, 0o700) })

	if _, err := createConfigBackup(configDir, []byte("api_key: secret\n")); err == nil {
		t.Fatal("expected createConfigBackup to fail closed on an unwritable parent")
	}
}

// config.dsl sits beside the backup directory, and a deploy carrying no DSL
// never rewrites it, so the repair pass has to reach it explicitly.
func TestRepairTightensLegacyDSLSnapshot(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(configDir, ".vllm-sr"), 0o755); err != nil {
		t.Fatalf("seed: %v", err)
	}
	dslPath := archivedDSLPath(configDir)
	if err := os.WriteFile(dslPath, []byte("api_key: leaked\n"), 0o644); err != nil {
		t.Fatalf("seed dsl: %v", err)
	}

	// A deploy with no DSL payload: archiveDeployDSL returns early.
	if _, err := createConfigBackup(configDir, []byte("version: v0.3\n")); err != nil {
		t.Fatalf("createConfigBackup: %v", err)
	}

	if got := filePerm(t, dslPath); got != configSnapshotFileMode {
		t.Fatalf("legacy config.dsl mode = %04o, want %04o", got, configSnapshotFileMode)
	}
}

// os.Chmod follows symlinks, so the repair must skip them entirely.
func TestRepairSkipsSymlinkedDSLSnapshot(t *testing.T) {
	skipWithoutPOSIXModes(t)
	configDir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.yaml")
	if err := os.WriteFile(outside, []byte("untouched\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(configDir, ".vllm-sr"), 0o755); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.Symlink(outside, archivedDSLPath(configDir)); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if _, err := createConfigBackup(configDir, []byte("version: v0.3\n")); err != nil {
		t.Fatalf("createConfigBackup: %v", err)
	}

	if got := filePerm(t, outside); got != 0o644 {
		t.Fatalf("symlink destination mode = %04o, want it untouched at 0644", got)
	}
}

// Only a missing file means "no config". Any other read error must abort so a
// deploy or rollback cannot overwrite an unreadable live config unprotected.
func TestReadLiveConfigSeparatesAbsenceFromFailure(t *testing.T) {
	skipWithoutPOSIXModes(t)
	if os.Geteuid() == 0 {
		t.Skip("root bypasses file permission checks")
	}
	dir := t.TempDir()

	missing, err := readLiveConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatalf("a missing config must not be an error: %v", err)
	}
	if missing != nil {
		t.Fatalf("missing config data = %q, want nil", missing)
	}

	unreadable := filepath.Join(dir, "unreadable.yaml")
	if err := os.WriteFile(unreadable, []byte("api_key: secret\n"), 0o000); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if _, err := readLiveConfig(unreadable); err == nil {
		t.Fatal("an unreadable config must be reported, not treated as absent")
	}
}

func TestSnapshotBeforeRollbackAbortsOnUnreadableConfig(t *testing.T) {
	skipWithoutPOSIXModes(t)
	if os.Geteuid() == 0 {
		t.Skip("root bypasses file permission checks")
	}
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("api_key: secret\n"), 0o000); err != nil {
		t.Fatalf("seed: %v", err)
	}

	data, err := snapshotCurrentConfigBeforeRollback(configPath, configDir)
	if err == nil {
		t.Fatal("expected rollback to abort on an unreadable live config")
	}
	if data != nil {
		t.Fatalf("data = %q, want nil on failure", data)
	}
}

// The rename is only durable once the parent directory entry is synced.
func TestWriteConfigSnapshotSyncsParentDirectory(t *testing.T) {
	skipWithoutPOSIXModes(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "config.20200101-000000.yaml")

	if err := writeConfigSnapshot(path, []byte("api_key: secret\n")); err != nil {
		t.Fatalf("writeConfigSnapshot: %v", err)
	}
	if err := syncSnapshotDirectory(dir); err != nil {
		t.Fatalf("syncSnapshotDirectory: %v", err)
	}

	// A directory that no longer exists cannot be synced, so the failure the
	// write path propagates is a real one rather than a silent success.
	if err := syncSnapshotDirectory(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("expected syncSnapshotDirectory to report a missing directory")
	}
}
