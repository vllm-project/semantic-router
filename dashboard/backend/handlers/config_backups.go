package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// config.yaml can carry plaintext provider credentials, so every copy the
// Dashboard writes is readable only by the user it runs as.
const (
	configSnapshotDirMode  os.FileMode = 0o700
	configSnapshotFileMode os.FileMode = 0o600
)

func configBackupDir(configDir string) string {
	return filepath.Join(configDir, ".vllm-sr", "config-backups")
}

// MkdirAll applies its mode only when it creates the directory, so an existing
// one left world-readable by an earlier build needs the explicit Chmod.
func ensureConfigSnapshotDir(dir string) error {
	if err := os.MkdirAll(dir, configSnapshotDirMode); err != nil {
		return err
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("config snapshot directory %s is a symlink", dir)
	}
	if info.Mode().Perm() == configSnapshotDirMode {
		return nil
	}
	return os.Chmod(dir, configSnapshotDirMode)
}

// A snapshot never overwrites in place: the old inode keeps its old mode until a
// chmod lands, and descriptors already open on it keep reading. Write a fresh
// 0600 file and rename it over the target instead.
func writeConfigSnapshot(path string, data []byte) error {
	if err := rejectUnsafeSnapshotTarget(path); err != nil {
		return err
	}
	temp, err := createConfigSnapshotTemp(path)
	if err != nil {
		return err
	}
	if err := writeAndCloseSnapshotTemp(temp, data); err != nil {
		_ = os.Remove(temp.Name())
		return err
	}
	if err := os.Rename(temp.Name(), path); err != nil {
		_ = os.Remove(temp.Name())
		return err
	}
	// The rename is not durable until the directory entry is synced, so a crash
	// here would drop the restore point the caller is about to rely on.
	return syncSnapshotDirectory(filepath.Dir(path))
}

func syncSnapshotDirectory(dir string) error {
	directory, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	return directory.Sync()
}

// Lstat never follows the final element, so a symlink fails IsRegular here and a
// planted path cannot redirect credentials outside the snapshot directory.
func rejectUnsafeSnapshotTarget(path string) error {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("config snapshot target %s is not a regular file", path)
	}
	return nil
}

// O_EXCL with an unpredictable name means the descriptor is always a file this
// process just created at 0600, never one that already existed.
func createConfigSnapshotTemp(path string) (*os.File, error) {
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		return nil, err
	}
	temp := filepath.Join(
		filepath.Dir(path),
		"."+filepath.Base(path)+"."+hex.EncodeToString(random)+".tmp",
	)
	return os.OpenFile(temp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, configSnapshotFileMode)
}

func writeAndCloseSnapshotTemp(temp *os.File, data []byte) error {
	_, err := temp.Write(data)
	if err == nil {
		err = temp.Sync()
	}
	if closeErr := temp.Close(); err == nil {
		err = closeErr
	}
	return err
}

// Shared by listing, cleanup and permission repair so they cannot disagree.
func isConfigBackupEntry(entry os.DirEntry) bool {
	return !entry.IsDir() &&
		strings.HasPrefix(entry.Name(), "config.") &&
		strings.HasSuffix(entry.Name(), ".yaml")
}

func archivedDSLPath(configDir string) string {
	return filepath.Join(configDir, ".vllm-sr", "config.dsl")
}

// Restricts snapshots an earlier build left world-readable, so an upgrade does
// not keep leaking them until they rotate out. Best effort: never fails a deploy.
//
// config.dsl is repaired here too. It lives beside the backup directory rather
// than in it, and a deploy that carries no DSL never rewrites it, so nothing
// else would ever tighten an upgraded install's copy.
func repairConfigSnapshotPermissions(configDir string) {
	backupDir := configBackupDir(configDir)
	entries, err := os.ReadDir(backupDir)
	if err == nil {
		for _, entry := range entries {
			if !isConfigBackupEntry(entry) {
				continue
			}
			repairSnapshotFilePermissions(filepath.Join(backupDir, entry.Name()))
		}
	}
	repairSnapshotFilePermissions(archivedDSLPath(configDir))
}

// Lstat keeps a symlink out of the chmod: os.Chmod follows links, so a planted
// one would let this widen or narrow a file outside the snapshot tree.
func repairSnapshotFilePermissions(path string) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return
	}
	if info.Mode().Perm() == configSnapshotFileMode {
		return
	}
	if err := os.Chmod(path, configSnapshotFileMode); err != nil {
		log.Printf("Warning: failed to restrict permissions on config snapshot %s: %v", path, err)
		return
	}
	log.Printf("Restricted permissions on pre-existing config snapshot: %s", path)
}

// Fails closed: an unsafe or unfixable backup directory means the credentials in
// config.yaml would land somewhere this process cannot keep owner-only, so
// nothing is written and the caller aborts rather than losing the restore point.
func createConfigBackup(configDir string, existingData []byte) (string, error) {
	backupDir := configBackupDir(configDir)
	if err := ensureConfigSnapshotDir(backupDir); err != nil {
		return "", fmt.Errorf("prepare config backup directory: %w", err)
	}
	repairConfigSnapshotPermissions(configDir)

	version := time.Now().Format("20060102-150405")
	if len(existingData) == 0 {
		return version, nil
	}

	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := writeConfigSnapshot(backupFile, existingData); err != nil {
		return "", fmt.Errorf("write config backup: %w", err)
	}
	log.Printf("[Deploy] Config backup created: %s", backupFile)

	return version, nil
}

func readArchivedDSL(configDir string) string {
	data, err := os.ReadFile(archivedDSLPath(configDir))
	if err != nil {
		return ""
	}
	return string(data)
}

func archiveDeployDSL(configDir string, dsl string) {
	if strings.TrimSpace(dsl) == "" {
		return
	}

	dslDir := filepath.Join(configDir, ".vllm-sr")
	if err := os.MkdirAll(dslDir, 0o755); err != nil {
		log.Printf("Warning: failed to create DSL archive directory: %v", err)
		return
	}

	// The DSL compiles into config.yaml and can carry the same credentials. Its
	// directory keeps its mode: other services read siblings out of it.
	if err := writeConfigSnapshot(archivedDSLPath(configDir), []byte(dsl)); err != nil {
		log.Printf("Warning: failed to archive DSL source: %v", err)
	}
}

// Only a missing file means "no existing config". Collapsing every read error
// into empty data would let a deploy or rollback overwrite an unreadable live
// config with no snapshot, and the nil result also disables the runtime restore.
func readLiveConfig(configPath string) ([]byte, error) {
	data, err := os.ReadFile(configPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read current config %s: %w", configPath, err)
	}
	return data, nil
}

func readConfigBackup(configDir string, version string) ([]byte, error) {
	backupFile := filepath.Join(configBackupDir(configDir), fmt.Sprintf("config.%s.yaml", version))
	return os.ReadFile(backupFile)
}

// Fails closed for the same reason as createConfigBackup.
func snapshotCurrentConfigBeforeRollback(configPath string, configDir string) ([]byte, error) {
	existingData, err := readLiveConfig(configPath)
	if err != nil {
		return nil, err
	}
	if len(existingData) == 0 {
		return existingData, nil
	}

	backupDir := configBackupDir(configDir)
	if err := ensureConfigSnapshotDir(backupDir); err != nil {
		return nil, fmt.Errorf("prepare config backup directory: %w", err)
	}
	repairConfigSnapshotPermissions(configDir)

	currentVersion := time.Now().Format("20060102-150405")
	preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
	if err := writeConfigSnapshot(preRollbackFile, existingData); err != nil {
		return nil, fmt.Errorf("snapshot current config before rollback: %w", err)
	}

	return existingData, nil
}

func versionsLocalList(w http.ResponseWriter, configPath string) {
	versions, err := listConfigVersions(configPath)
	if err != nil {
		versions = []ConfigVersion{}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(versions)
}

func listConfigVersions(configPath string) ([]ConfigVersion, error) {
	backupDir := configBackupDir(filepath.Dir(configPath))
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return nil, err
	}

	versions := []ConfigVersion{}
	for _, entry := range entries {
		if !isConfigBackupEntry(entry) {
			continue
		}

		versionStr := strings.TrimPrefix(entry.Name(), "config.")
		versionStr = strings.TrimSuffix(versionStr, ".yaml")

		timestamp := versionStr
		if t, parseErr := time.Parse("20060102-150405", versionStr); parseErr == nil {
			timestamp = t.Format("2006-01-02 15:04:05")
		}

		versions = append(versions, ConfigVersion{
			Version:   versionStr,
			Timestamp: timestamp,
			Source:    "dsl",
			Filename:  entry.Name(),
		})
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].Version > versions[j].Version
	})

	return versions, nil
}

// cleanupBackups removes old backups beyond maxBackups
func cleanupBackups(backupDir string) {
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return
	}

	var backups []os.DirEntry
	for _, entry := range entries {
		if isConfigBackupEntry(entry) {
			backups = append(backups, entry)
		}
	}

	if len(backups) <= maxBackups {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})

	toRemove := len(backups) - maxBackups
	for i := 0; i < toRemove; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			log.Printf("Warning: failed to remove old backup %s: %v", path, err)
		} else {
			log.Printf("Removed old backup: %s", backups[i].Name())
		}
	}
}
