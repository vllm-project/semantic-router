package handlers

import (
	"encoding/json"
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

// os.WriteFile applies its mode only when it creates the file, and backup names
// are timestamped to the second, so a reused path needs the explicit Chmod.
func writeConfigSnapshot(path string, data []byte) error {
	if err := os.WriteFile(path, data, configSnapshotFileMode); err != nil {
		return err
	}
	return os.Chmod(path, configSnapshotFileMode)
}

// Shared by listing, cleanup and permission repair so they cannot disagree.
func isConfigBackupEntry(entry os.DirEntry) bool {
	return !entry.IsDir() &&
		strings.HasPrefix(entry.Name(), "config.") &&
		strings.HasSuffix(entry.Name(), ".yaml")
}

// Restricts snapshots an earlier build left world-readable, so an upgrade does
// not keep leaking them until they rotate out. Best effort: never fails a deploy.
func repairConfigSnapshotPermissions(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if !isConfigBackupEntry(entry) {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		if info.Mode().Perm() == configSnapshotFileMode {
			continue
		}
		if err := os.Chmod(path, configSnapshotFileMode); err != nil {
			log.Printf("Warning: failed to restrict permissions on config backup %s: %v", path, err)
			continue
		}
		log.Printf("Restricted permissions on pre-existing config backup: %s", path)
	}
}

func createConfigBackup(configDir string, existingData []byte) string {
	backupDir := configBackupDir(configDir)
	if err := ensureConfigSnapshotDir(backupDir); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
	}
	repairConfigSnapshotPermissions(backupDir)

	version := time.Now().Format("20060102-150405")
	if len(existingData) == 0 {
		return version
	}

	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := writeConfigSnapshot(backupFile, existingData); err != nil {
		log.Printf("Warning: failed to create backup: %v", err)
	} else {
		log.Printf("[Deploy] Config backup created: %s", backupFile)
	}

	return version
}

func readArchivedDSL(configDir string) string {
	dslFile := filepath.Join(configDir, ".vllm-sr", "config.dsl")
	data, err := os.ReadFile(dslFile)
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
	dslFile := filepath.Join(dslDir, "config.dsl")
	if err := writeConfigSnapshot(dslFile, []byte(dsl)); err != nil {
		log.Printf("Warning: failed to archive DSL source: %v", err)
	}
}

func readConfigBackup(configDir string, version string) ([]byte, error) {
	backupFile := filepath.Join(configBackupDir(configDir), fmt.Sprintf("config.%s.yaml", version))
	return os.ReadFile(backupFile)
}

func snapshotCurrentConfigBeforeRollback(configPath string, configDir string) []byte {
	existingData, err := os.ReadFile(configPath)
	if err != nil || len(existingData) == 0 {
		return existingData
	}

	backupDir := configBackupDir(configDir)
	if err := ensureConfigSnapshotDir(backupDir); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
		return existingData
	}
	repairConfigSnapshotPermissions(backupDir)

	currentVersion := time.Now().Format("20060102-150405")
	preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
	if err := writeConfigSnapshot(preRollbackFile, existingData); err != nil {
		log.Printf("Warning: failed to snapshot current config before rollback: %v", err)
	}

	return existingData
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
