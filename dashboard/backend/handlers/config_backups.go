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

func configBackupDir(configDir string) string {
	return filepath.Join(configDir, ".vllm-sr", "config-backups")
}

func createConfigBackup(configDir string, existingData []byte) string {
	version := time.Now().Format("20060102-150405")
	backupDir := configBackupDir(configDir)
	if err := ensurePrivateStateDirectory(backupDir); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
		return version
	}

	if len(existingData) == 0 {
		return version
	}

	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := writePrivateStateFile(backupFile, existingData); err != nil {
		log.Printf("Warning: failed to create backup: %v", err)
	} else {
		log.Printf("[Deploy] Config backup created: %s", backupFile)
	}

	return version
}

func readArchivedDSL(configDir string) string {
	dslFile := filepath.Join(configDir, ".vllm-sr", "config.dsl")
	data, err := readPrivateStateFile(dslFile)
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
	if err := ensureSharedStateDirectory(dslDir); err != nil {
		log.Printf("Warning: failed to create DSL archive directory: %v", err)
		return
	}

	dslFile := filepath.Join(dslDir, "config.dsl")
	if err := writePrivateStateFile(dslFile, []byte(dsl)); err != nil {
		log.Printf("Warning: failed to archive DSL source: %v", err)
	}
}

func readConfigBackup(configDir string, version string) ([]byte, error) {
	if !validConfigBackupVersion(version) {
		return nil, os.ErrNotExist
	}
	backupFile := filepath.Join(configBackupDir(configDir), fmt.Sprintf("config.%s.yaml", version))
	data, err := readPrivateStateFile(backupFile)
	if err != nil {
		return nil, fmt.Errorf("read config backup: %w", err)
	}
	return data, nil
}

func snapshotCurrentConfigBeforeRollback(configPath string, configDir string) (configFileSnapshot, error) {
	previous, err := captureConfigFileSnapshot(configPath)
	if err != nil || !previous.existed || len(previous.data) == 0 {
		return previous, err
	}

	backupDir := configBackupDir(configDir)
	if err := ensurePrivateStateDirectory(backupDir); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
		return previous, nil
	}

	currentVersion := time.Now().Format("20060102-150405")
	preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
	if err := writePrivateStateFile(preRollbackFile, previous.data); err != nil {
		log.Printf("Warning: failed to snapshot current config before rollback: %v", err)
	}

	return previous, nil
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
		if !entry.Type().IsRegular() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		versionStr := strings.TrimPrefix(entry.Name(), "config.")
		versionStr = strings.TrimSuffix(versionStr, ".yaml")

		if !validConfigBackupVersion(versionStr) {
			continue
		}
		t, _ := time.Parse("20060102-150405", versionStr)
		timestamp := t.Format("2006-01-02 15:04:05")

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
		version := strings.TrimSuffix(strings.TrimPrefix(entry.Name(), "config."), ".yaml")
		if entry.Type().IsRegular() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") && validConfigBackupVersion(version) {
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

func validConfigBackupVersion(version string) bool {
	if len(version) != len("20060102-150405") {
		return false
	}
	for i, r := range version {
		if i == 8 {
			if r != '-' {
				return false
			}
			continue
		}
		if r < '0' || r > '9' {
			return false
		}
	}
	_, err := time.Parse("20060102-150405", version)
	return err == nil
}
