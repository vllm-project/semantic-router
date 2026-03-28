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
	backupDir := configBackupDir(configDir)
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
	}

	version := time.Now().Format("20060102-150405")
	if len(existingData) == 0 {
		return version
	}

	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
		log.Printf("Warning: failed to create backup: %v", err)
	} else {
		log.Printf("[Deploy] Config backup created: %s", backupFile)
	}

	return version
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

	dslFile := filepath.Join(dslDir, "config.dsl")
	if err := os.WriteFile(dslFile, []byte(dsl), 0o644); err != nil {
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
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
		return existingData
	}

	currentVersion := time.Now().Format("20060102-150405")
	preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
	if err := os.WriteFile(preRollbackFile, existingData, 0o644); err != nil {
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
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
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
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") {
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
