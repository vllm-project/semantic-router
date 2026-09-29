package handlers

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVersionedConfigBackupsNeverReplaceOneAnother(t *testing.T) {
	configDir := t.TempDir()
	t.Setenv("DASHBOARD_CONFIG_DIR", configDir)
	backupDir := configBackupDir(configDir)
	if err := ensureConfigSnapshotDir(backupDir); err != nil {
		t.Fatal(err)
	}
	stamp := time.Date(2026, 9, 24, 12, 0, 0, 123456789, time.UTC)
	firstVersion, err := writeVersionedConfigBackup(backupDir, []byte("first baseline"), stamp)
	if err != nil {
		t.Fatal(err)
	}
	secondVersion, err := writeVersionedConfigBackup(backupDir, []byte("second baseline"), stamp)
	if err != nil {
		t.Fatal(err)
	}
	if firstVersion == secondVersion {
		t.Fatal("two backups from one timestamp reused a version")
	}
	for _, tc := range []struct {
		version string
		want    string
	}{{firstVersion, "first baseline"}, {secondVersion, "second baseline"}} {
		got, readErr := readConfigBackup(configDir, tc.version)
		if readErr != nil || string(got) != tc.want {
			t.Fatalf("version %s = %q, error %v; want %q", tc.version, got, readErr, tc.want)
		}
	}

	firstPath := filepath.Join(backupDir, "config."+firstVersion+".yaml")
	if writeErr := writeConfigSnapshotExclusive(firstPath, []byte("clobber")); !errors.Is(writeErr, os.ErrExist) {
		t.Fatalf("existing backup replacement error = %v, want os.ErrExist", writeErr)
	}
	unchanged, err := os.ReadFile(firstPath)
	if err != nil || !bytes.Equal(unchanged, []byte("first baseline")) {
		t.Fatalf("first backup was overwritten: %q, error %v", unchanged, err)
	}

	legacyVersion := "20260923-110000"
	legacyPath := filepath.Join(backupDir, "config."+legacyVersion+".yaml")
	if writeErr := os.WriteFile(legacyPath, []byte("legacy baseline"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if got, readErr := readConfigBackup(configDir, legacyVersion); readErr != nil || string(got) != "legacy baseline" {
		t.Fatalf("legacy rollback version = %q, error %v", got, readErr)
	}
	versions, err := listConfigVersions(filepath.Join(configDir, "config.yaml"))
	if err != nil || len(versions) != 3 {
		t.Fatalf("backup versions = %+v, error %v", versions, err)
	}
	if versions[0].Version != secondVersion || versions[1].Version != firstVersion || versions[2].Version != legacyVersion {
		t.Fatalf("backup order = %+v", versions)
	}
}
