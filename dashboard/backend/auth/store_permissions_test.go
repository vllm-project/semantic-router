package auth

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestNewStoreSecuresCreatedDirectoryDatabaseAndWALFiles(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not enforced on Windows")
	}

	directory := filepath.Join(t.TempDir(), "private-auth")
	databasePath := filepath.Join(directory, "auth.db")
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	assertFileMode(t, directory, authDirectoryMode)
	assertFileMode(t, databasePath, authDatabaseMode)

	var journalMode string
	if err := store.db.QueryRowContext(context.Background(), `PRAGMA journal_mode=WAL`).Scan(&journalMode); err != nil {
		t.Fatalf("enable WAL: %v", err)
	}
	if journalMode != "wal" {
		t.Fatalf("journal mode = %q, want wal", journalMode)
	}
	if _, err := store.db.ExecContext(context.Background(), `INSERT OR IGNORE INTO role_permissions(role, permission_key, allowed) VALUES('read','mode-test',1)`); err != nil {
		t.Fatalf("write WAL fixture: %v", err)
	}
	assertFileMode(t, databasePath+"-wal", authDatabaseMode)
	assertFileMode(t, databasePath+"-shm", authDatabaseMode)
}

func TestNewStoreSecuresDatabaseWithoutChangingExistingSharedDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not enforced on Windows")
	}

	sharedDirectory := t.TempDir()
	if err := os.Chmod(sharedDirectory, 0o755); err != nil {
		t.Fatalf("chmod shared directory: %v", err)
	}
	databasePath := filepath.Join(sharedDirectory, "auth.db")
	if err := os.WriteFile(databasePath, nil, 0o644); err != nil {
		t.Fatalf("seed auth db: %v", err)
	}
	if err := os.Chmod(databasePath, 0o644); err != nil {
		t.Fatalf("chmod auth db fixture: %v", err)
	}

	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	assertFileMode(t, sharedDirectory, 0o755)
	assertFileMode(t, databasePath, authDatabaseMode)
}

func TestNewStoreDoesNotTreatModeMemoryFilenameAsMemoryDSN(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not enforced on Windows")
	}

	databasePath := filepath.Join(t.TempDir(), "mode=memory-auth.db")
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	assertFileMode(t, databasePath, authDatabaseMode)
}

func assertFileMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if got := info.Mode().Perm(); got != want.Perm() {
		t.Fatalf("mode %s = %04o, want %04o", path, got, want.Perm())
	}
}
