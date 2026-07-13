package workflowstore

import (
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const (
	workflowDirectoryMode os.FileMode = 0o700
	workflowDatabaseMode  os.FileMode = 0o600
)

var workflowSQLiteSidecarSuffixes = [...]string{"", "-wal", "-shm", "-journal"}

func prepareWorkflowDatabaseStorage(dsn string) (string, error) {
	filesystemPath, hasFile, err := workflowDatabaseFilesystemPath(dsn)
	if err != nil || !hasFile {
		return filesystemPath, err
	}
	if err := createWorkflowDatabaseDirectory(filesystemPath); err != nil {
		return "", err
	}
	if err := createOrSecureWorkflowDatabaseFile(filesystemPath); err != nil {
		return "", err
	}
	if err := secureExistingWorkflowDatabaseFiles(filesystemPath); err != nil {
		return "", err
	}
	return filesystemPath, nil
}

func workflowSQLiteDSN(dsn string) (string, error) {
	base, rawQuery := splitSQLiteDSN(dsn)
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return "", fmt.Errorf("workflowstore: parse database query: %w", err)
	}

	// Remove driver aliases before setting the service-owned values so callers
	// cannot silently disable WAL, the busy timeout, or foreign-key checks.
	for _, key := range []string{
		"_journal",
		"_journal_mode",
		"_timeout",
		"_busy_timeout",
		"_fk",
		"_foreign_keys",
	} {
		values.Del(key)
	}
	values.Set("_journal_mode", "WAL")
	values.Set("_busy_timeout", "5000")
	values.Set("_foreign_keys", "1")
	return base + "?" + values.Encode(), nil
}

func workflowDatabaseFilesystemPath(dsn string) (string, bool, error) {
	if dsn == "" {
		return "", false, errors.New("workflowstore: database path is required")
	}
	rawPath, rawQuery := splitSQLiteDSN(dsn)
	queryValues, err := url.ParseQuery(rawQuery)
	if err != nil {
		return "", false, fmt.Errorf("workflowstore: parse database query: %w", err)
	}
	if rawPath == ":memory:" || rawPath == "file::memory:" || queryValues.Get("mode") == "memory" {
		return "", false, nil
	}

	rawPath = strings.TrimPrefix(rawPath, "file:")
	unescaped, err := url.PathUnescape(rawPath)
	if err != nil {
		return "", false, fmt.Errorf("workflowstore: parse database path: %w", err)
	}
	if unescaped == "" {
		return "", false, errors.New("workflowstore: database path is required")
	}
	return filepath.Clean(unescaped), true, nil
}

func splitSQLiteDSN(dsn string) (string, string) {
	if query := strings.IndexByte(dsn, '?'); query >= 0 {
		return dsn[:query], dsn[query+1:]
	}
	return dsn, ""
}

func createWorkflowDatabaseDirectory(databasePath string) error {
	dir := filepath.Dir(databasePath)
	if dir == "." || dir == "" || dir == string(filepath.Separator) {
		return nil
	}
	info, err := os.Stat(dir)
	if err == nil {
		if !info.IsDir() {
			return errors.New("workflowstore: database parent must be a directory")
		}
		// Existing directories are caller-owned and may be intentionally shared.
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("workflowstore: inspect database directory: %w", err)
	}
	if err := os.MkdirAll(dir, workflowDirectoryMode); err != nil {
		return fmt.Errorf("workflowstore: create database directory: %w", err)
	}
	if err := os.Chmod(dir, workflowDirectoryMode); err != nil {
		return fmt.Errorf("workflowstore: secure database directory: %w", err)
	}
	return nil
}

func createOrSecureWorkflowDatabaseFile(databasePath string) error {
	info, err := os.Lstat(databasePath)
	if errors.Is(err, os.ErrNotExist) {
		file, createErr := os.OpenFile(
			databasePath,
			os.O_CREATE|os.O_EXCL|os.O_RDWR,
			workflowDatabaseMode,
		)
		if createErr != nil {
			return fmt.Errorf("workflowstore: create database: %w", createErr)
		}
		if closeErr := file.Close(); closeErr != nil {
			return fmt.Errorf("workflowstore: close new database: %w", closeErr)
		}
		return nil
	}
	if err != nil {
		return fmt.Errorf("workflowstore: inspect database: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("workflowstore: database must be a regular file")
	}
	if err := os.Chmod(databasePath, workflowDatabaseMode); err != nil {
		return fmt.Errorf("workflowstore: secure database: %w", err)
	}
	return nil
}

func secureExistingWorkflowDatabaseFiles(databasePath string) error {
	if databasePath == "" {
		return nil
	}
	for _, suffix := range workflowSQLiteSidecarSuffixes {
		path := databasePath + suffix
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("workflowstore: inspect sqlite file: %w", err)
		}
		if !info.Mode().IsRegular() {
			return errors.New("workflowstore: sqlite files must be regular files")
		}
		if err := os.Chmod(path, workflowDatabaseMode); err != nil {
			return fmt.Errorf("workflowstore: secure sqlite file: %w", err)
		}
	}
	return nil
}

func closeWorkflowDatabase(db *sql.DB, filesystemPath string) error {
	if db == nil {
		return secureExistingWorkflowDatabaseFiles(filesystemPath)
	}
	beforeCloseErr := secureExistingWorkflowDatabaseFiles(filesystemPath)
	databaseErr := db.Close()
	afterCloseErr := secureExistingWorkflowDatabaseFiles(filesystemPath)
	return errors.Join(beforeCloseErr, databaseErr, afterCloseErr)
}
