package auth

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const (
	authDirectoryMode os.FileMode = 0o700
	authDatabaseMode  os.FileMode = 0o600
)

func prepareAuthDatabaseStorage(dsn string) (string, error) {
	filesystemPath, hasFile, err := authDatabaseFilesystemPath(dsn)
	if err != nil || !hasFile {
		return filesystemPath, err
	}
	if err := createAuthDatabaseDirectory(filesystemPath); err != nil {
		return "", err
	}
	if err := createOrSecureAuthDatabaseFile(filesystemPath); err != nil {
		return "", err
	}
	return filesystemPath, nil
}

// authSQLiteDSN enforces foreign-key constraints on every physical connection
// opened by database/sql. Setting PRAGMA foreign_keys after Open is not enough:
// it is connection-local and would be lost when ConnMaxLifetime replaces the
// connection. Existing SQLite URI parameters and in-memory DSNs are retained.
func authSQLiteDSN(dsn string) (string, error) {
	base := dsn
	rawQuery := ""
	if query := strings.IndexByte(dsn, '?'); query >= 0 {
		base = dsn[:query]
		rawQuery = dsn[query+1:]
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return "", fmt.Errorf("parse auth database query: %w", err)
	}
	// The driver accepts both spellings. Remove the alias so a caller cannot
	// silently override the canonical, service-owned setting.
	values.Del("_fk")
	values.Set("_foreign_keys", "1")
	return base + "?" + values.Encode(), nil
}

func authDatabaseFilesystemPath(dsn string) (string, bool, error) {
	if dsn == "" {
		return "", false, errors.New("auth database path is required")
	}
	if dsn == ":memory:" || strings.HasPrefix(dsn, "file::memory:") {
		return "", false, nil
	}
	rawPath := dsn
	rawQuery := ""
	if query := strings.IndexByte(rawPath, '?'); query >= 0 {
		rawQuery = rawPath[query+1:]
		rawPath = rawPath[:query]
	}
	queryValues, err := url.ParseQuery(rawQuery)
	if err != nil {
		return "", false, fmt.Errorf("parse auth database query: %w", err)
	}
	if queryValues.Get("mode") == "memory" {
		return "", false, nil
	}
	rawPath = strings.TrimPrefix(rawPath, "file:")
	unescaped, err := url.PathUnescape(rawPath)
	if err != nil {
		return "", false, fmt.Errorf("parse auth database path: %w", err)
	}
	if unescaped == "" {
		return "", false, errors.New("auth database path is required")
	}
	return filepath.Clean(unescaped), true, nil
}

func createAuthDatabaseDirectory(databasePath string) error {
	dir := filepath.Dir(databasePath)
	if dir == "." || dir == "" || dir == string(filepath.Separator) {
		return nil
	}
	info, err := os.Stat(dir)
	if err == nil {
		if !info.IsDir() {
			return errors.New("auth database parent must be a directory")
		}
		// The caller may intentionally use a shared existing directory. Never
		// change its mode; only service-created leaf directories are private.
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect auth db directory: %w", err)
	}
	if err := os.MkdirAll(dir, authDirectoryMode); err != nil {
		return fmt.Errorf("create auth db directory: %w", err)
	}
	if err := os.Chmod(dir, authDirectoryMode); err != nil {
		return fmt.Errorf("secure auth db directory: %w", err)
	}
	return nil
}

func createOrSecureAuthDatabaseFile(databasePath string) error {
	info, err := os.Lstat(databasePath)
	if errors.Is(err, os.ErrNotExist) {
		file, createErr := os.OpenFile(
			databasePath,
			os.O_CREATE|os.O_EXCL|os.O_RDWR,
			authDatabaseMode,
		)
		if createErr != nil {
			return fmt.Errorf("create auth db: %w", createErr)
		}
		if closeErr := file.Close(); closeErr != nil {
			return fmt.Errorf("close new auth db: %w", closeErr)
		}
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect auth db: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("auth database must be a regular file")
	}
	if err := os.Chmod(databasePath, authDatabaseMode); err != nil {
		return fmt.Errorf("secure auth db: %w", err)
	}
	return nil
}

func secureExistingAuthDatabaseFiles(databasePath string) error {
	if databasePath == "" {
		return nil
	}
	for _, path := range []string{
		databasePath,
		databasePath + "-wal",
		databasePath + "-shm",
		databasePath + "-journal",
	} {
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect auth sqlite file: %w", err)
		}
		if !info.Mode().IsRegular() {
			return errors.New("auth sqlite files must be regular files")
		}
		if err := os.Chmod(path, authDatabaseMode); err != nil {
			return fmt.Errorf("secure auth sqlite file: %w", err)
		}
	}
	return nil
}
