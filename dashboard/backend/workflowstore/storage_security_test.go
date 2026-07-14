package workflowstore

import (
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestOpenSecuresCreatedDirectoryDatabaseAndWALSidecars(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	root := t.TempDir()
	directory := filepath.Join(root, "private-workflow")
	databasePath := filepath.Join(directory, "workflow.db")
	store, err := Open(databasePath, Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	if _, err := store.db.Exec(
		`INSERT INTO openclaw_container(name, json) VALUES(?, ?)`,
		"mode-test",
		`{"token":"sensitive"}`,
	); err != nil {
		_ = store.Close()
		t.Fatalf("write WAL fixture: %v", err)
	}
	assertWorkflowFileMode(t, directory, workflowDirectoryMode)
	assertWorkflowFileMode(t, databasePath, workflowDatabaseMode)
	assertWorkflowFileMode(t, databasePath+"-wal", workflowDatabaseMode)
	assertWorkflowFileMode(t, databasePath+"-shm", workflowDatabaseMode)

	journalPath := databasePath + "-journal"
	if err := os.WriteFile(journalPath, []byte("close-mode-fixture"), 0o644); err != nil {
		_ = store.Close()
		t.Fatalf("write journal fixture: %v", err)
	}
	if err := os.Chmod(journalPath, 0o644); err != nil {
		_ = store.Close()
		t.Fatalf("chmod journal fixture: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	assertWorkflowFileMode(t, databasePath, workflowDatabaseMode)
	assertWorkflowFileMode(t, journalPath, workflowDatabaseMode)
}

func TestOpenSecuresExistingDatabaseWithoutChangingCallerDirectory(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	sharedDirectory := t.TempDir()
	if err := os.Chmod(sharedDirectory, 0o755); err != nil {
		t.Fatalf("chmod shared directory: %v", err)
	}
	databasePath := filepath.Join(sharedDirectory, "workflow.db")
	if err := os.WriteFile(databasePath, nil, 0o644); err != nil {
		t.Fatalf("seed database: %v", err)
	}
	if err := os.Chmod(databasePath, 0o644); err != nil {
		t.Fatalf("chmod database fixture: %v", err)
	}

	store, err := Open(databasePath, Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	assertWorkflowFileMode(t, sharedDirectory, 0o755)
	assertWorkflowFileMode(t, databasePath, workflowDatabaseMode)
}

func TestOpenDoesNotChangeExistingParentWhenCreatingPrivateLeaf(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	parent := t.TempDir()
	if err := os.Chmod(parent, 0o755); err != nil {
		t.Fatalf("chmod parent: %v", err)
	}
	leaf := filepath.Join(parent, "service-owned")
	store, err := Open(filepath.Join(leaf, "workflow.db"), Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	assertWorkflowFileMode(t, parent, 0o755)
	assertWorkflowFileMode(t, leaf, workflowDirectoryMode)
}

func TestOpenRejectsSymlinkDatabase(t *testing.T) {
	requireSymlinkSupport(t)
	t.Parallel()

	directory := t.TempDir()
	targetPath := filepath.Join(directory, "target.db")
	if err := os.WriteFile(targetPath, []byte("do-not-touch"), 0o644); err != nil {
		t.Fatalf("write target: %v", err)
	}
	databasePath := filepath.Join(directory, "workflow.db")
	if err := os.Symlink(targetPath, databasePath); err != nil {
		t.Fatalf("create database symlink: %v", err)
	}

	store, err := Open(databasePath, Options{})
	if store != nil {
		_ = store.Close()
		t.Fatal("Open() unexpectedly accepted a symlink database")
	}
	if err == nil || !strings.Contains(err.Error(), "regular file") {
		t.Fatalf("Open() error = %v, want regular-file rejection", err)
	}
	contents, readErr := os.ReadFile(targetPath)
	if readErr != nil || string(contents) != "do-not-touch" {
		t.Fatalf("symlink target changed: contents=%q err=%v", contents, readErr)
	}
}

func TestOpenRejectsSymlinkSQLiteSidecars(t *testing.T) {
	requireSymlinkSupport(t)
	t.Parallel()

	for _, suffix := range workflowSQLiteSidecarSuffixes[1:] {
		t.Run(suffix, func(t *testing.T) {
			t.Parallel()
			directory := t.TempDir()
			databasePath := filepath.Join(directory, "workflow.db")
			if err := os.WriteFile(databasePath, nil, workflowDatabaseMode); err != nil {
				t.Fatalf("seed database: %v", err)
			}
			targetPath := filepath.Join(directory, "sidecar-target")
			if err := os.WriteFile(targetPath, []byte("do-not-touch"), 0o644); err != nil {
				t.Fatalf("write target: %v", err)
			}
			if err := os.Symlink(targetPath, databasePath+suffix); err != nil {
				t.Fatalf("create sidecar symlink: %v", err)
			}

			store, err := Open(databasePath, Options{})
			if store != nil {
				_ = store.Close()
				t.Fatal("Open() unexpectedly accepted a symlink sidecar")
			}
			if err == nil || !strings.Contains(err.Error(), "regular file") {
				t.Fatalf("Open() error = %v, want regular-file rejection", err)
			}
		})
	}
}

func TestOpenRejectsNonRegularDatabaseAndSidecar(t *testing.T) {
	t.Parallel()

	for _, testCase := range []struct {
		name   string
		suffix string
	}{
		{name: "database", suffix: ""},
		{name: "wal", suffix: "-wal"},
		{name: "shm", suffix: "-shm"},
		{name: "journal", suffix: "-journal"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			directory := t.TempDir()
			databasePath := filepath.Join(directory, "workflow.db")
			if testCase.suffix != "" {
				if err := os.WriteFile(databasePath, nil, workflowDatabaseMode); err != nil {
					t.Fatalf("seed database: %v", err)
				}
			}
			if err := os.Mkdir(databasePath+testCase.suffix, 0o700); err != nil {
				t.Fatalf("create non-regular fixture: %v", err)
			}

			store, err := Open(databasePath, Options{})
			if store != nil {
				_ = store.Close()
				t.Fatal("Open() unexpectedly accepted a non-regular SQLite file")
			}
			if err == nil || !strings.Contains(err.Error(), "regular file") {
				t.Fatalf("Open() error = %v, want regular-file rejection", err)
			}
		})
	}
}

func TestSecureExistingWorkflowDatabaseFilesTightensEverySidecar(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	databasePath := filepath.Join(t.TempDir(), "workflow.db")
	for _, suffix := range workflowSQLiteSidecarSuffixes {
		path := databasePath + suffix
		if err := os.WriteFile(path, []byte("fixture"), 0o644); err != nil {
			t.Fatalf("write %s: %v", suffix, err)
		}
		if err := os.Chmod(path, 0o644); err != nil {
			t.Fatalf("chmod %s: %v", suffix, err)
		}
	}

	if err := secureExistingWorkflowDatabaseFiles(databasePath); err != nil {
		t.Fatalf("secureExistingWorkflowDatabaseFiles() error = %v", err)
	}
	for _, suffix := range workflowSQLiteSidecarSuffixes {
		assertWorkflowFileMode(t, databasePath+suffix, workflowDatabaseMode)
	}
}

func TestCloseRejectsSymlinkSidecarAndStillClosesDatabase(t *testing.T) {
	requireSymlinkSupport(t)
	t.Parallel()

	directory := t.TempDir()
	databasePath := filepath.Join(directory, "workflow.db")
	store, err := Open(databasePath, Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	targetPath := filepath.Join(directory, "sidecar-target")
	if err := os.WriteFile(targetPath, []byte("do-not-touch"), 0o644); err != nil {
		_ = store.Close()
		t.Fatalf("write target: %v", err)
	}
	if err := os.Symlink(targetPath, databasePath+"-journal"); err != nil {
		_ = store.Close()
		t.Fatalf("create sidecar symlink: %v", err)
	}

	closeErr := store.Close()
	if closeErr == nil || !strings.Contains(closeErr.Error(), "regular file") {
		t.Fatalf("Close() error = %v, want regular-file rejection", closeErr)
	}
	if pingErr := store.db.Ping(); pingErr == nil {
		t.Fatal("database remained open after sidecar rejection")
	}
}

func TestOpenPreservesMemoryDSNs(t *testing.T) {
	for _, dsn := range []string{
		":memory:",
		"file::memory:?cache=shared",
		"file:workflow-memory?mode=memory&cache=shared",
	} {
		t.Run(dsn, func(t *testing.T) {
			store, err := Open(dsn, Options{})
			if err != nil {
				t.Fatalf("Open(%q) error = %v", dsn, err)
			}
			defer store.Close()
			if store.filesystemPath != "" {
				t.Fatalf("filesystemPath = %q, want empty for memory DSN", store.filesystemPath)
			}
			if _, err := store.db.Exec(
				`INSERT INTO openclaw_container(name, json) VALUES(?, ?)`,
				"memory-test",
				`{}`,
			); err != nil {
				t.Fatalf("write memory store: %v", err)
			}
		})
	}
}

func TestOpenPreservesLegalFileDSNQuery(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	databasePath := filepath.Join(t.TempDir(), "workflow with space.db")
	fileURL := (&url.URL{Scheme: "file", Path: databasePath}).String()
	store, err := Open(fileURL+"?cache=private&_busy_timeout=1", Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if store.filesystemPath != databasePath {
		t.Fatalf("filesystemPath = %q, want %q", store.filesystemPath, databasePath)
	}
	assertWorkflowFileMode(t, databasePath, workflowDatabaseMode)

	dsn, err := workflowSQLiteDSN(fileURL + "?cache=private&_fk=0&_journal=DELETE")
	if err != nil {
		t.Fatalf("workflowSQLiteDSN() error = %v", err)
	}
	_, rawQuery := splitSQLiteDSN(dsn)
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		t.Fatalf("ParseQuery() error = %v", err)
	}
	if values.Get("cache") != "private" || values.Get("_foreign_keys") != "1" || values.Get("_journal_mode") != "WAL" {
		t.Fatalf("unexpected DSN query: %v", values)
	}
	if values.Has("_fk") || values.Has("_journal") {
		t.Fatalf("driver aliases survived canonicalization: %v", values)
	}
}

func TestOpenDoesNotTreatModeMemoryFilenameAsMemoryDSN(t *testing.T) {
	requirePOSIXFileModes(t)
	t.Parallel()

	databasePath := filepath.Join(t.TempDir(), "mode=memory-workflow.db")
	store, err := Open(databasePath, Options{})
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if store.filesystemPath != databasePath {
		t.Fatalf("filesystemPath = %q, want %q", store.filesystemPath, databasePath)
	}
	assertWorkflowFileMode(t, databasePath, workflowDatabaseMode)
}

func requirePOSIXFileModes(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX mode bits are not enforced on Windows")
	}
}

func requireSymlinkSupport(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires additional privileges on Windows")
	}
}

func assertWorkflowFileMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if got := info.Mode().Perm(); got != want.Perm() {
		t.Fatalf("mode %s = %04o, want %04o", path, got, want.Perm())
	}
}
