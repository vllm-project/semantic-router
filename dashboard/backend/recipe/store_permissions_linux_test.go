//go:build linux

package recipe

import (
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// The CLI reads what the Dashboard writes through the store's group, whatever
// umask the Dashboard runs with.
func TestStoreSharesWhatItWritesWithTheStoreGroup(t *testing.T) {
	previous := syscall.Umask(0o077)
	t.Cleanup(func() { syscall.Umask(previous) })
	root := filepath.Join(t.TempDir(), "store")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	// The entrypoint gives the store root the shared group and the setgid bit.
	if err := os.Chmod(root, os.ModeSetgid|0o770); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: root})
	if err := store.ensureLayout(); err != nil {
		t.Fatal(err)
	}
	if err := writeJSONAtomically(filepath.Join(root, "refs", "pkg", "1.0.0.json"), map[string]string{}); err != nil {
		t.Fatal(err)
	}
	if err := makeStoreDirectory(filepath.Join(root, "transactions", "0123456789abcdef0123456789abcdef")); err != nil {
		t.Fatal(err)
	}

	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || path == root {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if info.Mode().Perm() != storeDirectoryMode || info.Mode()&os.ModeSetgid == 0 {
				t.Errorf("%s mode = %v, want %v with setgid", path, info.Mode(), storeDirectoryMode)
			}
		} else if info.Mode().Perm() != storeFileMode {
			t.Errorf("%s mode = %v, want %v", path, info.Mode(), storeFileMode)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
