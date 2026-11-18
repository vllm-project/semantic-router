//go:build !windows

package config

import (
	"errors"
	"io/fs"
	"os"
	"syscall"
)

// KeepReplacedFileMode gives file, written to replace the file at path, that
// file's mode, owner and group, so a save keeps a private configuration
// private. A writer that cannot give file the previous owner, such as the
// Dashboard's nonroot user replacing a file the host user owns, leaves file's
// mode as it is, so that owner can still read what replaced it. With no file
// at path, file keeps its mode too.
func KeepReplacedFileMode(file *os.File, path string) error {
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	owner, ok := info.Sys().(*syscall.Stat_t)
	if ok && file.Chown(int(owner.Uid), int(owner.Gid)) != nil && int(owner.Uid) != os.Geteuid() {
		return nil
	}
	return file.Chmod(info.Mode().Perm())
}
