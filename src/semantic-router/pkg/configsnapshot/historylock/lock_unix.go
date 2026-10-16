//go:build !windows

package historylock

import (
	"errors"

	"golang.org/x/sys/unix"
)

// lockFile is an open lock file.
type lockFile int

// openLock opens the lock file, refusing a symbolic link at its path so a
// writer cannot be pointed at another file's lock.
func openLock(path string) (lockFile, error) {
	fd, err := unix.Open(path, unix.O_RDWR|unix.O_CREAT|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0o600)
	return lockFile(fd), err
}

func tryLock(file lockFile) error {
	err := unix.Flock(int(file), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return errLocked
	}
	return err
}

func unlockLock(file lockFile) error {
	return unix.Flock(int(file), unix.LOCK_UN)
}

func closeLock(file lockFile) error {
	return unix.Close(int(file))
}
