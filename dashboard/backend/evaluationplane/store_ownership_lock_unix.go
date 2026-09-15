//go:build !windows

package evaluationplane

import (
	"errors"
	"fmt"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const evaluationStoreOwnershipLockName = ".evaluation-store.lock"

// evaluationStoreOwnershipLock is the platform resource retained by the one
// shared root coordinator while at least one Service is live. flock is released
// by the kernel on process exit, so a crash cannot leave a stale owner behind.
type evaluationStoreOwnershipLock struct{ fd int }

func (lock *evaluationStoreOwnershipLock) close() error {
	if lock == nil {
		return nil
	}
	return errors.Join(unix.Flock(lock.fd, unix.LOCK_UN), unix.Close(lock.fd))
}

func openEvaluationStoreOwnershipLock(root string) (*evaluationStoreOwnershipLock, error) {
	if err := requirePrivateDirectory(root); err != nil {
		return nil, fmt.Errorf("validate evaluation store root: %w", err)
	}
	directoryFD, err := unix.Open(filepath.Clean(root), unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open evaluation store root: %w", err)
	}
	defer func() { _ = unix.Close(directoryFD) }()
	fd, err := unix.Openat(
		directoryFD, evaluationStoreOwnershipLockName,
		unix.O_RDWR|unix.O_CREAT|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o600,
	)
	if err != nil {
		return nil, fmt.Errorf("open evaluation store ownership lock: %w", err)
	}
	cleanup := func() { _ = unix.Close(fd) }
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		cleanup()
		return nil, fmt.Errorf("stat evaluation store ownership lock: %w", err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG || stat.Nlink != 1 || stat.Mode&0o777 != 0o600 {
		cleanup()
		return nil, fmt.Errorf("evaluation store ownership lock is not a private regular file")
	}
	if err := unix.Flock(fd, unix.LOCK_EX|unix.LOCK_NB); err != nil {
		cleanup()
		if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
			return nil, fmt.Errorf("%w: evaluation data directory is owned by another process", ErrConflict)
		}
		return nil, fmt.Errorf("lock evaluation data directory: %w", err)
	}
	return &evaluationStoreOwnershipLock{fd: fd}, nil
}
