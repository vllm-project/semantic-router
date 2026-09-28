//go:build !windows

package handlers

import (
	"errors"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const runtimeConfigLockName = "runtime-config.lock"

type runtimeConfigStoreLock struct {
	directoryFD int
	lockFD      int
}

func acquireRuntimeConfigStoreLock(storeDir string) (*runtimeConfigStoreLock, bool, error) {
	if storeDir == "" {
		return nil, false, nil
	}
	info, err := os.Lstat(storeDir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, false, errors.New("recipe store must be a real directory")
	}
	directoryFD, err := unix.Open(filepath.Clean(storeDir), unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, false, err
	}
	lockFD, err := unix.Openat(directoryFD, runtimeConfigLockName, unix.O_RDWR|unix.O_CREAT|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o600)
	if err != nil {
		_ = unix.Close(directoryFD)
		return nil, false, err
	}
	cleanup := func() {
		_ = unix.Close(lockFD)
		_ = unix.Close(directoryFD)
	}
	var stat unix.Stat_t
	if err := unix.Fstat(lockFD, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG || stat.Nlink != 1 {
		cleanup()
		if err != nil {
			return nil, false, err
		}
		return nil, false, errors.New("runtime config lock must be a regular private file")
	}
	if stat.Mode&0o007 != 0 {
		cleanup()
		return nil, false, errors.New("runtime config lock must not grant access to other users")
	}
	if err := unix.Flock(lockFD, unix.LOCK_EX|unix.LOCK_NB); err != nil {
		cleanup()
		if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
			return nil, false, errRuntimeConfigLockBusy
		}
		return nil, false, err
	}
	return &runtimeConfigStoreLock{directoryFD: directoryFD, lockFD: lockFD}, true, nil
}

func (l *runtimeConfigStoreLock) hasManagedState() (bool, error) {
	for _, name := range []string{"active.json", "activation-pending.json"} {
		var stat unix.Stat_t
		err := unix.Fstatat(l.directoryFD, name, &stat, unix.AT_SYMLINK_NOFOLLOW)
		if err == nil {
			return true, nil
		}
		if !errors.Is(err, unix.ENOENT) {
			return false, err
		}
	}
	return false, nil
}

func (l *runtimeConfigStoreLock) close() error {
	unlockErr := unix.Flock(l.lockFD, unix.LOCK_UN)
	lockCloseErr := unix.Close(l.lockFD)
	directoryCloseErr := unix.Close(l.directoryFD)
	return errors.Join(unlockErr, lockCloseErr, directoryCloseErr)
}
