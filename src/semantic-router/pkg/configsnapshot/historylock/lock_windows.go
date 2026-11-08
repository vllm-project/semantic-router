//go:build windows

package historylock

import (
	"errors"
	"fmt"

	"golang.org/x/sys/windows"
)

// lockFile is an open lock file.
type lockFile windows.Handle

// openLock opens the lock file, refusing a symbolic link at its path so a
// writer cannot be pointed at another file's lock. The directory's mode bits
// do not restrict access on Windows the way they do elsewhere, so the lock
// file inherits the history directory's ACL.
func openLock(path string) (lockFile, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	handle, err := windows.CreateFile(
		name,
		windows.GENERIC_READ|windows.GENERIC_WRITE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE,
		nil,
		windows.OPEN_ALWAYS,
		windows.FILE_ATTRIBUTE_NORMAL|windows.FILE_FLAG_OPEN_REPARSE_POINT,
		0,
	)
	if err != nil {
		return 0, err
	}
	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &info); err != nil {
		_ = windows.CloseHandle(handle)
		return 0, err
	}
	if info.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		_ = windows.CloseHandle(handle)
		return 0, fmt.Errorf("refusing the lock of a symbolic link: %s", path)
	}
	return lockFile(handle), nil
}

// tryLock takes the exclusive lock of the file's first byte without waiting;
// it reports errLocked while another writer holds it.
func tryLock(file lockFile) error {
	err := windows.LockFileEx(
		windows.Handle(file),
		windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
		0, 1, 0, &windows.Overlapped{},
	)
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return errLocked
	}
	return err
}

func unlockLock(file lockFile) error {
	return windows.UnlockFileEx(windows.Handle(file), 0, 1, 0, &windows.Overlapped{})
}

func closeLock(file lockFile) error {
	return windows.CloseHandle(windows.Handle(file))
}
