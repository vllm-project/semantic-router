// Package historylock is the lock every writer of a configuration history
// directory holds while it writes: the Router and the Dashboard, in one
// process or several. It depends on nothing else in the Router.
package historylock

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/sys/unix"
)

// name is the file a writer locks.
const name = ".lock"

// wait bounds how long a writer waits for another to finish.
const wait = 10 * time.Second

// Lock holds the lock of the history directory dir, which it creates private
// to its owner, until unlock is called.
func Lock(dir string) (unlock func(), err error) {
	if err = os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	if err = os.Chmod(dir, 0o700); err != nil {
		return nil, err
	}
	fd, err := unix.Open(filepath.Join(dir, name), unix.O_RDWR|unix.O_CREAT|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open the lock of %s: %w", dir, err)
	}
	deadline := time.Now().Add(wait)
	for {
		err = unix.Flock(fd, unix.LOCK_EX|unix.LOCK_NB)
		if !errors.Is(err, unix.EWOULDBLOCK) || time.Now().After(deadline) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("lock %s: %w", dir, err)
	}
	return func() {
		_ = unix.Flock(fd, unix.LOCK_UN)
		_ = unix.Close(fd)
	}, nil
}
