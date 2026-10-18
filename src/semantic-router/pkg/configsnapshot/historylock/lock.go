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
)

// name is the file a writer locks.
const name = ".lock"

// wait bounds how long a writer waits for another to finish.
const wait = 10 * time.Second

// errLocked reports a lock another writer holds.
var errLocked = errors.New("locked by another writer")

// Lock holds the lock of the history directory dir, which it creates private
// to its owner, until unlock is called.
func Lock(dir string) (unlock func(), err error) {
	if err = os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	if err = os.Chmod(dir, 0o700); err != nil {
		return nil, err
	}
	file, err := openLock(filepath.Join(dir, name))
	if err != nil {
		return nil, fmt.Errorf("open the lock of %s: %w", dir, err)
	}
	deadline := time.Now().Add(wait)
	for {
		err = tryLock(file)
		if !errors.Is(err, errLocked) || time.Now().After(deadline) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		_ = closeLock(file)
		return nil, fmt.Errorf("lock %s: %w", dir, err)
	}
	return func() {
		_ = unlockLock(file)
		_ = closeLock(file)
	}, nil
}
