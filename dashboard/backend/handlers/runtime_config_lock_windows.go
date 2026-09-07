//go:build windows

package handlers

import (
	"errors"
	"os"
)

type runtimeConfigStoreLock struct{}

func acquireRuntimeConfigStoreLock(storeDir string) (*runtimeConfigStoreLock, bool, error) {
	if storeDir == "" {
		return nil, false, nil
	}
	_, err := os.Lstat(storeDir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return nil, false, errors.New("managed Recipe config coordination is unsupported on Windows")
}

func (l *runtimeConfigStoreLock) hasManagedState() (bool, error) { return false, nil }

func (l *runtimeConfigStoreLock) close() error { return nil }
