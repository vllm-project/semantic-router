package handlers

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const (
	privateStateDirectoryMode os.FileMode = 0o700
	privateStateFileMode      os.FileMode = 0o600
)

// ensurePrivateStateDirectory creates a Dashboard-owned state directory and
// rejects a symlink or non-directory at the final path. Parent directories are
// deployment-owned and intentionally left unchanged because .vllm-sr is also
// mounted into the managed Router and Envoy containers.
func ensurePrivateStateDirectory(dir string) error {
	dir = filepath.Clean(dir)
	parent := filepath.Dir(dir)
	if parent != dir {
		parentInfo, err := os.Lstat(parent)
		if errors.Is(err, os.ErrNotExist) {
			if mkdirErr := os.MkdirAll(parent, 0o755); mkdirErr != nil {
				return fmt.Errorf("create private state parent: %w", mkdirErr)
			}
			parentInfo, err = os.Lstat(parent)
		}
		if err != nil {
			return fmt.Errorf("inspect private state parent: %w", err)
		}
		if parentInfo.Mode()&os.ModeSymlink != 0 || !parentInfo.IsDir() {
			return errors.New("private state parent must be a directory, not a symlink or special file")
		}
	}

	info, err := os.Lstat(dir)
	if errors.Is(err, os.ErrNotExist) {
		if mkdirErr := os.Mkdir(dir, privateStateDirectoryMode); mkdirErr != nil {
			return fmt.Errorf("create private state directory: %w", mkdirErr)
		}
		info, err = os.Lstat(dir)
	}
	if err != nil {
		return fmt.Errorf("inspect private state directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("private state path must be a directory, not a symlink or special file")
	}
	if err := os.Chmod(dir, privateStateDirectoryMode); err != nil {
		return fmt.Errorf("secure private state directory: %w", err)
	}
	return nil
}

func ensureSharedStateDirectory(dir string) error {
	info, err := os.Lstat(dir)
	if errors.Is(err, os.ErrNotExist) {
		if mkdirErr := os.MkdirAll(dir, 0o755); mkdirErr != nil {
			return fmt.Errorf("create shared state directory: %w", mkdirErr)
		}
		info, err = os.Lstat(dir)
	}
	if err != nil {
		return fmt.Errorf("inspect shared state directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("shared state path must be a directory, not a symlink or special file")
	}
	return nil
}

// writePrivateStateFile uses a same-directory temporary file followed by a
// rename. Besides avoiding partial writes, rename replaces rather than follows
// a destination symlink, closing the common check/write symlink race.
func writePrivateStateFile(path string, data []byte) error {
	path = filepath.Clean(path)
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() {
			return errors.New("private state target must be a regular file")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect private state target: %w", err)
	}

	dir := filepath.Dir(path)
	info, err := os.Lstat(dir)
	if err != nil {
		return fmt.Errorf("inspect private state parent: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("private state parent must be a directory, not a symlink or special file")
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create private state temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if err := tmp.Chmod(privateStateFileMode); err != nil {
		return fmt.Errorf("secure private state file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write private state file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync private state file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close private state file: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("replace private state file: %w", err)
	}
	return nil
}

func readPrivateStateFile(path string) ([]byte, error) {
	path = filepath.Clean(path)
	parent := filepath.Dir(path)
	for _, dir := range []string{parent, filepath.Dir(parent)} {
		info, err := os.Lstat(dir)
		if err != nil {
			return nil, fmt.Errorf("inspect private state ancestor: %w", err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return nil, errors.New("private state ancestor must be a real directory")
		}
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return nil, errors.New("private state file must be a regular file")
	}
	return os.ReadFile(path)
}
