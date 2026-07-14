package evaluation

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

const (
	evaluationPrivateDirMode  fs.FileMode = 0o700
	evaluationPrivateFileMode fs.FileMode = 0o600
)

func ensurePrivateEvaluationDir(path string) error {
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if errors.Is(err, os.ErrNotExist) {
		if mkdirErr := os.MkdirAll(clean, evaluationPrivateDirMode); mkdirErr != nil {
			return fmt.Errorf("create private directory: %w", mkdirErr)
		}
		info, err = os.Lstat(clean)
	}
	if err != nil {
		return fmt.Errorf("inspect private directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return errors.New("private directory must be a real directory")
	}
	if err := os.Chmod(clean, evaluationPrivateDirMode); err != nil {
		return fmt.Errorf("protect private directory: %w", err)
	}
	return nil
}

func preparePrivateEvaluationFile(path string) error {
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if errors.Is(err, os.ErrNotExist) {
		file, createErr := os.OpenFile(clean, os.O_WRONLY|os.O_CREATE|os.O_EXCL, evaluationPrivateFileMode)
		if createErr != nil {
			return fmt.Errorf("create private file: %w", createErr)
		}
		if closeErr := file.Close(); closeErr != nil {
			return fmt.Errorf("close private file: %w", closeErr)
		}
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect private file: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return errors.New("private file must be a regular file")
	}
	if err := os.Chmod(clean, evaluationPrivateFileMode); err != nil {
		return fmt.Errorf("protect private file: %w", err)
	}
	return nil
}

func rejectEvaluationSidecarLinks(dbPath string) error {
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		info, err := os.Lstat(dbPath + suffix)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect database sidecar: %w", err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return errors.New("database sidecar must be a regular file")
		}
	}
	return nil
}

func protectEvaluationDBFiles(dbPath string) error {
	for _, path := range []string{dbPath, dbPath + "-wal", dbPath + "-shm", dbPath + "-journal"} {
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect database file: %w", err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return errors.New("database file must be a regular file")
		}
		if err := os.Chmod(path, evaluationPrivateFileMode); err != nil {
			return fmt.Errorf("protect database file: %w", err)
		}
	}
	return nil
}

func protectEvaluationArtifacts(root string) error {
	return filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return errors.New("evaluation artifact must not be a symbolic link")
		}
		if info.IsDir() {
			return os.Chmod(path, evaluationPrivateDirMode)
		}
		if !info.Mode().IsRegular() {
			return errors.New("evaluation artifact must be a regular file")
		}
		return os.Chmod(path, evaluationPrivateFileMode)
	})
}
