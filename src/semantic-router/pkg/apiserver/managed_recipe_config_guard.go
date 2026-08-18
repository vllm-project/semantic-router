//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	recipeStoreDirEnv       = "VLLM_SR_RECIPE_STORE_DIR"
	runtimeStateRootDirEnv  = "VLLM_SR_STATE_ROOT_DIR"
	recipeConfigLockName    = "runtime-config.lock"
	activeRecipePointerName = "active.json"
	activationJournalName   = "activation-pending.json"
)

type configMutationGuard struct {
	lockFile *os.File
}

func (s *ClassificationAPIServer) acquireConfigMutationGuard(
	w http.ResponseWriter,
) (*configMutationGuard, bool) {
	if !deployMu.TryLock() {
		s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another config update operation is in progress. Please try again.")
		return nil, false
	}

	guard := &configMutationGuard{}
	lockFile, managedRecipeActive, err := acquireRecipeStoreConfigLock()
	if err != nil {
		deployMu.Unlock()
		if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
			s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another config update operation is in progress. Please try again.")
			return nil, false
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "RECIPE_STATE_GUARD_ERROR", "The managed Recipe state could not be guarded safely.")
		return nil, false
	}
	guard.lockFile = lockFile
	if managedRecipeActive {
		guard.Release()
		s.writeErrorResponse(w, http.StatusConflict, "MANAGED_RECIPE_ACTIVE", "Router config mutations are disabled while a managed Recipe package is active or activating.")
		return nil, false
	}
	return guard, true
}

func (g *configMutationGuard) Release() {
	if g == nil {
		return
	}
	if g.lockFile != nil {
		_ = unix.Flock(int(g.lockFile.Fd()), unix.LOCK_UN)
		_ = g.lockFile.Close()
		g.lockFile = nil
	}
	deployMu.Unlock()
}

// acquireRecipeStoreConfigLock obtains the same stack-scoped advisory lock as
// the Dashboard activation service. The managed-state check happens only after
// the lock is held, and callers retain the lock for their complete filesystem
// transaction, closing the cross-process TOCTOU window.
func acquireRecipeStoreConfigLock() (*os.File, bool, error) {
	storeDir, err := configuredRecipeStoreDir()
	if err != nil || storeDir == "" {
		return nil, false, err
	}
	storeFD, err := unix.Open(
		storeDir,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, false, fmt.Errorf("open Recipe store: %w", err)
	}
	defer func() { _ = unix.Close(storeFD) }()

	lockFD, err := openRecipeConfigLock(storeFD)
	if err != nil {
		return nil, false, err
	}
	closeLockFD := true
	defer func() {
		if closeLockFD {
			_ = unix.Close(lockFD)
		}
	}()

	managedRecipeActive, err := recipeManagedStateExistsAt(storeFD)
	if err != nil {
		_ = unix.Flock(lockFD, unix.LOCK_UN)
		return nil, false, err
	}
	lockFile := os.NewFile(uintptr(lockFD), filepath.Join(storeDir, recipeConfigLockName))
	if lockFile == nil {
		_ = unix.Flock(lockFD, unix.LOCK_UN)
		return nil, false, errors.New("create Recipe config lock handle")
	}
	closeLockFD = false
	return lockFile, managedRecipeActive, nil
}

func configuredRecipeStoreDir() (string, error) {
	storeDir := strings.TrimSpace(os.Getenv(recipeStoreDirEnv))
	if storeDir == "" {
		if requiresLocalRecipeStoreGuard() {
			return "", fmt.Errorf("%s is required for the local runtime workspace", recipeStoreDirEnv)
		}
		return "", nil
	}
	if !filepath.IsAbs(storeDir) {
		return "", fmt.Errorf("%s must be absolute", recipeStoreDirEnv)
	}
	return filepath.Clean(storeDir), nil
}

func openRecipeConfigLock(storeFD int) (int, error) {
	lockFD, err := unix.Openat(
		storeFD,
		recipeConfigLockName,
		unix.O_RDWR|unix.O_CREAT|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0o600,
	)
	if err != nil {
		return -1, fmt.Errorf("open Recipe config lock: %w", err)
	}
	var lockInfo unix.Stat_t
	if err = unix.Fstat(lockFD, &lockInfo); err == nil && lockInfo.Mode&unix.S_IFMT != unix.S_IFREG {
		err = errors.New("recipe config lock is not a regular file")
	}
	if err == nil {
		err = unix.Fchmod(lockFD, 0o600)
	}
	if err == nil {
		err = unix.Flock(lockFD, unix.LOCK_EX|unix.LOCK_NB)
	}
	if err != nil {
		_ = unix.Close(lockFD)
		return -1, err
	}
	return lockFD, nil
}

func requiresLocalRecipeStoreGuard() bool {
	stateRoot := strings.TrimSpace(os.Getenv(runtimeStateRootDirEnv))
	runtimePath := strings.TrimSpace(os.Getenv(runtimeConfigPathEnv))
	if !filepath.IsAbs(stateRoot) || !filepath.IsAbs(runtimePath) {
		return false
	}
	runtimeDir := filepath.Join(filepath.Clean(stateRoot), ".vllm-sr")
	cleanRuntimePath := filepath.Clean(runtimePath)
	base := filepath.Base(cleanRuntimePath)
	return filepath.Dir(cleanRuntimePath) == runtimeDir &&
		strings.HasPrefix(base, "runtime-config") &&
		strings.HasSuffix(base, ".yaml")
}

func recipeManagedStateExistsAt(storeFD int) (bool, error) {
	for _, name := range []string{activeRecipePointerName, activationJournalName} {
		var info unix.Stat_t
		err := unix.Fstatat(storeFD, name, &info, unix.AT_SYMLINK_NOFOLLOW)
		switch {
		case err == nil:
			// Presence is authoritative. Malformed files and symlinks remain a
			// fail-closed managed state rather than reopening ordinary writers.
			return true, nil
		case errors.Is(err, unix.ENOENT):
			continue
		default:
			return false, fmt.Errorf("inspect managed Recipe state: %w", err)
		}
	}
	return false, nil
}
