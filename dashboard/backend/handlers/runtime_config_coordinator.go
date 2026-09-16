package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

const recipeStoreDirectoryEnv = "VLLM_SR_RECIPE_STORE_DIR"

var errRuntimeConfigLockBusy = errors.New("runtime config coordination lock is busy")

type runtimeConfigMutationError struct {
	code   string
	status int
	safe   string
	cause  error
}

func (e *runtimeConfigMutationError) Error() string { return e.safe }

func (e *runtimeConfigMutationError) Unwrap() error { return e.cause }

func beginOrdinaryRuntimeConfigMutation(configDir string) (func(), error) {
	return beginRuntimeConfigMutation(runtimeRecipeStoreDir(configDir), false)
}

func beginRecipeActivationMutation(storeDir string) (func(), error) {
	return beginRuntimeConfigMutation(storeDir, true)
}

func beginRuntimeConfigMutation(storeDir string, allowManaged bool) (func(), error) {
	if !deployMu.TryLock() {
		return nil, &runtimeConfigMutationError{
			code:   "deploy_in_progress",
			status: http.StatusConflict,
			safe:   "Another config operation is in progress. Please try again.",
			cause:  errRuntimeConfigLockBusy,
		}
	}
	releaseDeploy := func() { deployMu.Unlock() }
	lock, exists, err := acquireRuntimeConfigStoreLock(storeDir)
	if err != nil {
		releaseDeploy()
		status, code, safe := http.StatusInternalServerError, "config_coordination_failed", "Runtime config coordination is unavailable."
		if errors.Is(err, errRuntimeConfigLockBusy) {
			status, code, safe = http.StatusConflict, "deploy_in_progress", "Another config operation is in progress. Please try again."
		}
		return nil, &runtimeConfigMutationError{code: code, status: status, safe: safe, cause: err}
	}
	if !exists {
		return releaseDeploy, nil
	}
	release := func() {
		_ = lock.close()
		releaseDeploy()
	}
	if !allowManaged {
		managed, markerErr := lock.hasManagedState()
		if markerErr != nil {
			release()
			return nil, &runtimeConfigMutationError{
				code:   "config_coordination_failed",
				status: http.StatusInternalServerError,
				safe:   "Runtime config coordination is unavailable.",
				cause:  markerErr,
			}
		}
		if managed {
			release()
			return nil, &runtimeConfigMutationError{
				code:   "managed_recipe_active",
				status: http.StatusConflict,
				safe:   "Deactivate the managed Recipe before editing the runtime config.",
				cause:  errors.New("managed Recipe state is present"),
			}
		}
	}
	return release, nil
}

func runtimeRecipeStoreDir(configDir string) string {
	if configured := strings.TrimSpace(os.Getenv(recipeStoreDirectoryEnv)); configured != "" {
		return filepath.Clean(configured)
	}
	if strings.TrimSpace(configDir) == "" {
		return ""
	}
	return filepath.Join(filepath.Clean(configDir), ".vllm-sr", "recipe-store")
}

func writeRuntimeConfigMutationError(w http.ResponseWriter, err error) {
	var mutationErr *runtimeConfigMutationError
	if !errors.As(err, &mutationErr) {
		mutationErr = &runtimeConfigMutationError{
			code:   "config_coordination_failed",
			status: http.StatusInternalServerError,
			safe:   "Runtime config coordination is unavailable.",
			cause:  err,
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(mutationErr.status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   mutationErr.code,
		"message": mutationErr.safe,
	})
}

func packageRuntimeConfigMutationError(err error) error {
	var mutationErr *runtimeConfigMutationError
	if !errors.As(err, &mutationErr) {
		return fmt.Errorf("runtime config coordination: %w", err)
	}
	return recipe.NewPackageError(mutationErr.code, mutationErr.status, mutationErr.safe, mutationErr.cause)
}
