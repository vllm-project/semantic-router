package handlers

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

const (
	managedStorageIsolationRemediation = "Managed storage must use loopback-only host port bindings. Preserve its data mounts, recreate the affected managed storage container with loopback-only bindings, and retry. No Recipe activation or source restore was started."
	managedRouterIsolationRemediation  = "Managed Router management API host publication must use exactly one loopback-only binding that matches the active config. Recreate the managed Router with the CLI-managed stack and retry. No Recipe activation or source restore was started."
)

func packageTopologyInventoryError(safeMessage string, err error) error {
	if isUnsafeManagedRouterInventoryError(err) {
		return recipe.NewPackageError(
			recipe.ErrorActivationIncompatible,
			http.StatusConflict,
			managedRouterIsolationRemediation,
			err,
		)
	}
	if isUnsafeManagedStorageInventoryError(err) {
		return recipe.NewPackageError(
			recipe.ErrorManagedStorageIsolation,
			http.StatusConflict,
			managedStorageIsolationRemediation,
			err,
		)
	}
	return activationFailed(safeMessage, err)
}

func isUnsafeManagedRouterInventoryError(err error) bool {
	return err != nil && strings.HasPrefix(err.Error(), "managed Router management publication is unsafe:")
}

func isUnsafeManagedStorageInventoryError(err error) bool {
	if err == nil {
		return false
	}
	detail := err.Error()
	return strings.HasPrefix(detail, "managed ") && strings.Contains(detail, " storage is unsafe:")
}
