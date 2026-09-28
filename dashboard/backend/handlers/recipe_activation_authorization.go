package handlers

import (
	"context"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

// Preparation can run subprocesses and inspect runtime topology. Recheck the
// request's live authorization after that work, before publishing any state.
func revalidateRecipeMutation(ctx context.Context) error {
	err := auth.RevalidateContextIfPresent(ctx)
	if err == nil {
		return nil
	}
	if errors.Is(err, auth.ErrPermissionDenied) {
		return recipe.NewPackageError("permission_revoked", http.StatusForbidden, "Recipe mutation permission was revoked.", err)
	}
	return recipe.NewPackageError("unauthorized", http.StatusUnauthorized, "Recipe mutation session is no longer valid.", err)
}

func isRecipeAuthorizationError(err error) bool {
	packageErr, ok := recipe.AsPackageError(err)
	return ok && (packageErr.Status == http.StatusForbidden || packageErr.Status == http.StatusUnauthorized)
}
