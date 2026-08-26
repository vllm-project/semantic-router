package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

const maxRecipePackageRequestBytes = 64 << 10

const (
	recipeStoreReadonlyError   = "recipe_store_readonly"
	runtimeConfigReadonlyError = "runtime_config_readonly"
)

type recipePackageStore interface {
	List(recipe.PackageListOptions) (recipe.PackageList, error)
	Import(context.Context, recipe.ImportRequest) (recipe.PackageSummary, bool, error)
}

type recipePackageActivator interface {
	Preview(context.Context, recipe.ActivateRequest) (recipe.ActivationPlan, error)
	PreviewDeactivation(context.Context) (recipe.ActivationPlan, error)
	Activate(context.Context, recipe.ActivateRequest) (recipe.ActivateResult, error)
	Deactivate(context.Context, ...recipe.DeactivateRequest) (recipe.DeactivateResult, error)
}

func (h *RecipeHandler) PreviewPackageDeactivation(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/deactivate/preview") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodPost {
		writePackageMethodNotAllowed(w, http.MethodPost)
		return
	}
	if !h.runtimeConfigMutationAllowed(w) {
		return
	}
	if h.activator == nil {
		writePackageUnavailable(w)
		return
	}
	if r.ContentLength != 0 {
		var request struct{}
		if err := decodePackageRequest(w, r, &request); err != nil {
			writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Recipe deactivation preview request is invalid.", err))
			return
		}
	}
	result, err := h.activator.PreviewDeactivation(r.Context())
	if err != nil {
		writePackageError(w, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

func (h *RecipeHandler) PreviewPackageActivation(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/activate/preview") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodPost {
		writePackageMethodNotAllowed(w, http.MethodPost)
		return
	}
	if !h.runtimeConfigMutationAllowed(w) {
		return
	}
	if h.activator == nil {
		writePackageUnavailable(w)
		return
	}
	var request recipe.ActivateRequest
	if err := decodePackageRequest(w, r, &request); err != nil {
		writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Recipe activation preview request is invalid.", err))
		return
	}
	result, err := h.activator.Preview(r.Context(), request)
	if err != nil {
		writePackageError(w, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

type RecipeHandlerOption func(*RecipeHandler)

// RecipePackageCapabilities keeps the package store and runtime-config
// persistence surfaces independent. ServerReadonly is the explicit global
// hard deny; the writable fields reflect mount/store availability.
type RecipePackageCapabilities struct {
	ServerReadonly        bool
	RuntimeConfigWritable bool
	RecipeStoreWritable   bool
}

// WithRecipePackages preserves the original all-or-nothing test/embedding
// seam. Production wiring uses WithRecipePackageCapabilities so a read-only
// runtime config does not disable an independently writable package store.
func WithRecipePackages(store recipePackageStore, activator recipePackageActivator, readonly bool) RecipeHandlerOption {
	return WithRecipePackageCapabilities(store, activator, RecipePackageCapabilities{
		ServerReadonly:        readonly,
		RuntimeConfigWritable: !readonly,
		RecipeStoreWritable:   !readonly,
	})
}

func WithRecipePackageCapabilities(store recipePackageStore, activator recipePackageActivator, capabilities RecipePackageCapabilities) RecipeHandlerOption {
	return func(handler *RecipeHandler) {
		handler.packages = store
		handler.activator = activator
		handler.packageCapabilities = capabilities
	}
}

func (h *RecipeHandler) packageImportAllowed(w http.ResponseWriter) bool {
	if h.packageCapabilities.ServerReadonly {
		writePackageError(w, recipe.NewPackageError("readonly_mode", http.StatusForbidden, "Dashboard is in read-only mode.", nil))
		return false
	}
	if !h.packageCapabilities.RecipeStoreWritable {
		writePackageError(w, recipe.NewPackageError(recipeStoreReadonlyError, http.StatusForbidden, "Recipe package store is read-only.", nil))
		return false
	}
	return true
}

func (h *RecipeHandler) runtimeConfigMutationAllowed(w http.ResponseWriter) bool {
	if h.packageCapabilities.ServerReadonly {
		writePackageError(w, recipe.NewPackageError("readonly_mode", http.StatusForbidden, "Dashboard is in read-only mode.", nil))
		return false
	}
	if !h.packageCapabilities.RuntimeConfigWritable {
		writePackageError(w, recipe.NewPackageError(runtimeConfigReadonlyError, http.StatusForbidden, "Runtime configuration is read-only.", nil))
		return false
	}
	return true
}

func (h *RecipeHandler) Packages(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/packages") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodGet {
		writePackageMethodNotAllowed(w, http.MethodGet)
		return
	}
	if h.packages == nil {
		writePackageUnavailable(w)
		return
	}
	page, err := optionalPositiveInt(r, "page")
	if err != nil {
		writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Invalid package page.", err))
		return
	}
	pageSize, err := optionalPositiveInt(r, "page_size")
	if err != nil {
		writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Invalid package page size.", err))
		return
	}
	result, err := h.packages.List(recipe.PackageListOptions{Page: page, PageSize: pageSize, Query: r.URL.Query().Get("q")})
	if err != nil {
		writePackageError(w, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

func (h *RecipeHandler) ImportPackage(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/import") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodPost {
		writePackageMethodNotAllowed(w, http.MethodPost)
		return
	}
	if !h.packageImportAllowed(w) {
		return
	}
	if h.packages == nil {
		writePackageUnavailable(w)
		return
	}
	var request recipe.ImportRequest
	if err := decodePackageRequest(w, r, &request); err != nil {
		writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Recipe import request is invalid.", err))
		return
	}
	if rejectRevokedMutation(w, r) {
		return
	}
	result, created, err := h.packages.Import(r.Context(), request)
	if err != nil {
		writePackageError(w, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	writeRecipeJSON(w, status, result)
}

func (h *RecipeHandler) ActivatePackage(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/activate") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodPost {
		writePackageMethodNotAllowed(w, http.MethodPost)
		return
	}
	if !h.runtimeConfigMutationAllowed(w) {
		return
	}
	if h.activator == nil {
		writePackageUnavailable(w)
		return
	}
	var request recipe.ActivateRequest
	if err := decodePackageRequest(w, r, &request); err != nil {
		writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Recipe activation request is invalid.", err))
		return
	}
	if rejectRevokedMutation(w, r) {
		return
	}
	result, err := h.activator.Activate(r.Context(), request)
	if err != nil {
		writePackageError(w, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

func (h *RecipeHandler) DeactivatePackage(w http.ResponseWriter, r *http.Request) {
	if !canonicalRecipePackagePath(r.URL.Path, "/api/recipe/deactivate") {
		writeRecipeJSON(w, http.StatusNotFound, map[string]string{"error": "not_found", "message": "Recipe package route was not found."})
		return
	}
	if r.Method != http.MethodPost {
		writePackageMethodNotAllowed(w, http.MethodPost)
		return
	}
	if !h.runtimeConfigMutationAllowed(w) {
		return
	}
	if h.activator == nil {
		writePackageUnavailable(w)
		return
	}
	request := recipe.DeactivateRequest{}
	if r.ContentLength != 0 {
		if err := decodePackageRequest(w, r, &request); err != nil {
			writePackageError(w, recipe.NewPackageError(recipe.ErrorInvalidRequest, http.StatusBadRequest, "Recipe deactivation request is invalid.", err))
			return
		}
	}
	if rejectRevokedMutation(w, r) {
		return
	}
	result, err := h.activator.Deactivate(r.Context(), request)
	if err != nil {
		writePackageError(w, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

func canonicalRecipePackagePath(actual, expected string) bool {
	return actual == expected || actual == expected+"/"
}

func decodePackageRequest(w http.ResponseWriter, r *http.Request, target any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRecipePackageRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request must contain exactly one JSON document")
		}
		return err
	}
	return nil
}

func writePackageUnavailable(w http.ResponseWriter) {
	writePackageError(w, recipe.NewPackageError("service_unavailable", http.StatusServiceUnavailable, "Recipe package service is unavailable.", nil))
}

func writePackageMethodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeRecipeJSON(w, http.StatusMethodNotAllowed, map[string]string{
		"error":   "method_not_allowed",
		"message": "Recipe package route does not support this method.",
	})
}

func writePackageError(w http.ResponseWriter, err error) {
	packageErr, ok := recipe.AsPackageError(err)
	if !ok {
		packageErr = &recipe.PackageError{Code: recipe.ErrorActivationFailed, Status: http.StatusInternalServerError, Safe: "Recipe package operation failed."}
	}
	if packageErr.Status >= http.StatusInternalServerError {
		log.Printf("Recipe package operation failed: %v", redactPackageLogError(err))
	}
	writeRecipeJSON(w, packageErr.Status, map[string]string{
		"error":   packageErr.Code,
		"message": packageErr.Safe,
	})
}

func redactPackageLogError(err error) string {
	if packageErr, ok := recipe.AsPackageError(err); ok {
		return packageErr.Code
	}
	return "internal package operation error"
}
