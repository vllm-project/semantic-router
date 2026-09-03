package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

type fakeRecipePackageStore struct {
	listResult   recipe.PackageList
	importResult recipe.PackageSummary
	created      bool
	err          error
	listOptions  recipe.PackageListOptions
	importCalls  int
}

func (f *fakeRecipePackageStore) List(options recipe.PackageListOptions) (recipe.PackageList, error) {
	f.listOptions = options
	return f.listResult, f.err
}

func (f *fakeRecipePackageStore) Import(_ context.Context, _ recipe.ImportRequest) (recipe.PackageSummary, bool, error) {
	f.importCalls++
	return f.importResult, f.created, f.err
}

type fakeRecipePackageActivator struct {
	result                 recipe.ActivateResult
	deactivateResult       recipe.DeactivateResult
	err                    error
	calls                  int
	deactivateCalls        int
	previewCalls           int
	deactivatePreviewCalls int
	previewRequest         recipe.ActivateRequest
}

func (f *fakeRecipePackageActivator) Preview(_ context.Context, request recipe.ActivateRequest) (recipe.ActivationPlan, error) {
	f.previewCalls++
	f.previewRequest = request
	return recipe.ActivationPlan{RecipeDigest: request.RecipeDigest, PlanDigest: "sha256:" + strings.Repeat("f", 64), Mode: recipe.ActivationModeHotSwitch}, f.err
}

func (f *fakeRecipePackageActivator) Deactivate(context.Context, ...recipe.DeactivateRequest) (recipe.DeactivateResult, error) {
	f.deactivateCalls++
	return f.deactivateResult, f.err
}

func (f *fakeRecipePackageActivator) PreviewDeactivation(context.Context) (recipe.ActivationPlan, error) {
	f.deactivatePreviewCalls++
	return recipe.ActivationPlan{PlanDigest: "sha256:" + strings.Repeat("e", 64), Mode: recipe.ActivationModeHotSwitch}, f.err
}

func (f *fakeRecipePackageActivator) Activate(_ context.Context, _ recipe.ActivateRequest) (recipe.ActivateResult, error) {
	f.calls++
	return f.result, f.err
}

func TestRecipePackagesHandlerListContract(t *testing.T) {
	store := &fakeRecipePackageStore{listResult: recipe.PackageList{
		Items:           []recipe.PackageSummary{},
		Page:            2,
		PageSize:        10,
		Total:           0,
		TotalPages:      0,
		ActivationState: recipe.ActivationNone,
	}}
	handler := NewRecipeHandler(nil, WithRecipePackages(store, nil, false))
	response := httptest.NewRecorder()
	handler.Packages(response, httptest.NewRequest(http.MethodGet, "/api/recipe/packages/?page=2&page_size=10&q=digest", nil))

	if response.Code != http.StatusOK || store.listOptions != (recipe.PackageListOptions{Page: 2, PageSize: 10, Query: "digest"}) {
		t.Fatalf("response=%d options=%#v body=%s", response.Code, store.listOptions, response.Body.String())
	}
	if !strings.Contains(response.Header().Get("Cache-Control"), "no-store") {
		t.Fatalf("Cache-Control = %q", response.Header().Get("Cache-Control"))
	}
	var body recipe.PackageList
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.Items == nil || body.ActivationState != recipe.ActivationNone {
		t.Fatalf("body=%#v err=%v", body, err)
	}
}

func TestRecipePackageImportStatusStrictJSONAndReadonly(t *testing.T) {
	store := &fakeRecipePackageStore{
		importResult: recipe.PackageSummary{ID: "accuracy", RecipeDigest: "sha256:" + strings.Repeat("a", 64)},
		created:      true,
	}
	handler := NewRecipeHandler(nil, WithRecipePackages(store, nil, false))

	response := httptest.NewRecorder()
	handler.ImportPackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(`{"url":"https://example.com/recipe.zip"}`)))
	if response.Code != http.StatusCreated || store.importCalls != 1 {
		t.Fatalf("created response=%d calls=%d body=%s", response.Code, store.importCalls, response.Body.String())
	}
	store.created = false
	response = httptest.NewRecorder()
	handler.ImportPackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/import/", strings.NewReader(`{"url":"https://example.com/recipe.zip"}`)))
	if response.Code != http.StatusOK || store.importCalls != 2 {
		t.Fatalf("idempotent response=%d calls=%d body=%s", response.Code, store.importCalls, response.Body.String())
	}

	for name, body := range map[string]string{
		"unknown field": `{"url":"https://example.com/recipe.zip","secret":"no"}`,
		"two documents": `{"url":"https://example.com/recipe.zip"} {"url":"https://example.com/other.zip"}`,
	} {
		t.Run(name, func(t *testing.T) {
			before := store.importCalls
			invalidResponse := httptest.NewRecorder()
			handler.ImportPackage(invalidResponse, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(body)))
			if invalidResponse.Code != http.StatusBadRequest || store.importCalls != before {
				t.Fatalf("response=%d calls=%d body=%s", invalidResponse.Code, store.importCalls, invalidResponse.Body.String())
			}
		})
	}

	readonly := NewRecipeHandler(nil, WithRecipePackages(store, nil, true))
	before := store.importCalls
	response = httptest.NewRecorder()
	readonly.ImportPackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(`{"url":"https://example.com/recipe.zip"}`)))
	if response.Code != http.StatusForbidden || store.importCalls != before {
		t.Fatalf("readonly response=%d calls=%d", response.Code, store.importCalls)
	}
}

func TestRecipePackageRoutesRejectWrongMethodAndNestedTailWithoutMutation(t *testing.T) {
	store := &fakeRecipePackageStore{}
	activator := &fakeRecipePackageActivator{}
	handler := NewRecipeHandler(nil, WithRecipePackages(store, activator, false))
	tests := []struct {
		name   string
		method string
		path   string
		call   func(http.ResponseWriter, *http.Request)
		status int
	}{
		{name: "list method", method: http.MethodPost, path: "/api/recipe/packages", call: handler.Packages, status: http.StatusMethodNotAllowed},
		{name: "list tail", method: http.MethodGet, path: "/api/recipe/packages/extra", call: handler.Packages, status: http.StatusNotFound},
		{name: "import method", method: http.MethodGet, path: "/api/recipe/import", call: handler.ImportPackage, status: http.StatusMethodNotAllowed},
		{name: "import tail", method: http.MethodPost, path: "/api/recipe/import/anything", call: handler.ImportPackage, status: http.StatusNotFound},
		{name: "activate method", method: http.MethodGet, path: "/api/recipe/activate", call: handler.ActivatePackage, status: http.StatusMethodNotAllowed},
		{name: "activate tail", method: http.MethodPost, path: "/api/recipe/activate/anything", call: handler.ActivatePackage, status: http.StatusNotFound},
		{name: "deactivate method", method: http.MethodGet, path: "/api/recipe/deactivate", call: handler.DeactivatePackage, status: http.StatusMethodNotAllowed},
		{name: "deactivate tail", method: http.MethodPost, path: "/api/recipe/deactivate/anything", call: handler.DeactivatePackage, status: http.StatusNotFound},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			test.call(response, httptest.NewRequest(test.method, test.path, strings.NewReader(`{}`)))
			if response.Code != test.status {
				t.Fatalf("status=%d want=%d body=%s", response.Code, test.status, response.Body.String())
			}
		})
	}
	if store.importCalls != 0 || activator.calls != 0 || activator.deactivateCalls != 0 {
		t.Fatalf("nested/method routes mutated state: imports=%d activations=%d deactivations=%d", store.importCalls, activator.calls, activator.deactivateCalls)
	}
}

func TestRecipePackageDeactivateContractStrictJSONAndReadonly(t *testing.T) {
	activator := &fakeRecipePackageActivator{deactivateResult: recipe.DeactivateResult{Status: "inactive", PreviousRecipeDigest: "sha256:" + strings.Repeat("a", 64)}}
	handler := NewRecipeHandler(nil, WithRecipePackages(&fakeRecipePackageStore{}, activator, false))
	for _, body := range []string{"", `{}`} {
		response := httptest.NewRecorder()
		handler.DeactivatePackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/deactivate", strings.NewReader(body)))
		if response.Code != http.StatusOK {
			t.Fatalf("body=%q response=%d %s", body, response.Code, response.Body.String())
		}
	}
	before := activator.deactivateCalls
	response := httptest.NewRecorder()
	handler.DeactivatePackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/deactivate", strings.NewReader(`{"unexpected":true}`)))
	if response.Code != http.StatusBadRequest || activator.deactivateCalls != before {
		t.Fatalf("strict response=%d calls=%d body=%s", response.Code, activator.deactivateCalls, response.Body.String())
	}
	readonly := NewRecipeHandler(nil, WithRecipePackages(&fakeRecipePackageStore{}, activator, true))
	response = httptest.NewRecorder()
	readonly.DeactivatePackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/deactivate", nil))
	if response.Code != http.StatusForbidden || activator.deactivateCalls != before {
		t.Fatalf("readonly response=%d calls=%d", response.Code, activator.deactivateCalls)
	}
}

func TestRecipePackagePreviewContractsAreStrictAndSideEffectFree(t *testing.T) {
	digest := "sha256:" + strings.Repeat("c", 64)
	activator := &fakeRecipePackageActivator{}
	handler := NewRecipeHandler(nil, WithRecipePackages(&fakeRecipePackageStore{}, activator, false))

	activation := httptest.NewRecorder()
	handler.PreviewPackageActivation(activation, httptest.NewRequest(
		http.MethodPost,
		"/api/recipe/activate/preview",
		strings.NewReader(`{"recipe_digest":"`+digest+`","acknowledge_warnings":true}`),
	))
	if activation.Code != http.StatusOK || activator.previewCalls != 1 || activator.previewRequest.RecipeDigest != digest {
		t.Fatalf("activation preview status=%d calls=%d request=%#v body=%s", activation.Code, activator.previewCalls, activator.previewRequest, activation.Body.String())
	}

	deactivation := httptest.NewRecorder()
	handler.PreviewPackageDeactivation(deactivation, httptest.NewRequest(http.MethodPost, "/api/recipe/deactivate/preview", strings.NewReader(`{}`)))
	if deactivation.Code != http.StatusOK || activator.deactivatePreviewCalls != 1 {
		t.Fatalf("deactivation preview status=%d calls=%d body=%s", deactivation.Code, activator.deactivatePreviewCalls, deactivation.Body.String())
	}

	invalid := httptest.NewRecorder()
	handler.PreviewPackageActivation(invalid, httptest.NewRequest(http.MethodPost, "/api/recipe/activate/preview", strings.NewReader(`{"recipe_digest":"`+digest+`","unexpected":true}`)))
	if invalid.Code != http.StatusBadRequest || activator.previewCalls != 1 {
		t.Fatalf("invalid preview status=%d calls=%d body=%s", invalid.Code, activator.previewCalls, invalid.Body.String())
	}
	if activator.calls != 0 || activator.deactivateCalls != 0 {
		t.Fatalf("preview mutated runtime: activate=%d deactivate=%d", activator.calls, activator.deactivateCalls)
	}
}

func TestRecipePackageCapabilitiesKeepStoreAndRuntimeIndependent(t *testing.T) {
	t.Run("runtime config readonly still permits import", testRuntimeReadonlyRecipeCapabilities)
	t.Run("package store readonly still permits lifecycle operations", testStoreReadonlyRecipeCapabilities)
	t.Run("explicit server readonly hard denies both surfaces", testServerReadonlyRecipeCapabilities)
}

type recipePackageTestOperation struct {
	path string
	body string
	call func(http.ResponseWriter, *http.Request)
}

func recipeLifecycleTestOperations(handler *RecipeHandler, digest string) []recipePackageTestOperation {
	return []recipePackageTestOperation{
		{path: "/api/recipe/activate/preview", body: `{"recipe_digest":"` + digest + `"}`, call: handler.PreviewPackageActivation},
		{path: "/api/recipe/deactivate/preview", body: `{}`, call: handler.PreviewPackageDeactivation},
		{path: "/api/recipe/activate", body: `{"recipe_digest":"` + digest + `"}`, call: handler.ActivatePackage},
		{path: "/api/recipe/deactivate", body: `{}`, call: handler.DeactivatePackage},
	}
}

func assertRecipeOperationsForbidden(t *testing.T, operations []recipePackageTestOperation, errorCode string) {
	t.Helper()
	for _, operation := range operations {
		response := httptest.NewRecorder()
		operation.call(response, httptest.NewRequest(http.MethodPost, operation.path, strings.NewReader(operation.body)))
		if response.Code != http.StatusForbidden || !strings.Contains(response.Body.String(), `"error":"`+errorCode+`"`) {
			t.Fatalf("%s response=%d body=%s", operation.path, response.Code, response.Body.String())
		}
	}
}

func testRuntimeReadonlyRecipeCapabilities(t *testing.T) {
	digest := "sha256:" + strings.Repeat("d", 64)
	store := &fakeRecipePackageStore{created: true}
	activator := &fakeRecipePackageActivator{}
	handler := NewRecipeHandler(nil, WithRecipePackageCapabilities(store, activator, RecipePackageCapabilities{
		RuntimeConfigWritable: false,
		RecipeStoreWritable:   true,
	}))

	importResponse := httptest.NewRecorder()
	handler.ImportPackage(importResponse, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(`{"url":"https://example.com/recipe.zip"}`)))
	if importResponse.Code != http.StatusCreated || store.importCalls != 1 {
		t.Fatalf("import response=%d calls=%d body=%s", importResponse.Code, store.importCalls, importResponse.Body.String())
	}
	assertRecipeOperationsForbidden(t, recipeLifecycleTestOperations(handler, digest), "runtime_config_readonly")
	if activator.previewCalls != 0 || activator.deactivatePreviewCalls != 0 || activator.calls != 0 || activator.deactivateCalls != 0 {
		t.Fatalf("runtime-readonly operations reached activator: %#v", activator)
	}
}

func testStoreReadonlyRecipeCapabilities(t *testing.T) {
	digest := "sha256:" + strings.Repeat("d", 64)
	store := &fakeRecipePackageStore{}
	activator := &fakeRecipePackageActivator{}
	handler := NewRecipeHandler(nil, WithRecipePackageCapabilities(store, activator, RecipePackageCapabilities{
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   false,
	}))

	importResponse := httptest.NewRecorder()
	handler.ImportPackage(importResponse, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(`{"url":"https://example.com/recipe.zip"}`)))
	if importResponse.Code != http.StatusForbidden || store.importCalls != 0 || !strings.Contains(importResponse.Body.String(), `"error":"recipe_store_readonly"`) {
		t.Fatalf("import response=%d calls=%d body=%s", importResponse.Code, store.importCalls, importResponse.Body.String())
	}
	previewResponse := httptest.NewRecorder()
	handler.PreviewPackageActivation(previewResponse, httptest.NewRequest(http.MethodPost, "/api/recipe/activate/preview", strings.NewReader(`{"recipe_digest":"`+digest+`"}`)))
	if previewResponse.Code != http.StatusOK || activator.previewCalls != 1 {
		t.Fatalf("preview response=%d calls=%d body=%s", previewResponse.Code, activator.previewCalls, previewResponse.Body.String())
	}
}

func testServerReadonlyRecipeCapabilities(t *testing.T) {
	digest := "sha256:" + strings.Repeat("d", 64)
	store := &fakeRecipePackageStore{}
	activator := &fakeRecipePackageActivator{}
	handler := NewRecipeHandler(nil, WithRecipePackageCapabilities(store, activator, RecipePackageCapabilities{
		ServerReadonly:        true,
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   true,
	}))
	operations := append([]recipePackageTestOperation{{
		path: "/api/recipe/import", body: `{"url":"https://example.com/recipe.zip"}`, call: handler.ImportPackage,
	}}, recipeLifecycleTestOperations(handler, digest)...)
	assertRecipeOperationsForbidden(t, operations, "readonly_mode")
	if store.importCalls != 0 || activator.previewCalls != 0 || activator.deactivatePreviewCalls != 0 || activator.calls != 0 || activator.deactivateCalls != 0 {
		t.Fatalf("global readonly reached a mutation surface: store=%#v activator=%#v", store, activator)
	}
}

func TestRecipePackageActivationResultAndSafeErrorEnvelope(t *testing.T) {
	digest := "sha256:" + strings.Repeat("b", 64)
	activator := &fakeRecipePackageActivator{result: recipe.ActivateResult{Status: "active", RecipeDigest: digest}}
	handler := NewRecipeHandler(nil, WithRecipePackages(&fakeRecipePackageStore{}, activator, false))
	response := httptest.NewRecorder()
	handler.ActivatePackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/activate", strings.NewReader(`{"recipe_digest":"`+digest+`"}`)))
	if response.Code != http.StatusOK || activator.calls != 1 {
		t.Fatalf("response=%d calls=%d body=%s", response.Code, activator.calls, response.Body.String())
	}

	activator.err = recipe.NewPackageError(recipe.ErrorWarningsNotAcknowledged, http.StatusConflict, "Warnings must be acknowledged.", errors.New("secret-value"))
	response = httptest.NewRecorder()
	handler.ActivatePackage(response, httptest.NewRequest(http.MethodPost, "/api/recipe/activate", strings.NewReader(`{"recipe_digest":"`+digest+`"}`)))
	if response.Code != http.StatusConflict || strings.Contains(response.Body.String(), "secret-value") || !strings.Contains(response.Body.String(), `"error":"warnings_not_acknowledged"`) {
		t.Fatalf("error response=%d body=%s", response.Code, response.Body.String())
	}
}
