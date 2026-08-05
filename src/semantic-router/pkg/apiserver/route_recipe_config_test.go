//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRecipeLifecycleRequiresFreshETagAndProtectsReferences(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	server := &ClassificationAPIServer{configPath: configPath}
	etag := initialRecipeCollectionETag(t, server)

	createBody := recipeMutationRequest{
		Description: stringPointer("Managed test recipe"),
		Routing:     managedTestRouting("managed_route"),
		Entrypoints: stringSlicePointer([]string{"vllm-sr/managed"}),
	}
	missingPrecondition := executeRecipePut(t, server, "managed", createBody, "")
	requireRecipeResponseCode(t, missingPrecondition, configPreconditionRequiredStatus, "missing If-Match")

	created := executeRecipePut(t, server, "managed", createBody, etag)
	requireRecipeResponseCode(t, created, http.StatusCreated, "create recipe")
	createdETag := requireAdvancedRecipeETag(t, created, etag)
	requireManagedRecipeConfig(t, configPath)

	staleUpdate := executeRecipePut(t, server, "managed", createBody, etag)
	requireRecipeResponseCode(t, staleUpdate, http.StatusPreconditionFailed, "stale update")

	deleteReferenced := executeRecipeDelete(t, server, "managed", createdETag)
	requireRecipeResponseCode(t, deleteReferenced, http.StatusConflict, "referenced delete")

	detachBody := createBody
	detachBody.Entrypoints = stringSlicePointer([]string{})
	detached := executeRecipePut(t, server, "managed", detachBody, createdETag)
	requireRecipeResponseCode(t, detached, http.StatusOK, "detach recipe entrypoint")

	deleted := executeRecipeDelete(t, server, "managed", detached.Header().Get("ETag"))
	requireRecipeResponseCode(t, deleted, http.StatusOK, "delete recipe")
	requireRecipeAbsent(t, configPath, "managed")
}

func initialRecipeCollectionETag(t *testing.T, server *ClassificationAPIServer) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/config/router/recipes", nil)
	rr := httptest.NewRecorder()
	server.handleListRecipes(rr, req)
	requireRecipeResponseCode(t, rr, http.StatusOK, "list recipes")

	etag := rr.Header().Get("ETag")
	if etag == "" {
		t.Fatal("list recipes did not return an ETag")
	}
	var collection recipeCollectionResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &collection); err != nil {
		t.Fatalf("decode recipe collection: %v", err)
	}
	if len(collection.Recipes) != 1 || collection.Recipes[0].Name != "default" {
		t.Fatalf("unexpected initial recipes: %+v", collection.Recipes)
	}
	return etag
}

func requireRecipeResponseCode(t *testing.T, rr *httptest.ResponseRecorder, want int, action string) {
	t.Helper()
	if rr.Code != want {
		t.Fatalf("%s: status=%d, want=%d body=%s", action, rr.Code, want, rr.Body.String())
	}
}

func requireAdvancedRecipeETag(t *testing.T, rr *httptest.ResponseRecorder, previous string) string {
	t.Helper()
	etag := rr.Header().Get("ETag")
	if etag == "" || etag == previous {
		t.Fatalf("recipe mutation did not advance ETag: before=%q after=%q", previous, etag)
	}
	return etag
}

func requireManagedRecipeConfig(t *testing.T, configPath string) {
	t.Helper()
	parsed, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("parse config after recipe create: %v", err)
	}
	if recipe, ok := parsed.RecipeByName("managed"); !ok || len(recipe.Profile.Decisions) != 1 {
		t.Fatalf("managed recipe was not created: %+v", recipe)
	}
	if recipe, ok := parsed.RecipeForRequestModel("vllm-sr/managed"); !ok || recipe.Name != "managed" {
		t.Fatalf("managed entrypoint was not created: %+v", recipe)
	}
}

func requireRecipeAbsent(t *testing.T, configPath, recipeName string) {
	t.Helper()
	parsed, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("parse config after recipe delete: %v", err)
	}
	if _, ok := parsed.RecipeByName(config.RecipeName(recipeName)); ok {
		t.Fatalf("recipe %q still exists after delete", recipeName)
	}
}

func TestValidateRecipeDoesNotWriteConfig(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	server := &ClassificationAPIServer{configPath: configPath}
	originalETag := configDocumentETag(mustReadFile(t, configPath))

	body := recipeMutationRequest{
		Name:        "preview",
		Description: stringPointer("Preview recipe"),
		Routing:     managedTestRouting("preview_route"),
		Entrypoints: stringSlicePointer([]string{"vllm-sr/preview"}),
	}
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal validate request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/config/router/recipes/validate", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	server.handleValidateRecipe(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("validate recipe: status=%d body=%s", rr.Code, rr.Body.String())
	}
	if got := configDocumentETag(mustReadFile(t, configPath)); got != originalETag {
		t.Fatalf("validate-only request changed config: before=%s after=%s", originalETag, got)
	}
}

func executeRecipePut(
	t *testing.T,
	server *ClassificationAPIServer,
	name string,
	body recipeMutationRequest,
	etag string,
) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal recipe request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPut, "/config/router/recipes/"+name, bytes.NewReader(payload))
	req.SetPathValue("name", name)
	req.Header.Set("Content-Type", "application/json")
	if etag != "" {
		req.Header.Set("If-Match", etag)
	}
	rr := httptest.NewRecorder()
	server.handlePutRecipe(rr, req)
	return rr
}

func executeRecipeDelete(
	t *testing.T,
	server *ClassificationAPIServer,
	name string,
	etag string,
) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodDelete, "/config/router/recipes/"+name, nil)
	req.SetPathValue("name", name)
	req.Header.Set("If-Match", etag)
	rr := httptest.NewRecorder()
	server.handleDeleteRecipe(rr, req)
	return rr
}

func managedTestRouting(decisionName string) map[string]any {
	return map[string]any{
		"strategy": "priority",
		"decisions": []any{
			map[string]any{
				"name":     decisionName,
				"priority": 100,
				"rules": map[string]any{
					"operator":   "AND",
					"conditions": []any{},
				},
				"modelRefs": []any{map[string]any{"model": "qwen-test"}},
				"algorithm": map[string]any{"type": "static"},
			},
		},
	}
}

func stringPointer(value string) *string { return &value }

func stringSlicePointer(value []string) *[]string { return &value }

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return data
}
