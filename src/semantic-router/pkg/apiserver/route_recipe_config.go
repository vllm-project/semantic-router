//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type managedRecipeSource string

const (
	managedRecipeSourceDefault managedRecipeSource = "routing"
	managedRecipeSourceNamed   managedRecipeSource = "recipes"
)

type recipeMutationRequest struct {
	Name        string         `json:"name,omitempty"`
	Description *string        `json:"description,omitempty"`
	Routing     map[string]any `json:"routing"`
	Entrypoints *[]string      `json:"entrypoints,omitempty"`
}

type managedRecipeRecord struct {
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Routing     map[string]any      `json:"routing"`
	Entrypoints []string            `json:"entrypoints"`
	Source      managedRecipeSource `json:"source"`
	Deletable   bool                `json:"deletable"`
}

type recipeCollectionResponse struct {
	ETag    string                `json:"etag"`
	Recipes []managedRecipeRecord `json:"recipes"`
}

// handleListRecipes exposes the normalized management view without forcing a
// client to download and reshape the whole canonical document.
func (s *ClassificationAPIServer) handleListRecipes(w http.ResponseWriter, r *http.Request) {
	doc, data, ok := s.readManagedConfigDocument(w)
	if !ok {
		return
	}

	etag := configDocumentETag(data)
	w.Header().Set("ETag", etag)
	response := recipeCollectionResponse{
		ETag:    etag,
		Recipes: collectManagedRecipes(doc),
	}
	response.Recipes = s.maybeRedactManagedRecipes(r, response.Recipes)
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleGetRecipe(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE_NAME", "recipe name is required")
		return
	}

	doc, data, ok := s.readManagedConfigDocument(w)
	if !ok {
		return
	}
	recipe, found := findManagedRecipe(doc, name)
	if !found {
		s.writeErrorResponse(w, http.StatusNotFound, "RECIPE_NOT_FOUND", fmt.Sprintf("Recipe %q does not exist", name))
		return
	}

	w.Header().Set("ETag", configDocumentETag(data))
	recipe = s.maybeRedactManagedRecipes(r, []managedRecipeRecord{recipe})[0]
	s.writeJSONResponse(w, http.StatusOK, recipe)
}

// handleValidateRecipe performs the exact document-level validation used by a
// live mutation, but does not create a backup or write either config path.
func (s *ClassificationAPIServer) handleValidateRecipe(w http.ResponseWriter, r *http.Request) {
	var req recipeMutationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE_NAME", "name is required")
		return
	}

	doc, _, ok := s.readManagedConfigDocument(w)
	if !ok {
		return
	}
	created, err := applyRecipeMutation(doc, name, req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE", err.Error())
		return
	}
	if _, err := validateAndEncodeRouterConfigDocument(doc); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]any{
		"valid":  true,
		"name":   name,
		"action": map[bool]string{true: "create", false: "replace"}[created],
	})
}

func (s *ClassificationAPIServer) handlePutRecipe(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE_NAME", "recipe name is required")
		return
	}
	guard, ok := s.acquireConfigMutationGuard(w)
	if !ok {
		return
	}
	defer guard.Release()

	var req recipeMutationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if requestName := strings.TrimSpace(req.Name); requestName != "" && requestName != name {
		s.writeErrorResponse(w, http.StatusBadRequest, "RECIPE_NAME_MISMATCH", "body name must match the recipe name in the path")
		return
	}

	doc, existingData, ok := s.readManagedConfigDocument(w)
	if !ok || !checkConfigPrecondition(w, r, existingData, true) {
		return
	}
	created, err := applyRecipeMutation(doc, name, req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE", err.Error())
		return
	}
	yamlBytes, err := validateAndEncodeRouterConfigDocument(doc)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", err.Error())
		return
	}

	statusCode := http.StatusOK
	action := "recipe.replace"
	verb := "replaced"
	if created {
		statusCode = http.StatusCreated
		action = "recipe.create"
		verb = "created"
	}
	paths := resolveConfigPersistencePaths(s.configPath)
	s.commitRouterConfigDocument(
		w,
		paths,
		existingData,
		yamlBytes,
		"",
		statusCode,
		action,
		fmt.Sprintf("Recipe %q %s successfully.", name, verb),
	)
}

func (s *ClassificationAPIServer) handleDeleteRecipe(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_RECIPE_NAME", "recipe name is required")
		return
	}
	if routerconfig.RecipeName(name) == routerconfig.DefaultRecipeName {
		s.writeErrorResponse(w, http.StatusConflict, "DEFAULT_RECIPE_REQUIRED", "The default recipe is the top-level routing profile and cannot be deleted")
		return
	}
	guard, ok := s.acquireConfigMutationGuard(w)
	if !ok {
		return
	}
	defer guard.Release()

	doc, existingData, ok := s.readManagedConfigDocument(w)
	if !ok || !checkConfigPrecondition(w, r, existingData, true) {
		return
	}
	entrypoints := recipeEntrypointNames(doc, name)
	if len(entrypoints) > 0 {
		s.writeErrorResponse(
			w,
			http.StatusConflict,
			"RECIPE_IN_USE",
			fmt.Sprintf("Recipe %q is still referenced by entrypoints %s; replace or remove those mappings first", name, strings.Join(entrypoints, ", ")),
		)
		return
	}
	if !removeNamedRecipe(doc, name) {
		s.writeErrorResponse(w, http.StatusNotFound, "RECIPE_NOT_FOUND", fmt.Sprintf("Recipe %q does not exist", name))
		return
	}

	yamlBytes, err := validateAndEncodeRouterConfigDocument(doc)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", err.Error())
		return
	}
	paths := resolveConfigPersistencePaths(s.configPath)
	s.commitRouterConfigDocument(
		w,
		paths,
		existingData,
		yamlBytes,
		"",
		http.StatusOK,
		"recipe.delete",
		fmt.Sprintf("Recipe %q deleted successfully.", name),
	)
}

func (s *ClassificationAPIServer) readManagedConfigDocument(w http.ResponseWriter) (map[string]any, []byte, bool) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return nil, nil, false
	}
	paths := resolveConfigPersistencePaths(s.configPath)
	doc, data, err := readConfigDocument(paths.sourcePath)
	if err != nil {
		status := http.StatusInternalServerError
		if os.IsNotExist(err) {
			status = http.StatusNotFound
		}
		s.writeErrorResponse(w, status, "READ_ERROR", fmt.Sprintf("Failed to read router config: %v", err))
		return nil, nil, false
	}
	return doc, data, true
}

func collectManagedRecipes(doc map[string]any) []managedRecipeRecord {
	recipes := []managedRecipeRecord{{
		Name:        string(routerconfig.DefaultRecipeName),
		Description: "Default routing profile",
		Routing:     mappingValue(doc["routing"]),
		Entrypoints: []string{},
		Source:      managedRecipeSourceDefault,
		Deletable:   false,
	}}
	for _, item := range sequenceValue(doc["recipes"]) {
		recipe := mappingValue(item)
		name := strings.TrimSpace(stringValue(recipe["name"]))
		if name == "" {
			continue
		}
		recipes = append(recipes, managedRecipeRecord{
			Name:        name,
			Description: stringValue(recipe["description"]),
			Routing:     mappingValue(recipe["routing"]),
			Entrypoints: recipeEntrypointNames(doc, name),
			Source:      managedRecipeSourceNamed,
			Deletable:   true,
		})
	}
	return recipes
}

func findManagedRecipe(doc map[string]any, name string) (managedRecipeRecord, bool) {
	for _, recipe := range collectManagedRecipes(doc) {
		if recipe.Name == name {
			return recipe, true
		}
	}
	return managedRecipeRecord{}, false
}

func applyRecipeMutation(doc map[string]any, name string, req recipeMutationRequest) (bool, error) {
	if len(req.Routing) == 0 {
		return false, fmt.Errorf("routing is required and must not be empty")
	}
	if routerconfig.RecipeName(name) == routerconfig.DefaultRecipeName {
		if req.Description != nil && strings.TrimSpace(*req.Description) != "" {
			return false, fmt.Errorf("the default recipe has no description field; update its top-level routing block only")
		}
		if req.Entrypoints != nil {
			return false, fmt.Errorf("default recipe aliases are managed by global.router.auto_model_names, not entrypoints")
		}
		doc["routing"] = cloneYAMLValue(req.Routing)
		return false, nil
	}

	recipes := sequenceValue(doc["recipes"])
	replacement := map[string]any{
		"name":    name,
		"routing": cloneYAMLValue(req.Routing),
	}
	if req.Description != nil && strings.TrimSpace(*req.Description) != "" {
		replacement["description"] = strings.TrimSpace(*req.Description)
	}

	created := true
	for index, item := range recipes {
		recipe := mappingValue(item)
		if strings.TrimSpace(stringValue(recipe["name"])) != name {
			continue
		}
		recipes[index] = replacement
		created = false
		break
	}
	if created {
		recipes = append(recipes, replacement)
	}
	doc["recipes"] = recipes
	if req.Entrypoints != nil {
		replaceRecipeEntrypoints(doc, name, *req.Entrypoints)
	}
	return created, nil
}

func replaceRecipeEntrypoints(doc map[string]any, recipeName string, modelNames []string) {
	normalized := make([]string, 0, len(modelNames))
	for _, modelName := range modelNames {
		modelName = strings.TrimSpace(modelName)
		if modelName != "" && !slices.Contains(normalized, modelName) {
			normalized = append(normalized, modelName)
		}
	}

	entrypoints := sequenceValue(doc["entrypoints"])
	filtered := make([]any, 0, len(entrypoints)+1)
	inserted := false
	for _, item := range entrypoints {
		entrypoint := mappingValue(item)
		if strings.TrimSpace(stringValue(entrypoint["recipe"])) != recipeName {
			filtered = append(filtered, item)
			continue
		}
		if !inserted && len(normalized) > 0 {
			filtered = append(filtered, map[string]any{
				"model_names": append([]string(nil), normalized...),
				"recipe":      recipeName,
			})
			inserted = true
		}
	}
	if !inserted && len(normalized) > 0 {
		filtered = append(filtered, map[string]any{
			"model_names": append([]string(nil), normalized...),
			"recipe":      recipeName,
		})
	}
	doc["entrypoints"] = filtered
}

func recipeEntrypointNames(doc map[string]any, recipeName string) []string {
	var names []string
	for _, item := range sequenceValue(doc["entrypoints"]) {
		entrypoint := mappingValue(item)
		if strings.TrimSpace(stringValue(entrypoint["recipe"])) != recipeName {
			continue
		}
		for _, modelName := range sequenceValue(entrypoint["model_names"]) {
			name := strings.TrimSpace(stringValue(modelName))
			if name != "" && !slices.Contains(names, name) {
				names = append(names, name)
			}
		}
	}
	return names
}

func removeNamedRecipe(doc map[string]any, name string) bool {
	recipes := sequenceValue(doc["recipes"])
	filtered := make([]any, 0, len(recipes))
	removed := false
	for _, item := range recipes {
		recipe := mappingValue(item)
		if strings.TrimSpace(stringValue(recipe["name"])) == name {
			removed = true
			continue
		}
		filtered = append(filtered, item)
	}
	if removed {
		doc["recipes"] = filtered
	}
	return removed
}

func mappingValue(value any) map[string]any {
	if mapping, ok := value.(map[string]any); ok {
		return mapping
	}
	return map[string]any{}
}

func sequenceValue(value any) []any {
	switch sequence := value.(type) {
	case []any:
		return sequence
	case []string:
		items := make([]any, len(sequence))
		for index, item := range sequence {
			items[index] = item
		}
		return items
	default:
		return []any{}
	}
}

func stringValue(value any) string {
	if value == nil {
		return ""
	}
	return fmt.Sprint(value)
}

func (s *ClassificationAPIServer) maybeRedactManagedRecipes(
	r *http.Request,
	recipes []managedRecipeRecord,
) []managedRecipeRecord {
	if s.canViewSecrets(r) {
		return recipes
	}
	redacted := make([]managedRecipeRecord, len(recipes))
	copy(redacted, recipes)
	for index := range redacted {
		if routing, ok := redactSensitiveConfigValue(redacted[index].Routing).(map[string]any); ok {
			redacted[index].Routing = routing
		}
	}
	return redacted
}
