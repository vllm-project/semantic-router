package handlers

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

var recipeDigestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// RecipeHandler is the HTTP transport for the process-scoped active Recipe.
// Parsing, indexing, materialization, and strict Router evaluation stay in the
// recipe service.
type RecipeHandler struct {
	service             *recipe.Service
	packages            recipePackageStore
	activator           recipePackageActivator
	packageCapabilities RecipePackageCapabilities
}

func NewRecipeHandler(service *recipe.Service, options ...RecipeHandlerOption) *RecipeHandler {
	handler := &RecipeHandler{service: service}
	for _, option := range options {
		option(handler)
	}
	return handler
}

func (h *RecipeHandler) Descriptor(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeRecipeMethodNotAllowed(w, http.MethodGet)
		return
	}
	descriptor, err := h.service.Describe()
	if err != nil {
		status := recipeErrorStatus(err)
		if errors.Is(err, recipe.ErrInvalid) {
			writeRecipeJSON(w, status, descriptor)
			return
		}
		writeRecipeError(w, status, err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, descriptor)
}

func (h *RecipeHandler) Probes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeRecipeMethodNotAllowed(w, http.MethodGet)
		return
	}
	page, err := optionalPositiveInt(r, "page")
	if err != nil {
		writeRecipeError(w, http.StatusBadRequest, err)
		return
	}
	pageSize, err := optionalPositiveInt(r, "page_size")
	if err != nil {
		writeRecipeError(w, http.StatusBadRequest, err)
		return
	}
	result, err := h.service.ListProbes(recipe.ListOptions{
		Page:     page,
		PageSize: pageSize,
		Query:    r.URL.Query().Get("q"),
		Recipe:   r.URL.Query().Get("recipe"),
		Decision: r.URL.Query().Get("decision"),
		Tag:      r.URL.Query().Get("tag"),
		Model:    r.URL.Query().Get("model"),
		Shape:    r.URL.Query().Get("shape"),
	})
	if err != nil {
		writeRecipeError(w, recipeErrorStatus(err), err)
		return
	}
	writeRecipeJSON(w, http.StatusOK, result)
}

func (h *RecipeHandler) ProbeAction(w http.ResponseWriter, r *http.Request) {
	decisionID, variantID, action, ok := parseProbeActionPath(r.URL.Path)
	if !ok {
		writeRecipeError(w, http.StatusNotFound, recipe.ErrProbeNotFound)
		return
	}
	if action == "" {
		h.serveProbeDetail(w, r, decisionID, variantID)
		return
	}
	h.serveProbeMutation(w, r, decisionID, variantID, action)
}

func parseProbeActionPath(path string) (string, string, string, bool) {
	relative := strings.Trim(strings.TrimPrefix(path, "/api/recipe/probes/"), "/")
	parts := strings.Split(relative, "/")
	if len(parts) < 2 || len(parts) > 3 || parts[0] == "" || parts[1] == "" {
		return "", "", "", false
	}
	if len(parts) == 2 {
		return parts[0], parts[1], "", true
	}
	return parts[0], parts[1], parts[2], true
}

func (h *RecipeHandler) serveProbeDetail(w http.ResponseWriter, r *http.Request, decisionID, variantID string) {
	if r.Method != http.MethodGet {
		writeRecipeMethodNotAllowed(w, http.MethodGet)
		return
	}
	probe, digest, err := h.service.Probe(decisionID, variantID)
	if err != nil {
		writeRecipeError(w, recipeErrorStatus(err), err)
		return
	}
	w.Header().Set("ETag", `"`+digest+`"`)
	writeRecipeJSON(w, http.StatusOK, probe)
}

func (h *RecipeHandler) serveProbeMutation(
	w http.ResponseWriter,
	r *http.Request,
	decisionID, variantID, action string,
) {
	if r.Method != http.MethodPost {
		writeRecipeMethodNotAllowed(w, http.MethodPost)
		return
	}
	if action != "run-plan" && action != "validate" {
		writeRecipeError(w, http.StatusNotFound, recipe.ErrProbeNotFound)
		return
	}
	expectedDigest, ok := requiredRecipeDigest(w, r)
	if !ok {
		return
	}
	switch action {
	case "run-plan":
		plan, err := h.service.RunPlan(decisionID, variantID, expectedDigest)
		if err != nil {
			writeRecipeError(w, recipeErrorStatus(err), err)
			return
		}
		writeRecipeJSON(w, http.StatusOK, plan)
	case "validate":
		result, err := h.service.Validate(r.Context(), decisionID, variantID, expectedDigest)
		if err != nil {
			if errors.Is(err, recipe.ErrUpstream) {
				writeRecipeJSON(w, http.StatusBadGateway, result)
				return
			}
			writeRecipeError(w, recipeErrorStatus(err), err)
			return
		}
		writeRecipeJSON(w, http.StatusOK, result)
	}
}

func requiredRecipeDigest(w http.ResponseWriter, r *http.Request) (string, bool) {
	raw := strings.TrimSpace(r.Header.Get("If-Match"))
	if raw == "" {
		writeRecipeError(w, http.StatusPreconditionRequired, errors.New("If-Match recipe digest is required"))
		return "", false
	}
	if len(raw) < 2 || raw[0] != '"' || raw[len(raw)-1] != '"' || strings.Contains(raw[1:len(raw)-1], "\"") {
		writeRecipeError(w, http.StatusBadRequest, errors.New("If-Match must contain one quoted recipe digest"))
		return "", false
	}
	digest := raw[1 : len(raw)-1]
	if !recipeDigestPattern.MatchString(digest) {
		writeRecipeError(w, http.StatusBadRequest, errors.New("If-Match must contain a sha256 recipe digest"))
		return "", false
	}
	return digest, true
}

func optionalPositiveInt(r *http.Request, name string) (int, error) {
	raw := strings.TrimSpace(r.URL.Query().Get(name))
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 1 {
		return 0, errors.New(name + " must be a positive integer")
	}
	return value, nil
}

func recipeErrorStatus(err error) int {
	switch {
	case errors.Is(err, recipe.ErrBadRequest):
		return http.StatusBadRequest
	case errors.Is(err, recipe.ErrUnmanaged), errors.Is(err, recipe.ErrProbeNotFound):
		return http.StatusNotFound
	case errors.Is(err, recipe.ErrInvalid):
		return http.StatusUnprocessableEntity
	case errors.Is(err, recipe.ErrStale):
		return http.StatusPreconditionFailed
	case errors.Is(err, recipe.ErrActivationState):
		return http.StatusConflict
	case errors.Is(err, recipe.ErrUpstream):
		return http.StatusBadGateway
	default:
		return http.StatusInternalServerError
	}
}

func writeRecipeMethodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeRecipeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func writeRecipeError(w http.ResponseWriter, status int, err error) {
	message := err.Error()
	if status >= http.StatusInternalServerError && !errors.Is(err, recipe.ErrUpstream) {
		log.Printf("Recipe API error: %v", err)
		message = "recipe service unavailable"
	}
	writeRecipeJSON(w, status, map[string]string{"error": message})
}

func writeRecipeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("Recipe API response encoding failed: %v", err)
	}
}
