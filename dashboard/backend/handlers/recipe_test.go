package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func TestRecipeHandlerUnmanagedDescriptorAndProbeBoundary(t *testing.T) {
	handler := NewRecipeHandler(recipe.NewService(recipe.Options{Directory: t.TempDir()}))

	descriptorResponse := httptest.NewRecorder()
	handler.Descriptor(descriptorResponse, httptest.NewRequest(http.MethodGet, "/api/recipe", nil))
	if descriptorResponse.Code != http.StatusOK {
		t.Fatalf("descriptor status = %d, body=%s", descriptorResponse.Code, descriptorResponse.Body.String())
	}
	if cacheControl := descriptorResponse.Header().Get("Cache-Control"); cacheControl != "no-cache, no-store, must-revalidate" {
		t.Fatalf("Cache-Control = %q", cacheControl)
	}
	var descriptor recipe.Descriptor
	if err := json.NewDecoder(descriptorResponse.Body).Decode(&descriptor); err != nil {
		t.Fatalf("decode descriptor: %v", err)
	}
	if descriptor.Managed || descriptor.SourceHealth.Status != "unmanaged" {
		t.Fatalf("descriptor = %#v", descriptor)
	}

	probeResponse := httptest.NewRecorder()
	handler.Probes(probeResponse, httptest.NewRequest(http.MethodGet, "/api/recipe/probes", nil))
	if probeResponse.Code != http.StatusNotFound {
		t.Fatalf("probe list status = %d, body=%s", probeResponse.Code, probeResponse.Body.String())
	}

	methodResponse := httptest.NewRecorder()
	handler.Descriptor(methodResponse, httptest.NewRequest(http.MethodPost, "/api/recipe", nil))
	if methodResponse.Code != http.StatusMethodNotAllowed || methodResponse.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("method response = %d Allow=%q", methodResponse.Code, methodResponse.Header().Get("Allow"))
	}

	preconditionResponse := httptest.NewRecorder()
	handler.ProbeAction(preconditionResponse, httptest.NewRequest(http.MethodPost, "/api/recipe/probes/lane/variant/run-plan", nil))
	if preconditionResponse.Code != http.StatusPreconditionRequired {
		t.Fatalf("missing If-Match status = %d, want %d", preconditionResponse.Code, http.StatusPreconditionRequired)
	}
}

func TestRecipeHandlerETagRoundTripsIntoActionPrecondition(t *testing.T) {
	service := recipe.NewService(recipe.Options{
		Directory: filepath.Join("..", "..", "..", "config", "recipes", "accuracy"),
	})
	probes, err := service.ListProbes(recipe.ListOptions{Page: 1, PageSize: 1})
	if err != nil || len(probes.Items) != 1 {
		t.Fatalf("ListProbes() = %#v, %v", probes, err)
	}
	probe := probes.Items[0]
	handler := NewRecipeHandler(service)
	basePath := "/api/recipe/probes/" + url.PathEscape(probe.DecisionID) + "/" + url.PathEscape(probe.VariantID)

	detailResponse := httptest.NewRecorder()
	handler.ProbeAction(detailResponse, httptest.NewRequest(http.MethodGet, basePath, nil))
	if detailResponse.Code != http.StatusOK {
		t.Fatalf("detail status = %d, body=%s", detailResponse.Code, detailResponse.Body.String())
	}
	etag := detailResponse.Header().Get("ETag")
	if strings.Trim(etag, "\"") != probes.RecipeDigest {
		t.Fatalf("ETag = %q, recipe digest = %q", etag, probes.RecipeDigest)
	}

	actionRequest := httptest.NewRequest(http.MethodPost, basePath+"/run-plan", nil)
	actionRequest.Header.Set("If-Match", etag)
	actionResponse := httptest.NewRecorder()
	handler.ProbeAction(actionResponse, actionRequest)
	if actionResponse.Code != http.StatusOK {
		t.Fatalf("run-plan status = %d, body=%s", actionResponse.Code, actionResponse.Body.String())
	}
}

func TestRecipeHandlerRejectsEmptyActionPrecondition(t *testing.T) {
	handler := NewRecipeHandler(recipe.NewService(recipe.Options{Directory: t.TempDir()}))
	for _, header := range []string{`""`, `"   "`, `"sha256:not-a-digest"`} {
		t.Run(header, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/api/recipe/probes/lane/variant/run-plan", nil)
			request.Header.Set("If-Match", header)
			response := httptest.NewRecorder()
			handler.ProbeAction(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusBadRequest)
			}
		})
	}
}
