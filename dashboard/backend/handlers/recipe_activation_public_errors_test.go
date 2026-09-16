package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func TestUnsafeManagedStoragePreviewReturnsStablePublicRemediation(t *testing.T) {
	internal := errors.New("managed redis storage is unsafe: actual published port 6379/tcp uses non-loopback host address 0.0.0.0; private-container-marker")
	mapped := packageTopologyInventoryError("generic topology failure", internal)
	packageErr, ok := recipe.AsPackageError(mapped)
	if !ok || packageErr.Code != recipe.ErrorManagedStorageIsolation || packageErr.Status != http.StatusConflict {
		t.Fatalf("mapped error = %#v, ok=%v", packageErr, ok)
	}

	digest := "sha256:" + strings.Repeat("a", 64)
	activator := &fakeRecipePackageActivator{err: mapped}
	handler := NewRecipeHandler(nil, WithRecipePackages(&fakeRecipePackageStore{}, activator, false))
	response := httptest.NewRecorder()
	handler.PreviewPackageActivation(response, httptest.NewRequest(
		http.MethodPost,
		"/api/recipe/activate/preview",
		strings.NewReader(`{"recipe_digest":"`+digest+`"}`),
	))

	encoded := response.Body.String()
	var body map[string]string
	if err := json.Unmarshal([]byte(encoded), &body); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusConflict || body["error"] != recipe.ErrorManagedStorageIsolation || body["message"] != managedStorageIsolationRemediation {
		t.Fatalf("status=%d body=%#v", response.Code, body)
	}
	for _, private := range []string{"redis", "6379", "0.0.0.0", "private-container-marker"} {
		if strings.Contains(encoded, private) {
			t.Fatalf("public preview error leaked %q: %s", private, encoded)
		}
	}
	if !strings.Contains(body["message"], "loopback-only") || !strings.Contains(body["message"], "Preserve") {
		t.Fatalf("public preview remediation is incomplete: %q", body["message"])
	}
}

func TestUnsafeManagedRouterPreviewReturnsStablePublicRemediation(t *testing.T) {
	internal := errors.New("managed Router management publication is unsafe: wildcard 0.0.0.0:8080; private-container-marker")
	mapped := packageTopologyInventoryError("generic topology failure", internal)
	packageErr, ok := recipe.AsPackageError(mapped)
	if !ok || packageErr.Code != recipe.ErrorActivationIncompatible || packageErr.Status != http.StatusConflict {
		t.Fatalf("mapped error = %#v, ok=%v", packageErr, ok)
	}
	if packageErr.Safe != managedRouterIsolationRemediation {
		t.Fatalf("safe remediation = %q", packageErr.Safe)
	}
	for _, private := range []string{"8080", "0.0.0.0", "private-container-marker"} {
		if strings.Contains(packageErr.Error(), private) {
			t.Fatalf("public preview error leaked %q: %s", private, packageErr.Error())
		}
	}
}

func TestUnrelatedTopologyPreviewErrorKeepsGenericFailureContract(t *testing.T) {
	mapped := packageTopologyInventoryError("Managed runtime topology could not be inspected.", errors.New("docker daemon unavailable"))
	packageErr, ok := recipe.AsPackageError(mapped)
	if !ok || packageErr.Code != recipe.ErrorActivationFailed || packageErr.Safe != "Managed runtime topology could not be inspected." {
		t.Fatalf("mapped error = %#v, ok=%v", packageErr, ok)
	}
}
