//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestConfigPreconditionRequiresExactCurrentETag(t *testing.T) {
	current := []byte("version: v0.3\n")

	missingRequest := httptest.NewRequest(http.MethodPut, apiConfigPath, nil)
	missingResponse := httptest.NewRecorder()
	if checkConfigPrecondition(missingResponse, missingRequest, current) {
		t.Fatal("missing If-Match unexpectedly passed")
	}
	if missingResponse.Code != configPreconditionRequiredStatus {
		t.Fatalf("missing If-Match status = %d, want %d", missingResponse.Code, configPreconditionRequiredStatus)
	}

	staleRequest := httptest.NewRequest(http.MethodPut, apiConfigPath, nil)
	staleRequest.Header.Set("If-Match", `"stale"`)
	staleResponse := httptest.NewRecorder()
	if checkConfigPrecondition(staleResponse, staleRequest, current) {
		t.Fatal("stale If-Match unexpectedly passed")
	}
	if staleResponse.Code != http.StatusPreconditionFailed {
		t.Fatalf("stale If-Match status = %d, want %d", staleResponse.Code, http.StatusPreconditionFailed)
	}
	if got := staleResponse.Header().Get("ETag"); got != configDocumentETag(current) {
		t.Fatalf("stale response ETag = %q, want %q", got, configDocumentETag(current))
	}

	currentRequest := httptest.NewRequest(http.MethodPut, apiConfigPath, nil)
	currentRequest.Header.Set("If-Match", configDocumentETag(current))
	if !checkConfigPrecondition(httptest.NewRecorder(), currentRequest, current) {
		t.Fatal("current If-Match did not pass")
	}
}
