package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestSaveConfigRevisionDraftHandlerWithService(t *testing.T) {
	service := newRevisionMutationTestService(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/draft", bytes.NewBufferString(`{"summary":"draft test","runtimeConfigYAML":`+quoteJSONString(configlifecycleTestConfig)+`}`))
	w := httptest.NewRecorder()

	SaveConfigRevisionDraftHandlerWithService(service, false)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response SaveConfigRevisionDraftResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if response.ID == "" || response.Status != string(console.ConfigRevisionStatusDraft) {
		t.Fatalf("expected saved draft response, got %#v", response)
	}
}

func TestValidateConfigRevisionHandlerWithService(t *testing.T) {
	service := newRevisionMutationTestService(t)
	draft := saveHandlerDraft(t, service)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/validate", bytes.NewBufferString(`{"id":"`+draft.ID+`"}`))
	w := httptest.NewRecorder()

	ValidateConfigRevisionHandlerWithService(service, false)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response ValidateConfigRevisionResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if response.ID != draft.ID || response.Status != string(console.ConfigRevisionStatusValidated) {
		t.Fatalf("expected validated response for %s, got %#v", draft.ID, response)
	}
}

func TestSaveConfigRevisionDraftHandlerWithService_ReadonlyMode(t *testing.T) {
	service := newRevisionMutationTestService(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/draft", bytes.NewBufferString(`{"summary":"draft test","runtimeConfigYAML":`+quoteJSONString(configlifecycleTestConfig)+`}`))
	w := httptest.NewRecorder()

	SaveConfigRevisionDraftHandlerWithService(service, true)(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d", w.Code)
	}
}

func newRevisionMutationTestService(t *testing.T) *configlifecycle.Service {
	t.Helper()

	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(configlifecycleTestConfig), 0o644); err != nil {
		t.Fatalf("WriteFile(configPath) error = %v", err)
	}

	stores := emptyRevisionTestStores(t)
	saveHandlerRevisionFixture(
		t,
		stores,
		console.ConfigRevisionStatusActive,
		"revision-active",
		configlifecycleTestConfig,
		time.Now().UTC().Add(-1*time.Minute),
	)
	return configlifecycle.NewWithStores(configPath, dir, stores)
}

func saveHandlerDraft(t *testing.T, service *configlifecycle.Service) *configlifecycle.RevisionSaveResult {
	t.Helper()

	result, err := service.SaveDraftRevision(configlifecycle.RevisionDraftInput{
		Summary:           "handler draft",
		RuntimeConfigYAML: configlifecycleTestConfig,
	})
	if err != nil {
		t.Fatalf("SaveDraftRevision() error = %v", err)
	}
	return result
}

func quoteJSONString(value string) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}
