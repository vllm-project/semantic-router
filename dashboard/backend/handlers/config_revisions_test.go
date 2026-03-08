package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestConfigRevisionsHandlerWithService(t *testing.T) {
	service := newRevisionTestService(t)
	req := httptest.NewRequest(http.MethodGet, "/api/router/config/revisions", nil)
	w := httptest.NewRecorder()

	ConfigRevisionsHandlerWithService(service)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response []ConfigRevisionSummaryResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(response) != 1 {
		t.Fatalf("expected 1 revision, got %d", len(response))
	}
	if response[0].Status != string(console.ConfigRevisionStatusActive) {
		t.Fatalf("expected active revision status, got %q", response[0].Status)
	}
	if response[0].RuntimeTarget != "workspace-files" {
		t.Fatalf("expected runtime target workspace-files, got %q", response[0].RuntimeTarget)
	}
}

func TestCurrentConfigRevisionHandlerWithService(t *testing.T) {
	service := newRevisionTestService(t)
	req := httptest.NewRequest(http.MethodGet, "/api/router/config/revisions/current", nil)
	w := httptest.NewRecorder()

	CurrentConfigRevisionHandlerWithService(service)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response ConfigRevisionDetailResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if response.Status != string(console.ConfigRevisionStatusActive) {
		t.Fatalf("expected active revision status, got %q", response.Status)
	}
	if response.Document == nil {
		t.Fatal("expected revision document payload")
	}
}

func TestActivateConfigRevisionHandlerWithService(t *testing.T) {
	service := newRevisionActivationTestService(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/activate", bytes.NewBufferString(`{"id":"revision-target"}`))
	w := httptest.NewRecorder()

	ActivateConfigRevisionHandlerWithService(service, false)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response ActivateConfigRevisionResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if response.ID != "revision-target" {
		t.Fatalf("expected activated revision-target, got %q", response.ID)
	}
	if response.Status != string(console.ConfigRevisionStatusActive) {
		t.Fatalf("expected active revision status, got %q", response.Status)
	}
	if response.Message == "" {
		t.Fatal("expected activation message")
	}
}

func TestActivateConfigRevisionHandlerWithService_ReadonlyMode(t *testing.T) {
	service := newRevisionActivationTestService(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/activate", bytes.NewBufferString(`{"id":"revision-target"}`))
	w := httptest.NewRecorder()

	ActivateConfigRevisionHandlerWithService(service, true)(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d", w.Code)
	}
}

func TestActivateConfigRevisionHandlerWithService_NotFound(t *testing.T) {
	service := newRevisionActivationTestService(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/revisions/activate", bytes.NewBufferString(`{"id":"missing-revision"}`))
	w := httptest.NewRecorder()

	ActivateConfigRevisionHandlerWithService(service, false)(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d", w.Code)
	}
}

func TestCurrentConfigRevisionHandlerWithService_NotFound(t *testing.T) {
	service := configlifecycle.NewWithStores(filepath.Join(t.TempDir(), "config.yaml"), t.TempDir(), emptyRevisionTestStores(t))
	req := httptest.NewRequest(http.MethodGet, "/api/router/config/revisions/current", nil)
	w := httptest.NewRecorder()

	CurrentConfigRevisionHandlerWithService(service)(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d", w.Code)
	}
}

func newRevisionTestService(t *testing.T) *configlifecycle.Service {
	t.Helper()

	dir := t.TempDir()
	stores := emptyRevisionTestStores(t)
	now := time.Now().UTC()
	documentJSON, err := json.Marshal(map[string]interface{}{
		"version": "v0.1",
		"providers": map[string]interface{}{
			"default_model": "persisted-model",
		},
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	revision := &console.ConfigRevision{
		Status:            console.ConfigRevisionStatusActive,
		Source:            "test_revision_api",
		Summary:           "test revision",
		DocumentJSON:      documentJSON,
		RuntimeConfigYAML: "version: v0.1\nproviders:\n  default_model: persisted-model\n",
		CreatedBy:         "test",
		ActivatedAt:       &now,
		Metadata: map[string]interface{}{
			"runtime_target": "workspace-files",
		},
	}
	if err := stores.Revisions.SaveConfigRevision(context.Background(), revision); err != nil {
		t.Fatalf("SaveConfigRevision() error = %v", err)
	}
	if err := stores.Deployments.SaveDeployEvent(context.Background(), &console.DeployEvent{
		RevisionID:    revision.ID,
		Status:        console.DeployEventStatusSucceeded,
		TriggerSource: "test",
		Message:       "test deploy event",
		RuntimeTarget: "workspace-files",
		StartedAt:     &now,
		CompletedAt:   &now,
	}); err != nil {
		t.Fatalf("SaveDeployEvent() error = %v", err)
	}

	return configlifecycle.NewWithStores(filepath.Join(dir, "config.yaml"), dir, stores)
}

func newRevisionActivationTestService(t *testing.T) *configlifecycle.Service {
	t.Helper()

	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(configlifecycleTestConfig), 0o644); err != nil {
		t.Fatalf("WriteFile(configPath) error = %v", err)
	}

	stores := emptyRevisionTestStores(t)
	now := time.Now().UTC()

	currentID := saveHandlerRevisionFixture(t, stores, console.ConfigRevisionStatusActive, "revision-current", configlifecycleTestConfig, now.Add(-1*time.Minute))
	if err := stores.Deployments.SaveDeployEvent(context.Background(), &console.DeployEvent{
		RevisionID:    currentID,
		Status:        console.DeployEventStatusSucceeded,
		TriggerSource: "test",
		Message:       "current revision deployed",
		RuntimeTarget: "workspace-files",
		StartedAt:     &now,
		CompletedAt:   &now,
	}); err != nil {
		t.Fatalf("SaveDeployEvent(current) error = %v", err)
	}

	targetConfig := configlifecycleTestConfig + "\nui_test_flag: true\n"
	saveHandlerRevisionFixture(t, stores, console.ConfigRevisionStatusValidated, "revision-target", targetConfig, now.Add(-2*time.Minute))

	return configlifecycle.NewWithStores(configPath, dir, stores)
}

func emptyRevisionTestStores(t *testing.T) *console.Stores {
	t.Helper()

	store, err := console.NewSQLiteStore(filepath.Join(t.TempDir(), "console.db"))
	if err != nil {
		t.Fatalf("NewSQLiteStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	return console.NewStores(store)
}

func saveHandlerRevisionFixture(
	t *testing.T,
	stores *console.Stores,
	status console.ConfigRevisionStatus,
	id string,
	yamlConfig string,
	createdAt time.Time,
) string {
	t.Helper()

	var document interface{}
	if err := yaml.Unmarshal([]byte(yamlConfig), &document); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}
	documentJSON, err := json.Marshal(document)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	revision := &console.ConfigRevision{
		ID:                id,
		Status:            status,
		Source:            "test_revision_api",
		Summary:           "test revision",
		DocumentJSON:      documentJSON,
		RuntimeConfigYAML: yamlConfig,
		CreatedBy:         "test",
		Metadata: map[string]interface{}{
			"runtime_target": "workspace-files",
		},
		CreatedAt: createdAt,
		UpdatedAt: createdAt,
	}
	if status == console.ConfigRevisionStatusActive {
		revision.ActivatedAt = &createdAt
	}
	if err := stores.Revisions.SaveConfigRevision(context.Background(), revision); err != nil {
		t.Fatalf("SaveConfigRevision() error = %v", err)
	}
	return revision.ID
}

const configlifecycleTestConfig = `
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

classifier:
  category_model:
    model_id: models/lora_intent_classifier_bert-base-uncased_model
    threshold: 0.6
    use_cpu: true
    category_mapping_path: models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json
  pii_model:
    model_id: models/lora_pii_detector_bert-base-uncased_model
    threshold: 0.9
    use_cpu: true
    pii_mapping_path: models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json

categories:
  - name: business
    description: Business and management related queries

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

default_model: test-model

model_config:
  test-model:
    reasoning_family: qwen3
`
