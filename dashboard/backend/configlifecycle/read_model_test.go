package configlifecycle

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const persistedActiveConfig = `
version: v0.1
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
setup:
  mode: true
  state: bootstrap
providers:
  default_model: persisted-model
  models:
    - name: persisted-model
      endpoints:
        - name: persisted-endpoint
          endpoint: host.docker.internal:8000
          protocol: http
          weight: 1
decisions:
  - name: persisted-decision
    description: Persisted decision
    priority: 100
    modelRefs:
      - model: persisted-model
`

func TestConfigJSONPrefersActiveRevision(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	saveActiveRevisionForReadModel(t, stores, []byte(persistedActiveConfig))

	config, err := service.ConfigJSON()
	if err != nil {
		t.Fatalf("ConfigJSON() error = %v", err)
	}

	configMap, ok := ToStringKeyMap(config)
	if !ok {
		t.Fatalf("expected config map, got %#v", config)
	}

	providers, ok := ToStringKeyMap(configMap["providers"])
	if !ok {
		t.Fatalf("expected providers map, got %#v", configMap["providers"])
	}
	if providers["default_model"] != "persisted-model" {
		t.Fatalf("expected persisted default model, got %#v", providers["default_model"])
	}
	if _, hasLegacyField := configMap["bert_model"]; hasLegacyField {
		t.Fatal("expected persisted active revision to override file-backed config")
	}
}

func TestConfigYAMLPrefersActiveRevision(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	saveActiveRevisionForReadModel(t, stores, []byte(persistedActiveConfig))

	yamlData, err := service.ConfigYAML()
	if err != nil {
		t.Fatalf("ConfigYAML() error = %v", err)
	}
	if string(yamlData) != persistedActiveConfig {
		t.Fatalf("expected persisted YAML to be returned, got:\n%s", string(yamlData))
	}
}

func TestSetupStatePrefersActiveRevision(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	saveActiveRevisionForReadModel(t, stores, []byte(persistedActiveConfig))

	state, err := service.SetupState()
	if err != nil {
		t.Fatalf("SetupState() error = %v", err)
	}
	if !state.SetupMode {
		t.Fatal("expected setup mode from persisted active revision")
	}
	if state.ListenerPort != 8899 {
		t.Fatalf("expected listener port 8899, got %d", state.ListenerPort)
	}
	if state.Models != 1 || state.Decisions != 1 {
		t.Fatalf("expected persisted counts models=1 decisions=1, got models=%d decisions=%d", state.Models, state.Decisions)
	}
	if !state.CanActivate {
		t.Fatal("expected persisted setup state to be activatable")
	}
}

func saveActiveRevisionForReadModel(t *testing.T, stores *console.Stores, yamlData []byte) {
	t.Helper()

	documentJSON, err := marshalRevisionDocument(yamlData)
	if err != nil {
		t.Fatalf("marshalRevisionDocument() error = %v", err)
	}

	now := time.Now().UTC()
	err = stores.Revisions.SaveConfigRevision(context.Background(), &console.ConfigRevision{
		Status:            console.ConfigRevisionStatusActive,
		Source:            "test_active_revision",
		Summary:           "Persisted active config for compatibility read tests",
		DocumentJSON:      documentJSON,
		RuntimeConfigYAML: string(yamlData),
		CreatedBy:         "test",
		ActivatedAt:       &now,
	})
	if err != nil {
		t.Fatalf("SaveConfigRevision() error = %v", err)
	}
}
