package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ============================================================
// deepMerge unit tests
// ============================================================

func TestDeepMerge_Basic(t *testing.T) {
	tests := []struct {
		name     string
		dst      map[string]interface{}
		src      map[string]interface{}
		expected map[string]interface{}
	}{
		{
			name:     "empty src into empty dst",
			dst:      map[string]interface{}{},
			src:      map[string]interface{}{},
			expected: map[string]interface{}{},
		},
		{
			name: "add new key",
			dst:  map[string]interface{}{"a": "1"},
			src:  map[string]interface{}{"b": "2"},
			expected: map[string]interface{}{
				"a": "1",
				"b": "2",
			},
		},
		{
			name: "overwrite scalar",
			dst:  map[string]interface{}{"a": "old"},
			src:  map[string]interface{}{"a": "new"},
			expected: map[string]interface{}{
				"a": "new",
			},
		},
		{
			name: "overwrite array (no array merge)",
			dst: map[string]interface{}{
				"items": []interface{}{"a", "b"},
			},
			src: map[string]interface{}{
				"items": []interface{}{"c"},
			},
			expected: map[string]interface{}{
				"items": []interface{}{"c"},
			},
		},
		{
			name: "preserve dst keys not in src",
			dst: map[string]interface{}{
				"keep_me":   "yes",
				"overwrite": "old",
				"also_keep": 42,
			},
			src: map[string]interface{}{
				"overwrite": "new",
				"added":     "fresh",
			},
			expected: map[string]interface{}{
				"keep_me":   "yes",
				"overwrite": "new",
				"also_keep": 42,
				"added":     "fresh",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := deepMerge(tt.dst, tt.src)
			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d keys, got %d. Result: %v", len(tt.expected), len(result), result)
				return
			}
			for k, v := range tt.expected {
				if fmt.Sprintf("%v", result[k]) != fmt.Sprintf("%v", v) {
					t.Errorf("Key %q: expected %v, got %v", k, v, result[k])
				}
			}
		})
	}
}

func TestDeepMerge_NestedMaps(t *testing.T) {
	t.Run("recursive merge of nested maps", func(t *testing.T) {
		dst := map[string]interface{}{
			"classifier": map[string]interface{}{
				"category_model": map[string]interface{}{
					"model_id":  "original-model",
					"threshold": 0.6,
					"use_cpu":   true,
				},
				"pii_model": map[string]interface{}{
					"model_id": "pii-model",
				},
			},
			"default_model": "gpt-4",
		}
		src := map[string]interface{}{
			"classifier": map[string]interface{}{
				"category_model": map[string]interface{}{
					"threshold": 0.8, // update
				},
			},
		}

		result := deepMerge(dst, src)

		classifier, ok := result["classifier"].(map[string]interface{})
		if !ok {
			t.Fatal("classifier should be a map")
		}

		// pii_model should be preserved
		if _, ok := classifier["pii_model"]; !ok {
			t.Error("pii_model should be preserved from dst")
		}

		catModel, ok := classifier["category_model"].(map[string]interface{})
		if !ok {
			t.Fatal("category_model should be a map")
		}

		// threshold should be updated
		if catModel["threshold"] != 0.8 {
			t.Errorf("threshold should be 0.8, got %v", catModel["threshold"])
		}

		// model_id and use_cpu should be preserved
		if catModel["model_id"] != "original-model" {
			t.Errorf("model_id should be preserved, got %v", catModel["model_id"])
		}
		if catModel["use_cpu"] != true {
			t.Errorf("use_cpu should be preserved, got %v", catModel["use_cpu"])
		}

		// default_model should be preserved
		if result["default_model"] != "gpt-4" {
			t.Errorf("default_model should be preserved, got %v", result["default_model"])
		}
	})

	t.Run("src map overwrites dst scalar", func(t *testing.T) {
		dst := map[string]interface{}{
			"field": "scalar_value",
		}
		src := map[string]interface{}{
			"field": map[string]interface{}{"nested": true},
		}

		result := deepMerge(dst, src)
		if _, ok := result["field"].(map[string]interface{}); !ok {
			t.Error("src map should overwrite dst scalar")
		}
	})

	t.Run("src scalar overwrites dst map", func(t *testing.T) {
		dst := map[string]interface{}{
			"field": map[string]interface{}{"nested": true},
		}
		src := map[string]interface{}{
			"field": "scalar_value",
		}

		result := deepMerge(dst, src)
		if result["field"] != "scalar_value" {
			t.Error("src scalar should overwrite dst map")
		}
	})
}

func TestDeepMerge_YAMLv2MapTypes(t *testing.T) {
	t.Run("handles map[interface{}]interface{} from yaml.v2", func(t *testing.T) {
		// yaml.v2 produces map[interface{}]interface{} instead of map[string]interface{}
		dst := map[string]interface{}{
			"config": map[interface{}]interface{}{
				"keep": "old_value",
				"both": "dst_value",
			},
		}
		src := map[string]interface{}{
			"config": map[interface{}]interface{}{
				"both": "src_value",
				"new":  "added",
			},
		}

		result := deepMerge(dst, src)

		configRaw, exists := result["config"]
		if !exists {
			t.Fatal("config key should exist")
		}

		config, ok := configRaw.(map[string]interface{})
		if !ok {
			t.Fatalf("config should be converted to map[string]interface{}, got %T", configRaw)
		}

		if config["keep"] != "old_value" {
			t.Errorf("keep should be preserved, got %v", config["keep"])
		}
		if config["both"] != "src_value" {
			t.Errorf("both should be overwritten to src_value, got %v", config["both"])
		}
		if config["new"] != "added" {
			t.Errorf("new should be added, got %v", config["new"])
		}
	})
}

func TestToStringKeyMap(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected bool
	}{
		{
			name:     "map[string]interface{} returns true",
			input:    map[string]interface{}{"key": "val"},
			expected: true,
		},
		{
			name:     "map[interface{}]interface{} returns true",
			input:    map[interface{}]interface{}{"key": "val"},
			expected: true,
		},
		{
			name:     "string returns false",
			input:    "not a map",
			expected: false,
		},
		{
			name:     "slice returns false",
			input:    []interface{}{"a", "b"},
			expected: false,
		},
		{
			name:     "nil returns false",
			input:    nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := toStringKeyMap(tt.input)
			if ok != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, ok)
			}
		})
	}
}

// ============================================================
// DeployHandler tests
// ============================================================

func TestDeployHandler_MethodValidation(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	methods := []string{http.MethodGet, http.MethodPut, http.MethodDelete, http.MethodPatch}
	for _, method := range methods {
		t.Run("reject "+method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/router/config/deploy", nil)
			w := httptest.NewRecorder()

			handler := DeployHandler(configPath, false, tempDir)
			handler(w, req)

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("Expected 405, got %d", w.Code)
			}
		})
	}
}

func TestDeployHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body := DeployRequest{YAML: "test: value"}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, true, tempDir)
	handler(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("Expected 403, got %d. Body: %s", w.Code, w.Body.String())
	}

	if !contains(w.Body.String(), "readonly_mode") {
		t.Errorf("Expected readonly_mode error, got: %s", w.Body.String())
	}
}

func TestDeployHandler_EmptyYAML(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body := DeployRequest{YAML: "   "}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestDeployHandler_InvalidJSON(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestDeployHandler_InvalidYAMLSyntax(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body := DeployRequest{YAML: "invalid: yaml: [unclosed"}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d. Body: %s", w.Code, w.Body.String())
	}

	if !contains(w.Body.String(), "yaml_parse_error") {
		t.Errorf("Expected yaml_parse_error, got: %s", w.Body.String())
	}
}

func TestDeployHandler_SuccessfulDeploy(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Read original config to verify preservation after merge
	originalData, _ := os.ReadFile(configPath)

	// Deploy a minimal valid config (just adds a new key)
	deployYAML := `default_model: deployed-model
`
	body := DeployRequest{
		YAML: deployYAML,
		DSL:  "route default { model deployed-model }",
	}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
	}

	// Verify response structure
	var resp DeployResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if resp.Status != "success" {
		t.Errorf("Expected status 'success', got '%s'", resp.Status)
	}
	if resp.Version == "" {
		t.Error("Version should not be empty")
	}

	// Verify config was written
	newData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read config after deploy: %v", err)
	}
	if len(newData) == 0 {
		t.Error("Config file is empty after deploy")
	}

	// Verify backup was created
	backupDir := filepath.Join(tempDir, ".vllm-sr", "config-backups")
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		t.Fatalf("Failed to read backup dir: %v", err)
	}
	if len(entries) == 0 {
		t.Error("No backup was created")
	}

	// Verify backup content matches original
	backupData, _ := os.ReadFile(filepath.Join(backupDir, entries[0].Name()))
	if string(backupData) != string(originalData) {
		t.Error("Backup content does not match original config")
	}

	// Verify DSL source was archived
	dslFile := filepath.Join(tempDir, ".vllm-sr", "config.dsl")
	dslData, err := os.ReadFile(dslFile)
	if err != nil {
		t.Fatalf("DSL source was not archived: %v", err)
	}
	if string(dslData) != body.DSL {
		t.Errorf("Archived DSL content mismatch: %s", dslData)
	}
}

func TestDeployHandler_DeepMergePreservesExistingFields(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Deploy config with only keyword_rules (simulating DSL output that only has signals)
	deployYAML := `keyword_rules:
  - name: test-signal
    keywords:
      - hello
      - world
`
	body := DeployRequest{YAML: deployYAML}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
	}

	// Read deployed config and verify deep merge preserved existing fields
	data, _ := os.ReadFile(configPath)
	configStr := string(data)

	// These fields from the original config should be preserved
	if !contains(configStr, "default_model") {
		t.Error("default_model should be preserved after deploy")
	}
	if !contains(configStr, "vllm_endpoints") {
		t.Error("vllm_endpoints should be preserved after deploy")
	}

	// The deployed keyword_rules should be present
	if !contains(configStr, "keyword_rules") {
		t.Error("keyword_rules from deploy should be present")
	}
}

func TestDeployHandler_NoDSLSource(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Deploy without DSL source
	body := DeployRequest{YAML: "default_model: new-model\n"}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := DeployHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
	}

	// DSL file should not exist
	dslFile := filepath.Join(tempDir, ".vllm-sr", "config.dsl")
	if _, err := os.Stat(dslFile); err == nil {
		t.Error("DSL file should not be created when DSL source is empty")
	}
}

// ============================================================
// RollbackHandler tests
// ============================================================

func TestRollbackHandler_MethodValidation(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	methods := []string{http.MethodGet, http.MethodPut, http.MethodDelete}
	for _, method := range methods {
		t.Run("reject "+method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/router/config/rollback", nil)
			w := httptest.NewRecorder()

			handler := RollbackHandler(configPath, false, tempDir)
			handler(w, req)

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("Expected 405, got %d", w.Code)
			}
		})
	}
}

func TestRollbackHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body, _ := json.Marshal(map[string]string{"version": "20260101-120000"})
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := RollbackHandler(configPath, true, tempDir)
	handler(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("Expected 403, got %d", w.Code)
	}
}

func TestRollbackHandler_MissingVersion(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body, _ := json.Marshal(map[string]string{"version": ""})
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := RollbackHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestRollbackHandler_VersionNotFound(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	body, _ := json.Marshal(map[string]string{"version": "99990101-000000"})
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := RollbackHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d. Body: %s", w.Code, w.Body.String())
	}

	if !contains(w.Body.String(), "version_not_found") {
		t.Errorf("Expected version_not_found error, got: %s", w.Body.String())
	}
}

func TestRollbackHandler_SuccessfulRollback(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Read original config
	originalData, _ := os.ReadFile(configPath)

	// Create a backup to rollback to
	backupDir := filepath.Join(tempDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("Failed to create backup dir: %v", err)
	}

	backupVersion := "20260101-120000"
	backupContent := string(originalData)
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", backupVersion))
	if err := os.WriteFile(backupFile, []byte(backupContent), 0o644); err != nil {
		t.Fatalf("Failed to create backup file: %v", err)
	}

	// Modify current config so we can verify rollback restores it
	modifiedConfig := `default_model: modified-model
`
	if err := os.WriteFile(configPath, []byte(modifiedConfig), 0o644); err != nil {
		t.Fatalf("Failed to modify config: %v", err)
	}

	// Rollback
	body, _ := json.Marshal(map[string]string{"version": backupVersion})
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := RollbackHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
	}

	// Verify response
	var resp DeployResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if resp.Status != "success" {
		t.Errorf("Expected status 'success', got '%s'", resp.Status)
	}
	if resp.Version != backupVersion {
		t.Errorf("Expected version '%s', got '%s'", backupVersion, resp.Version)
	}

	// Verify config was restored to backup content
	restoredData, _ := os.ReadFile(configPath)
	if string(restoredData) != backupContent {
		t.Errorf("Config was not restored to backup content.\nExpected: %s\nGot: %s", backupContent, restoredData)
	}

	// Verify a pre-rollback backup was created (the modified config)
	entries, _ := os.ReadDir(backupDir)
	if len(entries) < 2 {
		t.Error("Pre-rollback backup should have been created")
	}
}

// ============================================================
// ConfigVersionsHandler tests
// ============================================================

func TestConfigVersionsHandler_MethodValidation(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	methods := []string{http.MethodPost, http.MethodPut, http.MethodDelete}
	for _, method := range methods {
		t.Run("reject "+method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/api/router/config/versions", nil)
			w := httptest.NewRecorder()

			handler := ConfigVersionsHandler(configPath)
			handler(w, req)

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("Expected 405, got %d", w.Code)
			}
		})
	}
}

func TestConfigVersionsHandler_EmptyBackups(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil)
	w := httptest.NewRecorder()

	handler := ConfigVersionsHandler(configPath)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}

	var versions []ConfigVersion
	if err := json.NewDecoder(w.Body).Decode(&versions); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if len(versions) != 0 {
		t.Errorf("Expected 0 versions, got %d", len(versions))
	}
}

func TestConfigVersionsHandler_ListsVersions(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Create backup directory with some versions
	backupDir := filepath.Join(tempDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("Failed to create backup dir: %v", err)
	}

	versionNames := []string{"20260101-100000", "20260101-120000", "20260101-080000"}
	for _, v := range versionNames {
		f := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", v))
		if err := os.WriteFile(f, []byte("test: data"), 0o644); err != nil {
			t.Fatalf("Failed to create backup: %v", err)
		}
	}

	// Also create a non-backup file that should be ignored
	if err := os.WriteFile(filepath.Join(backupDir, "notes.txt"), []byte("ignore me"), 0o644); err != nil {
		t.Fatalf("Failed to create non-backup file: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil)
	w := httptest.NewRecorder()

	handler := ConfigVersionsHandler(configPath)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}

	var versions []ConfigVersion
	if err := json.NewDecoder(w.Body).Decode(&versions); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if len(versions) != 3 {
		t.Fatalf("Expected 3 versions, got %d", len(versions))
	}

	// Should be sorted descending (newest first)
	if versions[0].Version != "20260101-120000" {
		t.Errorf("First version should be newest, got %s", versions[0].Version)
	}
	if versions[2].Version != "20260101-080000" {
		t.Errorf("Last version should be oldest, got %s", versions[2].Version)
	}

	// Verify timestamp parsing
	if versions[0].Timestamp != "2026-01-01 12:00:00" {
		t.Errorf("Timestamp not properly formatted, got %s", versions[0].Timestamp)
	}
}

// ============================================================
// cleanupBackups tests
// ============================================================

func TestCleanupBackups(t *testing.T) {
	t.Run("no cleanup needed when under limit", func(t *testing.T) {
		tempDir := t.TempDir()
		for i := 0; i < 5; i++ {
			f := filepath.Join(tempDir, fmt.Sprintf("config.2026010%d-120000.yaml", i))
			os.WriteFile(f, []byte("test"), 0o644)
		}

		cleanupBackups(tempDir)

		entries, _ := os.ReadDir(tempDir)
		if len(entries) != 5 {
			t.Errorf("Expected 5 files, got %d", len(entries))
		}
	})

	t.Run("removes oldest when over limit", func(t *testing.T) {
		tempDir := t.TempDir()
		// Create maxBackups + 3 files
		for i := 0; i < maxBackups+3; i++ {
			ts := time.Date(2026, 1, 1, 0, 0, i, 0, time.UTC).Format("20060102-150405")
			f := filepath.Join(tempDir, fmt.Sprintf("config.%s.yaml", ts))
			os.WriteFile(f, []byte("test"), 0o644)
		}

		cleanupBackups(tempDir)

		entries, _ := os.ReadDir(tempDir)
		count := 0
		for _, e := range entries {
			if !e.IsDir() {
				count++
			}
		}
		if count != maxBackups {
			t.Errorf("Expected %d files after cleanup, got %d", maxBackups, count)
		}
	})

	t.Run("non-config files are not removed", func(t *testing.T) {
		tempDir := t.TempDir()
		// Create over limit of config backups
		for i := 0; i < maxBackups+2; i++ {
			ts := time.Date(2026, 1, 1, 0, 0, i, 0, time.UTC).Format("20060102-150405")
			f := filepath.Join(tempDir, fmt.Sprintf("config.%s.yaml", ts))
			os.WriteFile(f, []byte("test"), 0o644)
		}
		// Also create non-config files
		os.WriteFile(filepath.Join(tempDir, "notes.txt"), []byte("keep"), 0o644)
		os.WriteFile(filepath.Join(tempDir, "config.dsl"), []byte("keep"), 0o644)

		cleanupBackups(tempDir)

		// Non-config files should still exist
		if _, err := os.Stat(filepath.Join(tempDir, "notes.txt")); os.IsNotExist(err) {
			t.Error("notes.txt should not be removed")
		}
		if _, err := os.Stat(filepath.Join(tempDir, "config.dsl")); os.IsNotExist(err) {
			t.Error("config.dsl should not be removed")
		}
	})

	t.Run("handles non-existent directory", func(t *testing.T) {
		// Should not panic
		cleanupBackups("/non/existent/path")
	})
}

// ============================================================
// Deploy + Rollback integration test
// ============================================================

func TestDeployAndRollback_Integration(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Read original
	originalData, _ := os.ReadFile(configPath)

	// Deploy
	deployYAML := "default_model: integrated-deploy\n"
	body := DeployRequest{YAML: deployYAML}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	DeployHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Deploy failed: %d - %s", w.Code, w.Body.String())
	}

	var deployResp DeployResponse
	json.NewDecoder(w.Body).Decode(&deployResp)

	// Config should be changed
	afterDeploy, _ := os.ReadFile(configPath)
	if string(afterDeploy) == string(originalData) {
		t.Error("Config should have changed after deploy")
	}

	// Verify the deployed config contains both new and old values (deep merge)
	afterDeployStr := string(afterDeploy)
	if !contains(afterDeployStr, "integrated-deploy") {
		t.Error("Deployed default_model not found in config")
	}

	// List versions — should have at least 1 backup
	req = httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil)
	w = httptest.NewRecorder()
	ConfigVersionsHandler(configPath)(w, req)

	var versions []ConfigVersion
	json.NewDecoder(w.Body).Decode(&versions)
	if len(versions) == 0 {
		t.Fatal("Expected at least 1 backup version")
	}

	// Rollback to the backup (original config)
	rollbackBody, _ := json.Marshal(map[string]string{"version": versions[0].Version})
	req = httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(rollbackBody))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()

	RollbackHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Rollback failed: %d - %s", w.Code, w.Body.String())
	}

	// Config should be restored to original
	afterRollback, _ := os.ReadFile(configPath)
	if string(afterRollback) != string(originalData) {
		t.Errorf("Config not restored after rollback.\nExpected:\n%s\nGot:\n%s", originalData, afterRollback)
	}
}

// ============================================================
// UpdateConfigHandler deep merge upgrade test
// ============================================================

func TestUpdateConfigHandler_DeepMergeNested(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Update only a deeply nested field — classifier.category_model.threshold
	// The rest of classifier should be preserved
	updateBody := map[string]interface{}{
		"classifier": map[string]interface{}{
			"category_model": map[string]interface{}{
				"threshold": 0.9,
			},
		},
	}

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := UpdateConfigHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
	}

	// Read back and verify deep merge
	data, _ := os.ReadFile(configPath)
	configStr := string(data)

	// The pii_model should be preserved (not in the update, but was in original)
	if !contains(configStr, "pii_model") {
		t.Error("pii_model should be preserved by deep merge")
	}

	// The category_model.model_id should be preserved
	if !contains(configStr, "all-MiniLM-L12-v2") || !contains(configStr, "bert_model") {
		t.Error("bert_model should be preserved")
	}

	// default_model should be preserved
	if !contains(configStr, "test-model") {
		t.Error("default_model should be preserved")
	}
}
