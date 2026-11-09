package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const envReferenceDeployRouting = `routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: default-business
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`

const envReferenceProviderConfig = `version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      provider_model_id: test-model
      reliability:
        consecutive_5xx: 5
        base_ejection_time: ${DEPLOY_TEST_EJECTION_TIME:-30s}
      backend_refs:
        - name: endpoint1
          provider: vllm
          endpoint: ${DEPLOY_TEST_BACKEND_HOST:-127.0.0.1}:8000
          protocol: http
` + envReferenceDeployRouting

const envReferenceRAGRouting = `routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: docs-route
      priority: 5
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_docs
              api_key: ${DEPLOY_TEST_OPENAI_API_KEY}
`

const envReferenceContextRouting = `routing:
  modelCards:
    - name: test-model
  signals:
    context:
      - name: long_context
        min_tokens: ${DEPLOY_TEST_CONTEXT_MIN:-8K}
        max_tokens: ${DEPLOY_TEST_CONTEXT_MAX}
  decisions:
    - name: long-context-route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: context
            name: long_context
      modelRefs:
        - model: test-model
          use_reasoning: false
`

// The RAG key is valid only as written, the context limit only when unset.
const envReferenceMixedRouting = `routing:
  modelCards:
    - name: test-model
  signals:
    context:
      - name: long_context
        min_tokens: 8K
        max_tokens: ${DEPLOY_TEST_CONTEXT_MAX}
  decisions:
    - name: long-context-docs
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: context
            name: long_context
      modelRefs:
        - model: test-model
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_docs
              api_key: ${DEPLOY_TEST_OPENAI_API_KEY}
`

const envReferenceKBRouting = `routing:
  modelCards:
    - name: test-model
  signals:
    kb:
      - name: private-signal
        kb: private
        target:
          kind: label
          value: private
  decisions:
    - name: private-route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: kb
            name: private-signal
      modelRefs:
        - model: test-model
          use_reasoning: false
`

func absoluteKBConfig(kbDir string) string {
	return `version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          provider: vllm
          endpoint: 127.0.0.1:8000
          protocol: http
` + envReferenceKBRouting + `global:
  model_catalog:
    kbs:
      - name: private
        source:
          path: ` + kbDir + `
          manifest: labels.json
        threshold: 0.5
`
}

// The Router resolves these references with its own environment, which the
// Dashboard does not have, so Deploy must accept them and write them as is.
func TestDeployHandler_LeavesEnvironmentReferencesToRouter(t *testing.T) {
	for _, name := range []string{
		"DEPLOY_TEST_OPENAI_API_KEY",
		"DEPLOY_TEST_EJECTION_TIME",
		"DEPLOY_TEST_BACKEND_HOST",
		"DEPLOY_TEST_CONTEXT_MIN",
		"DEPLOY_TEST_CONTEXT_MAX",
	} {
		t.Setenv(name, "")
	}
	kbDir := t.TempDir()
	manifest := []byte(`{"labels":{"private":{"exemplars":["my account number is"]}}}`)
	if err := os.WriteFile(filepath.Join(kbDir, "labels.json"), manifest, 0o600); err != nil {
		t.Fatalf("write labels manifest: %v", err)
	}

	tests := []struct {
		name    string
		config  string
		deploy  string
		written []string
	}{
		{
			name:    "RAG API key without a default",
			deploy:  envReferenceRAGRouting,
			written: []string{"api_key: ${DEPLOY_TEST_OPENAI_API_KEY}"},
		},
		{
			name:   "preserved provider settings with defaults",
			config: envReferenceProviderConfig,
			deploy: envReferenceDeployRouting,
			written: []string{
				"base_ejection_time: ${DEPLOY_TEST_EJECTION_TIME:-30s}",
				"endpoint: ${DEPLOY_TEST_BACKEND_HOST:-127.0.0.1}:8000",
			},
		},
		{
			name:   "context limits",
			deploy: envReferenceContextRouting,
			written: []string{
				"min_tokens: ${DEPLOY_TEST_CONTEXT_MIN:-8K}",
				"max_tokens: ${DEPLOY_TEST_CONTEXT_MAX}",
			},
		},
		{
			name:   "required key with unset formatted values",
			config: strings.Replace(envReferenceProviderConfig, "${DEPLOY_TEST_EJECTION_TIME:-30s}", "${DEPLOY_TEST_EJECTION_TIME}", 1),
			deploy: envReferenceMixedRouting,
			written: []string{
				"api_key: ${DEPLOY_TEST_OPENAI_API_KEY}",
				"max_tokens: ${DEPLOY_TEST_CONTEXT_MAX}",
				"base_ejection_time: ${DEPLOY_TEST_EJECTION_TIME}",
			},
		},
		{
			name:    "absolute knowledge base source",
			config:  absoluteKBConfig(kbDir),
			deploy:  envReferenceKBRouting,
			written: []string{"path: " + kbDir},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempDir := t.TempDir()
			configPath := createValidTestConfig(t, tempDir)
			if tt.config != "" {
				if err := os.WriteFile(configPath, []byte(tt.config), 0o600); err != nil {
					t.Fatalf("write config: %v", err)
				}
			}
			bodyBytes, _ := json.Marshal(DeployRequest{YAML: tt.deploy, Mode: DeployModeReplace})
			req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			DeployHandler(configPath, false, tempDir)(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("Expected 200, got %d. Body: %s", w.Code, w.Body.String())
			}
			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatalf("Failed to read config after deploy: %v", err)
			}
			for _, want := range tt.written {
				if !strings.Contains(string(data), want) {
					t.Fatalf("expected %q in the written config:\n%s", want, data)
				}
			}
		})
	}
}
