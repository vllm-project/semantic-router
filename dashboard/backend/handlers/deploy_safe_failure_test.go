package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// safeFailureFragment references a model the merged config does not define.
// The YAML is syntactically valid, so it clears the parse stage and can only be
// rejected by config validation.
const safeFailureFragment = `routing:
  decisions:
    - name: e2e-safe-failure
      description: Deploy probe that must be rejected by config validation
      priority: 3
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: e2e-nonexistent-model
          use_reasoning: false
`

// TestDeployHandler_SemanticRejectionLeavesConfigUnchanged pins the safe-failure
// half of the deploy contract: a config that parses but does not validate must
// be rejected before the active config is touched. deployDirectWrite validates
// the merged document before it creates a backup or writes, so a rejected
// deploy must leave the serving config byte-for-byte identical.
func TestDeployHandler_SemanticRejectionLeavesConfigUnchanged(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read seeded config: %v", err)
	}

	body, err := json.Marshal(DeployRequest{YAML: safeFailureFragment, Mode: DeployModeMerge})
	if err != nil {
		t.Fatalf("failed to encode deploy request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	DeployHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for a semantically invalid config, got %d: %s", w.Code, w.Body.String())
	}

	var response struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if decodeErr := json.NewDecoder(w.Body).Decode(&response); decodeErr != nil {
		t.Fatalf("deploy rejection is not JSON: %v", decodeErr)
	}

	// The distinction matters: a yaml_parse_error would mean the fragment was
	// rejected for syntax and the validation stage was never reached, which
	// would make this test vacuous.
	if response.Error != "config_validation_error" {
		t.Fatalf("expected error=config_validation_error, got %q (message: %s)", response.Error, response.Message)
	}
	if !strings.Contains(response.Message, "e2e-nonexistent-model") {
		t.Fatalf("rejection message should name the unknown model, got: %s", response.Message)
	}

	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to re-read config: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("a rejected deploy must not modify the active config\nbefore:\n%s\nafter:\n%s", before, after)
	}
}
