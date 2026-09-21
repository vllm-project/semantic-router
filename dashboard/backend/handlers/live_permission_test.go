package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func revokedRequest(t *testing.T, method, path string, body []byte) *http.Request {
	t.Helper()
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	return request.WithContext(auth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
		return fmt.Errorf("%w: permission revoked", auth.ErrPermissionDenied)
	}))
}

func TestUpdateConfigHandlerRejectsRevokedPermissionBeforeWrite(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read original config: %v", err)
	}
	body, err := json.Marshal(canonicalConfigBody("10.0.0.9:8000"))
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}

	recorder := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(recorder, revokedRequest(t, http.MethodPost, "/api/router/config/update", body))

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d body=%s", recorder.Code, http.StatusForbidden, recorder.Body.String())
	}
	current, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after rejected update: %v", err)
	}
	if !bytes.Equal(current, original) {
		t.Fatal("config file was mutated after permission revocation")
	}
}

func TestDeployHandlerRejectsRevokedPermissionBeforeWrite(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read original config: %v", err)
	}
	body, err := json.Marshal(DeployRequest{YAML: "routing: {}\n"})
	if err != nil {
		t.Fatalf("marshal deploy request: %v", err)
	}

	recorder := httptest.NewRecorder()
	DeployHandler(configPath, false, tempDir)(recorder, revokedRequest(t, http.MethodPost, "/api/router/config/deploy", body))

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d body=%s", recorder.Code, http.StatusForbidden, recorder.Body.String())
	}
	current, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after rejected deploy: %v", err)
	}
	if !bytes.Equal(current, original) {
		t.Fatal("config file was mutated after permission revocation")
	}
}
