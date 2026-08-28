package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestUpdateConfigHandlerRejectsRevokedPermissionBeforeWrite(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read original config: %v", err)
	}

	bodyBytes, err := json.Marshal(canonicalConfigBody("10.0.0.9:8000"))
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(auth.WithPermissionRevalidator(req.Context(), func(context.Context) error {
		return errors.New("permission revoked")
	}))
	recorder := httptest.NewRecorder()

	UpdateConfigHandler(configPath, false, "")(recorder, req)

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d body=%s", recorder.Code, http.StatusForbidden, recorder.Body.String())
	}
	current, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after rejected update: %v", err)
	}
	if string(current) != string(original) {
		t.Fatal("config file was mutated after permission revocation")
	}
}
