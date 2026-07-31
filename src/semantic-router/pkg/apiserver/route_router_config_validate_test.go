//go:build !windows && cgo

package apiserver

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleConfigValidate(t *testing.T) {
	body := `{"yaml":"version: v0.3\nproviders:\n  defaults:\n    default_model: m1\n  models:\n    - name: m1\n      backend_refs:\n        - endpoint: 127.0.0.1:8000\nrouting:\n  modelCards:\n    - name: m1\n"}`
	request := httptest.NewRequest("POST", "/config/router/validate", strings.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 200 {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"valid":true`) {
		t.Fatalf("body = %s", response.Body.String())
	}
}
