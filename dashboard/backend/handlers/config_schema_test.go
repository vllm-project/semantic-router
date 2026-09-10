package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func TestConfigSchemaHandlerServesRouterContract(t *testing.T) {
	handler := ConfigSchemaHandler()
	request := httptest.NewRequest(http.MethodGet, "/api/router/config/schema", nil)
	response := httptest.NewRecorder()
	handler(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var schema map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &schema); err != nil {
		t.Fatalf("invalid schema: %v", err)
	}
	if schema["$id"] != configschema.SchemaID {
		t.Fatalf("schema id=%v", schema["$id"])
	}
	if response.Header().Get("ETag") != configschema.ETag() {
		t.Fatal("schema ETag is missing")
	}

	notModified := httptest.NewRecorder()
	cachedRequest := httptest.NewRequest(http.MethodGet, "/api/router/config/schema", nil)
	cachedRequest.Header.Set("If-None-Match", configschema.ETag())
	handler(notModified, cachedRequest)
	if notModified.Code != http.StatusNotModified || notModified.Body.Len() != 0 {
		t.Fatalf("conditional response status=%d body=%q", notModified.Code, notModified.Body.String())
	}

	methodNotAllowed := httptest.NewRecorder()
	handler(methodNotAllowed, httptest.NewRequest(http.MethodPost, "/api/router/config/schema", nil))
	if methodNotAllowed.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status=%d, want %d", methodNotAllowed.Code, http.StatusMethodNotAllowed)
	}
}
