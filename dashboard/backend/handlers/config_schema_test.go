package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

type configSchemaCredentialProvider struct {
	token string
}

func (p configSchemaCredentialProvider) ManagementCredential() (string, error) {
	return p.token, nil
}

func TestConfigSchemaHandlerDefaultsToBundledIndex(t *testing.T) {
	handler := ConfigSchemaHandler("", nil)
	request := httptest.NewRequest(http.MethodGet, "/api/router/config/schema", nil)
	response := httptest.NewRecorder()
	handler(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var index map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &index); err != nil {
		t.Fatalf("invalid index: %v", err)
	}
	if index["default_view"] != configschema.ViewIndex {
		t.Fatalf("default view=%v", index["default_view"])
	}
	etag := response.Header().Get("ETag")
	if etag == "" {
		t.Fatal("schema ETag is missing")
	}
	if response.Header().Get(configSchemaSourceHeader) != "bundled" {
		t.Fatalf("schema source=%q", response.Header().Get(configSchemaSourceHeader))
	}

	notModified := httptest.NewRecorder()
	cachedRequest := httptest.NewRequest(http.MethodGet, "/api/router/config/schema", nil)
	cachedRequest.Header.Set("If-None-Match", etag)
	handler(notModified, cachedRequest)
	if notModified.Code != http.StatusNotModified || notModified.Body.Len() != 0 {
		t.Fatalf("conditional response status=%d body=%q", notModified.Code, notModified.Body.String())
	}

	methodNotAllowed := httptest.NewRecorder()
	handler(methodNotAllowed, httptest.NewRequest(http.MethodPost, "/api/router/config/schema", nil))
	if methodNotAllowed.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status=%d, want %d", methodNotAllowed.Code, http.StatusMethodNotAllowed)
	}

	full := httptest.NewRecorder()
	handler(full, httptest.NewRequest(http.MethodGet, "/api/router/config/schema?view=full", nil))
	if full.Header().Get("ETag") != configschema.ETag() {
		t.Fatal("full schema ETag is missing")
	}
	var schema map[string]any
	if err := json.Unmarshal(full.Body.Bytes(), &schema); err != nil {
		t.Fatalf("invalid full schema: %v", err)
	}
	if schema["$id"] != configschema.SchemaID {
		t.Fatalf("schema id=%v", schema["$id"])
	}
}

func TestConfigSchemaHandlerPrefersRuntimeAndForwardsViews(t *testing.T) {
	var receivedQuery url.Values
	var receivedAuthorization string
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedQuery = r.URL.Query()
		receivedAuthorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("ETag", `"runtime"`)
		_, _ = w.Write([]byte(`{"source":"router","x-vllm-sr-view":{"view":"section"}}`))
	}))
	t.Cleanup(runtime.Close)

	handler := ConfigSchemaHandler(runtime.URL, configSchemaCredentialProvider{token: "router-service-token"})
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/router/config/schema?view=section&path=global.router", nil)
	request.Header.Set("Authorization", "Bearer browser-user-token")
	request.Header.Set("Cookie", "dashboard_session=browser-session")
	handler(response, request)

	if response.Code != http.StatusOK || response.Body.String() != `{"source":"router","x-vllm-sr-view":{"view":"section"}}` {
		t.Fatalf("status=%d body=%q", response.Code, response.Body.String())
	}
	if receivedQuery.Get("view") != "section" || receivedQuery.Get("path") != "global.router" {
		t.Fatalf("runtime query=%q", receivedQuery)
	}
	if receivedAuthorization != "Bearer router-service-token" {
		t.Fatalf("runtime Authorization=%q", receivedAuthorization)
	}
	if response.Header().Get(configSchemaSourceHeader) != "runtime" {
		t.Fatalf("schema source=%q", response.Header().Get(configSchemaSourceHeader))
	}
	if response.Header().Get(configSchemaMatchHeader) != "false" {
		t.Fatalf("schema match=%q", response.Header().Get(configSchemaMatchHeader))
	}
}

func TestConfigSchemaHandlerFallsBackWhenRuntimeIsUnavailable(t *testing.T) {
	runtime := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(runtime.Close)

	handler := ConfigSchemaHandler(runtime.URL, nil)
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodGet, "/api/router/config/schema?view=index", nil))

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get(configSchemaSourceHeader) != "bundled" {
		t.Fatalf("schema source=%q", response.Header().Get(configSchemaSourceHeader))
	}
	var index map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &index); err != nil {
		t.Fatalf("invalid index: %v", err)
	}
	if index["default_view"] != configschema.ViewIndex {
		t.Fatalf("index=%v", index)
	}
}

var _ routerauth.CredentialProvider = configSchemaCredentialProvider{}
