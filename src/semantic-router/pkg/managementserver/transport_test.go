package managementserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const transportEventPath = "/management/v1/agent-sessions/10000000-0000-4000-8000-000000000001/events"

type transportRegistrar struct{}

func (transportRegistrar) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /management/v1/transport", func(response http.ResponseWriter, _ *http.Request) {
		writeProviderJSON(response, http.StatusOK, map[string]bool{"ok": true}, "transport-request")
	})
	mux.HandleFunc("POST /management/v1/transport", func(response http.ResponseWriter, _ *http.Request) {
		writeProviderJSON(response, http.StatusOK, map[string]bool{"ok": true}, "transport-request")
	})
	mux.HandleFunc("GET "+transportEventPath, func(response http.ResponseWriter, request *http.Request) {
		if acceptsAgentSSE(request) {
			setProviderResponseHeaders(response, "transport-request")
			response.Header().Set("Content-Type", managementapi.EventStreamMediaType)
			_, _ = response.Write([]byte("event: ready\ndata: {}\n\n"))
			return
		}
		writeProviderJSON(response, http.StatusOK, map[string]bool{"history": true}, "transport-request")
	})
	mux.HandleFunc("GET "+routingCurrentExportPath, func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", managementapi.YAMLMediaType+"; charset=utf-8")
		response.WriteHeader(http.StatusOK)
		_, _ = response.Write([]byte("version: v0.3\n"))
	})
	mux.HandleFunc("GET /management/v1/no-content", func(response http.ResponseWriter, _ *http.Request) {
		setProviderResponseHeaders(response, "transport-request")
		response.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /management/v1/typed-error", func(response http.ResponseWriter, _ *http.Request) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", "transport-request")
	})
}

func TestManagementTransportNegotiatesYAMLOnlyForRoutingExport(t *testing.T) {
	handler := newTransportTestHandler(t)
	tests := []struct {
		name       string
		accept     string
		wantStatus int
	}{
		{name: "YAML export", accept: managementapi.YAMLMediaType, wantStatus: http.StatusOK},
		{name: "vendor JSON rejected", accept: managementapi.JSONMediaType, wantStatus: http.StatusNotAcceptable},
		{name: "wildcard rejected", accept: "*/*", wantStatus: http.StatusNotAcceptable},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, routingCurrentExportPath, nil)
			request.Header.Set("Accept", test.accept)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			if test.wantStatus == http.StatusOK {
				if got := response.Header().Get("Content-Type"); got != managementapi.YAMLMediaType+"; charset=utf-8" {
					t.Fatalf("Content-Type = %q", got)
				}
				if got := response.Body.String(); got != "version: v0.3\n" {
					t.Fatalf("body = %q", got)
				}
			}
		})
	}
}

func TestManagementTransportNegotiatesExplicitVendorAccept(t *testing.T) {
	handler := newTransportTestHandler(t)
	tests := []struct {
		name       string
		accept     []string
		wantStatus int
	}{
		{name: "exact", accept: []string{managementapi.JSONMediaType}, wantStatus: http.StatusOK},
		{name: "list and quality", accept: []string{"application/json, " + managementapi.JSONMediaType + ";q=0.5"}, wantStatus: http.StatusOK},
		{name: "multiple fields", accept: []string{"application/json", managementapi.JSONMediaType}, wantStatus: http.StatusOK},
		{name: "missing", wantStatus: http.StatusNotAcceptable},
		{name: "generic JSON", accept: []string{"application/json"}, wantStatus: http.StatusNotAcceptable},
		{name: "wildcard", accept: []string{"*/*"}, wantStatus: http.StatusNotAcceptable},
		{name: "explicit veto beats wildcard", accept: []string{managementapi.JSONMediaType + ";q=0, */*;q=1"}, wantStatus: http.StatusNotAcceptable},
		{name: "malformed quality", accept: []string{managementapi.JSONMediaType + ";q=1.1"}, wantStatus: http.StatusNotAcceptable},
		{name: "event stream is not ordinary JSON", accept: []string{managementapi.EventStreamMediaType}, wantStatus: http.StatusNotAcceptable},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/management/v1/transport", nil)
			for _, value := range test.accept {
				request.Header.Add("Accept", value)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			if response.Header().Get("Content-Type") != managementapi.JSONMediaType {
				t.Fatalf("Content-Type = %q", response.Header().Get("Content-Type"))
			}
			if test.wantStatus == http.StatusNotAcceptable && !strings.Contains(response.Body.String(), `"code":"not_acceptable"`) {
				t.Fatalf("error body = %s", response.Body.String())
			}
		})
	}
}

func TestManagementTransportRequiresVendorContentTypeOnlyForBodies(t *testing.T) {
	handler := newTransportTestHandler(t)
	tests := []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
	}{
		{name: "vendor", contentType: managementapi.JSONMediaType, body: `{}`, wantStatus: http.StatusOK},
		{name: "utf8", contentType: managementapi.JSONMediaType + "; charset=UTF-8", body: `{}`, wantStatus: http.StatusOK},
		{name: "missing", body: `{}`, wantStatus: http.StatusUnsupportedMediaType},
		{name: "generic JSON", contentType: "application/json", body: `{}`, wantStatus: http.StatusUnsupportedMediaType},
		{name: "unsupported parameter", contentType: managementapi.JSONMediaType + "; profile=legacy", body: `{}`, wantStatus: http.StatusUnsupportedMediaType},
		{name: "bodyless ignores content type", contentType: "application/json", wantStatus: http.StatusOK},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var body *strings.Reader
			if test.body != "" {
				body = strings.NewReader(test.body)
			} else {
				body = strings.NewReader("")
			}
			request := httptest.NewRequest(http.MethodPost, "/management/v1/transport", body)
			request.Header.Set("Accept", managementapi.JSONMediaType)
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			if test.wantStatus == http.StatusUnsupportedMediaType && !strings.Contains(response.Body.String(), `"code":"unsupported_media_type"`) {
				t.Fatalf("error body = %s", response.Body.String())
			}
		})
	}
}

func TestManagementTransportNegotiatesAgentEventRepresentations(t *testing.T) {
	handler := newTransportTestHandler(t)
	tests := []struct {
		name            string
		accept          string
		wantStatus      int
		wantContentType string
	}{
		{name: "stream", accept: managementapi.EventStreamMediaType, wantStatus: 200, wantContentType: managementapi.EventStreamMediaType},
		{name: "durable history", accept: managementapi.JSONMediaType, wantStatus: 200, wantContentType: managementapi.JSONMediaType},
		{name: "quality selects stream", accept: managementapi.JSONMediaType + ";q=0.2, " + managementapi.EventStreamMediaType + ";q=0.8", wantStatus: 200, wantContentType: managementapi.EventStreamMediaType},
		{name: "quality selects history", accept: managementapi.EventStreamMediaType + ";q=0, " + managementapi.JSONMediaType + ";q=0.1", wantStatus: 200, wantContentType: managementapi.JSONMediaType},
		{name: "wildcard", accept: "*/*", wantStatus: 406, wantContentType: managementapi.JSONMediaType},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, transportEventPath, nil)
			request.Header.Set("Accept", test.accept)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus || response.Header().Get("Content-Type") != test.wantContentType {
				t.Fatalf("response = %d %q %s", response.Code, response.Header().Get("Content-Type"), response.Body.String())
			}
		})
	}
}

func TestManagementTransportPreservesNoContentErrorsAndUnknownRoutes(t *testing.T) {
	handler := newTransportTestHandler(t)

	noContent := httptest.NewRequest(http.MethodGet, "/management/v1/no-content", nil)
	noContent.Header.Set("Accept", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, noContent)
	if response.Code != http.StatusNoContent || response.Header().Get("Content-Type") != "" {
		t.Fatalf("no-content response = %d %q", response.Code, response.Header().Get("Content-Type"))
	}

	typedError := httptest.NewRequest(http.MethodGet, "/management/v1/typed-error", nil)
	typedError.Header.Set("Accept", managementapi.JSONMediaType)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, typedError)
	if response.Code != http.StatusBadRequest || response.Header().Get("Content-Type") != managementapi.JSONMediaType {
		t.Fatalf("typed error response = %d %q", response.Code, response.Header().Get("Content-Type"))
	}

	unknown := httptest.NewRequest(http.MethodGet, "/management/v1/not-registered", nil)
	unknown.Header.Set("Accept", managementapi.JSONMediaType)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, unknown)
	if response.Code != http.StatusNotFound {
		t.Fatalf("unknown route status = %d", response.Code)
	}
}

func newTransportTestHandler(t *testing.T) http.Handler {
	t.Helper()
	server, err := NewServer(&catalogRuntimeStub{}, transportRegistrar{})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	server.Register(mux)
	return mux
}
