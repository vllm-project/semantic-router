package router

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRegisterAccessControlUnavailableCoversInferenceSurfaces(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerAccessControlUnavailable(mux, errors.New("store unavailable"))

	tests := []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/v1/access-control/self"},
		{method: http.MethodGet, path: "/api/playground/v1/models"},
		{method: http.MethodPost, path: "/api/playground/v1/chat/completions"},
		{method: http.MethodGet, path: "/v1/models"},
		{method: http.MethodPost, path: "/v1/chat/completions"},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(test.method, test.path, nil)
			mux.ServeHTTP(response, request)

			if response.Code != http.StatusServiceUnavailable {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusServiceUnavailable, response.Body.String())
			}
			if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
				t.Fatalf("Content-Type=%q want application/json", contentType)
			}
			if !strings.Contains(response.Body.String(), "inference access control is unavailable") {
				t.Fatalf("unexpected body: %s", response.Body.String())
			}
		})
	}
}
