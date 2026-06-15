package apiserver

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// TestSetupRoutesConfigEndpoints verifies the config API surface exposed by setupRoutes.
func TestSetupRoutesConfigEndpoints(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	mux := apiServer.setupRoutes()

	tests := []struct {
		method      string
		path        string
		shouldExist bool
	}{
		{method: http.MethodGet, path: "/health", shouldExist: true},
		{method: http.MethodGet, path: "/config/router", shouldExist: true},
		{method: http.MethodPatch, path: "/config/router", shouldExist: true},
		{method: http.MethodPut, path: "/config/router", shouldExist: true},
		{method: http.MethodPost, path: "/config/router/rollback", shouldExist: true},
		{method: http.MethodGet, path: "/config/router/versions", shouldExist: true},
		{method: http.MethodGet, path: "/config/classification", shouldExist: false},
		{method: http.MethodPut, path: "/config/classification", shouldExist: false},
		{method: http.MethodGet, path: "/config/system-prompts", shouldExist: false},
		{method: http.MethodPut, path: "/config/system-prompts", shouldExist: false},
		{method: http.MethodPost, path: "/config/deploy", shouldExist: false},
		{method: http.MethodPost, path: "/config/rollback", shouldExist: false},
		{method: http.MethodGet, path: "/config/versions", shouldExist: false},
	}

	for _, tt := range tests {
		req := httptest.NewRequest(tt.method, tt.path, nil)
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)

		if tt.shouldExist && rr.Code == http.StatusNotFound {
			t.Errorf("expected endpoint %s %s to exist, but got 404", tt.method, tt.path)
		}
		if !tt.shouldExist && rr.Code != http.StatusNotFound {
			t.Errorf("expected endpoint %s %s to return 404, got %d", tt.method, tt.path, rr.Code)
		}
	}
}
