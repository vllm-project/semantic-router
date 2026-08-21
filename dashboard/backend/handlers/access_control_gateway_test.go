package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestAccessGatewayRejectsAnonymousOpenAIEndpoints(t *testing.T) {
	handler, err := NewAccessGatewayHandler(nil, "http://envoy.internal:8899")
	if err != nil {
		t.Fatalf("NewAccessGatewayHandler() error = %v", err)
	}
	tests := []struct {
		name   string
		method string
		path   string
		body   string
		handle http.HandlerFunc
	}{
		{name: "models", method: http.MethodGet, path: "/v1/models", handle: handler.Models},
		{name: "chat", method: http.MethodPost, path: "/v1/chat/completions", body: `{"model":"vllm-sr/mom"}`, handle: handler.ChatCompletions},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
			response := httptest.NewRecorder()
			test.handle(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d, want %d; body=%s", response.Code, http.StatusUnauthorized, response.Body.String())
			}
			if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
				t.Fatalf("Content-Type = %q, want application/json", contentType)
			}
			if !strings.Contains(response.Body.String(), "bearer API key is required") {
				t.Fatalf("response must explain the missing API key: %s", response.Body.String())
			}
		})
	}
}

func TestDashboardIndividualModelAccessIsAdminOnly(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/api/playground/v1/chat/completions", nil)
	if dashboardAllowsIndividualModels(request) {
		t.Fatal("anonymous request must not access individual models")
	}

	for _, test := range []struct {
		role string
		want bool
	}{
		{role: auth.RoleRead, want: false},
		{role: auth.RoleWrite, want: false},
		{role: auth.RoleAdmin, want: true},
	} {
		ctx := auth.WithAuthContext(context.Background(), auth.AuthContext{Role: test.role})
		if got := dashboardAllowsIndividualModels(request.WithContext(ctx)); got != test.want {
			t.Fatalf("role %q allowed = %v, want %v", test.role, got, test.want)
		}
	}
}

func TestGatewayResponseHeadersPreserveRouterMetadata(t *testing.T) {
	for _, header := range []string{
		"Content-Type",
		"X-Request-ID",
		"X-VSR-Selected-Model",
		"x-vsr-selected-decision",
	} {
		if !isGatewayResponseHeader(header) {
			t.Fatalf("expected %q to be forwarded", header)
		}
	}
	if isGatewayResponseHeader("X-Internal-Secret") {
		t.Fatal("unrecognized upstream headers must remain private")
	}
}
