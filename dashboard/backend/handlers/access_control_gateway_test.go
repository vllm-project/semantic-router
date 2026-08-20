package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
