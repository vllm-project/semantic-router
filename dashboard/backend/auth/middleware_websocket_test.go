package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAuthenticateRequestRejectsUntrustedProtectedWebSocketOrigins(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "websocket-origin@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	testCases := []struct {
		name   string
		path   string
		origin string
	}{
		{name: "room sibling origin", path: "/api/openclaw/rooms/room/ws", origin: "http://evil.example.com"},
		{name: "embedded proxy sibling origin", path: "/embedded/openclaw/worker/socket", origin: "http://evil.example.com"},
		{name: "room missing origin", path: "/api/openclaw/rooms/room/ws"},
		{name: "embedded proxy missing origin", path: "/embedded/openclaw/worker/socket"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nextCalled := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusNoContent)
			}))
			request := httptest.NewRequest(http.MethodGet, "http://play.example.com"+tc.path, nil)
			request.Host = "play.example.com"
			request.Header.Set("Authorization", "Bearer "+token)
			request.Header.Set("Connection", "keep-alive, Upgrade")
			request.Header.Set("Upgrade", "websocket")
			if tc.origin != "" {
				request.Header.Set("Origin", tc.origin)
			}
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, request)

			if recorder.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
			}
			if nextCalled {
				t.Fatal("untrusted WebSocket request reached the protected handler")
			}
			if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
		})
	}
}

func TestAuthenticateRequestAllowsSameOriginProtectedWebSocket(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "same-origin-websocket@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	nextCalled := false
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))
	request := httptest.NewRequest(http.MethodGet, "http://internal:8700/embedded/openclaw/worker/socket", nil)
	request.Host = "play.example.com"
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Origin", "https://play.example.com")
	request.Header.Set(forwardedProtoHeader, "https")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNoContent || !nextCalled {
		t.Fatalf("same-origin WebSocket result = status %d, next %v", recorder.Code, nextCalled)
	}
}
