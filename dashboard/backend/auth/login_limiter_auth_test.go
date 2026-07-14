package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestLoginUsesUniformFailureAndDummyHashWork(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	active := newTestUser(t, svc, "active@example.com", RoleRead, defaultUserStatusActive)
	inactive := newTestUser(t, svc, "inactive@example.com", RoleRead, "inactive")
	_ = active
	_ = inactive

	tests := []struct {
		name  string
		email string
	}{
		{name: "missing", email: "missing@example.com"},
		{name: "inactive", email: "inactive@example.com"},
		{name: "wrong password", email: "active@example.com"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			seenHash := ""
			originalVerify := svc.verify
			svc.verify = func(hash, password string) bool {
				calls++
				seenHash = hash
				return originalVerify(hash, password)
			}
			t.Cleanup(func() { svc.verify = originalVerify })

			_, _, err := svc.LoginWithSource(
				context.Background(),
				test.email,
				"definitely-wrong-password",
				"source-"+test.name,
			)
			if !errors.Is(err, ErrInvalidCredentials) {
				t.Fatalf("LoginWithSource() error = %v, want generic invalid credentials", err)
			}
			if calls != 1 {
				t.Fatalf("password verifier calls = %d, want 1", calls)
			}
			if test.name == "missing" && seenHash != dummyPasswordHash {
				t.Fatalf("missing account hash = %q, want dummy hash", seenHash)
			}
		})
	}
}

func TestLoginHandlerReturns429WithRetryAfter(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  100,
		BaseDelay:               3 * time.Second,
		MaxDelay:                3 * time.Second,
	})

	request := func() *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(
			http.MethodPost,
			"/api/auth/login",
			strings.NewReader(`{"email":"missing@example.com","password":"wrong-password-value"}`),
		)
		req.Header.Set("Content-Type", "application/json")
		req.RemoteAddr = "192.0.2.10:4567"
		loginHandler(svc).ServeHTTP(recorder, req)
		return recorder
	}

	if first := request(); first.Code != http.StatusUnauthorized {
		t.Fatalf("first status = %d, want 401", first.Code)
	}
	second := request()
	if second.Code != http.StatusTooManyRequests {
		t.Fatalf("second status = %d, want 429", second.Code)
	}
	if second.Header().Get("Retry-After") != "3" {
		t.Fatalf("Retry-After = %q, want 3", second.Header().Get("Retry-After"))
	}
	if !strings.Contains(second.Body.String(), "too many authentication attempts") {
		t.Fatalf("body = %q, want generic rate-limit message", second.Body.String())
	}
}
