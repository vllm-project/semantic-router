package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// newBrokenService builds a Service whose store DB has been closed, so every
// store call returns a driver error. This lets tests verify that handlers do
// not echo driver text (e.g. "sql: database is closed") to clients.
func newBrokenService(t *testing.T) *Service {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	// Close the DB now so all subsequent store calls fail with a driver error.
	if err := store.Close(); err != nil {
		t.Fatalf("store.Close() error = %v", err)
	}
	svc := NewService(store, "test-secret", 1)
	svc.SetAllowOpenBootstrap(true)
	return svc
}

// newSchemaDriftService builds a Service with a live DB connection but with the
// users table dropped, simulating schema drift after a failed migration.
func newSchemaDriftService(t *testing.T) *Service {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if _, err := store.db.ExecContext(context.Background(), `DROP TABLE users`); err != nil {
		t.Fatalf("DROP TABLE users: %v", err)
	}
	svc := NewService(store, "test-secret", 1)
	svc.SetAllowOpenBootstrap(true)
	return svc
}

func assertNoDriverText(t *testing.T, body string) {
	t.Helper()
	driverPhrases := []string{
		"no such table",
		"sql: database is closed",
		"database is locked",
		"SQLITE_",
	}
	for _, phrase := range driverPhrases {
		if strings.Contains(body, phrase) {
			t.Errorf("response body contains driver text %q: %s", phrase, body)
		}
	}
}

func TestLoginHandler_StoreErrorHidesDriverText(t *testing.T) {
	svc := newBrokenService(t)
	body := `{"email":"user@example.com","password":"any-password"}`
	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(body))
	rec := httptest.NewRecorder()

	loginHandler(svc)(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if want := "authentication failed\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
	assertNoDriverText(t, rec.Body.String())
}

func TestBootstrapCanRegisterHandler_StoreErrorHidesDriverText(t *testing.T) {
	svc := newBrokenService(t)
	req := httptest.NewRequest(http.MethodGet, "/api/auth/bootstrap/can-register", nil)
	rec := httptest.NewRecorder()

	bootstrapCanRegisterHandler(svc)(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if want := "failed to check bootstrap status\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
	assertNoDriverText(t, rec.Body.String())
}

func TestLoginHandler_InvalidCredentialsReturnsMessage(t *testing.T) {
	svc := newTestAuthService(t)
	newTestUser(t, svc, "real@example.com", RoleRead, "active")

	body := `{"email":"real@example.com","password":"wrong-password-9"}`
	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(body))
	rec := httptest.NewRecorder()

	loginHandler(svc)(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if want := "invalid credentials\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
}

func TestLoginHandler_InactiveUserReturnsMessage(t *testing.T) {
	svc := newTestAuthService(t)
	newTestUser(t, svc, "inactive@example.com", RoleRead, "inactive")

	body := `{"email":"inactive@example.com","password":"secret-password"}`
	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(body))
	rec := httptest.NewRecorder()

	loginHandler(svc)(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if want := "user is not active\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
}

func TestLoginHandler_SchemaDriftHidesDriverText(t *testing.T) {
	svc := newSchemaDriftService(t)
	body := `{"email":"user@example.com","password":"any-password"}`
	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(body))
	rec := httptest.NewRecorder()

	loginHandler(svc)(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if want := "authentication failed\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
	assertNoDriverText(t, rec.Body.String())
}

func TestAdminUsersListHandler_StoreErrorHidesDriverText(t *testing.T) {
	svc := newSchemaDriftService(t)
	req := httptest.NewRequest(http.MethodGet, "/api/admin/users", nil)
	req = req.WithContext(WithAuthContext(req.Context(), AuthContext{
		Perms: map[string]bool{PermUsersView: true},
	}))
	rec := httptest.NewRecorder()

	adminUsersCollectionHandler(svc)(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if want := "failed to list users\n"; rec.Body.String() != want {
		t.Fatalf("body = %q, want %q", rec.Body.String(), want)
	}
	assertNoDriverText(t, rec.Body.String())
}
