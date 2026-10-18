package auth

import (
	"bytes"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/crypto/bcrypt"
)

func assertAuthStoreError(t *testing.T, handler http.Handler, request *http.Request, status int, message, storeError string) {
	t.Helper()
	var logs bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&logs)
	defer log.SetOutput(previous)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != status || response.Body.String() != message+"\n" {
		t.Errorf("status = %d, body = %q; want %d with %q", response.Code, response.Body.String(), status, message)
	}
	if strings.Contains(response.Body.String(), storeError) {
		t.Errorf("response leaks store error: %q", response.Body.String())
	}
	if !strings.Contains(logs.String(), storeError) {
		t.Errorf("log = %q, want store error %q", logs.String(), storeError)
	}
}

func TestAuthHandlersHideStoreErrors(t *testing.T) {
	endpoints := []struct {
		method, path, body, table, message string
		status                             int
	}{
		{http.MethodPost, "/api/auth/login", `{"email":"admin@example.com","password":"secret-password"}`, "users", "authentication failed", http.StatusUnauthorized},
		{http.MethodGet, "/api/auth/bootstrap/can-register", "", "users", "failed to check bootstrap status", http.StatusInternalServerError},
		{http.MethodPost, "/api/auth/bootstrap/register", `{"email":"new@example.com","password":"secret-password"}`, "users", "failed to check bootstrap status", http.StatusInternalServerError},
		{http.MethodGet, "/api/admin/users", "", "users", "failed to list users", http.StatusInternalServerError},
		{http.MethodGet, "/api/admin/users/target", "", "users", "failed to load user", http.StatusInternalServerError},
		{http.MethodPatch, "/api/admin/users/target", `{"role":"read"}`, "users", "failed to load user", http.StatusInternalServerError},
		{http.MethodDelete, "/api/admin/users/target", "", "users", "failed to load user", http.StatusInternalServerError},
		{http.MethodGet, "/api/admin/permissions", "", "role_permissions", "failed to list permissions", http.StatusInternalServerError},
		{http.MethodGet, "/api/admin/audit-logs", "", "user_audit_logs", "failed to query audit logs", http.StatusInternalServerError},
		{http.MethodPost, "/api/admin/users/password", `{"userId":"target","password":"secret-password"}`, "users", "failed to load user", http.StatusInternalServerError},
		{http.MethodPost, "/api/admin/invitations", `{"email":"new@example.com","name":"New User","role":"read"}`, "users", "failed to create invitation", http.StatusInternalServerError},
		{http.MethodGet, "/api/admin/invitations", "", "dashboard_invitations", "failed to list invitations", http.StatusInternalServerError},
		{http.MethodPost, "/api/admin/invitations/target/rotate", "", "dashboard_invitations", "failed to process invitation", http.StatusInternalServerError},
		{http.MethodDelete, "/api/admin/invitations/target", "", "dashboard_invitations", "failed to process invitation", http.StatusInternalServerError},
	}
	for _, failure := range []string{"closed database", "schema drift"} {
		t.Run(failure, func(t *testing.T) {
			for _, endpoint := range endpoints {
				t.Run(endpoint.method+" "+endpoint.path, func(t *testing.T) {
					svc := newTestAuthService(t)
					newStoreErrorTestUser(t, svc, "active")
					svc.SetAllowOpenBootstrap(true)
					storeError := "sql: database is closed"
					if failure == "closed database" {
						if err := svc.store.Close(); err != nil {
							t.Fatal(err)
						}
					} else {
						if _, err := svc.store.db.Exec("DROP TABLE " + endpoint.table); err != nil {
							t.Fatal(err)
						}
						storeError = "no such table: " + endpoint.table
					}
					mux := AuthRoutes(svc)
					RegisterAdminRoutes(mux, svc)
					request := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(endpoint.body))
					request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
						UserID: "admin", Perms: map[string]bool{PermUsersManage: true, PermUsersView: true},
					}))
					assertAuthStoreError(t, mux, request, endpoint.status, endpoint.message, storeError)
				})
			}
		})
	}
}

func newStoreErrorTestUser(t *testing.T, svc *Service, status string) *User {
	t.Helper()
	hash, err := bcrypt.GenerateFromPassword([]byte("secret-password"), bcrypt.MinCost)
	if err != nil {
		t.Fatal(err)
	}
	user, err := svc.store.CreateUser(t.Context(), "admin@example.com", "Admin", string(hash), RoleAdmin, status)
	if err != nil {
		t.Fatal(err)
	}
	return user
}

func TestLoginHandlerKeepsUserVisibleErrors(t *testing.T) {
	for _, test := range []struct{ name, email, password, status, message string }{
		{"missing user", "unknown@example.com", "secret-password", "active", "invalid credentials"},
		{"wrong password", "admin@example.com", "wrong-password", "active", "invalid credentials"},
		{"inactive user", "admin@example.com", "secret-password", "inactive", "user is not active"},
	} {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			newStoreErrorTestUser(t, svc, test.status)
			request := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(`{"email":"`+test.email+`","password":"`+test.password+`"}`))
			response := httptest.NewRecorder()
			AuthRoutes(svc).ServeHTTP(response, request)
			if response.Code != http.StatusUnauthorized || response.Body.String() != test.message+"\n" {
				t.Fatalf("status = %d, body = %q; want 401 with %q", response.Code, response.Body.String(), test.message)
			}
		})
	}
}

func TestAuthHandlersHideErrorsAfterInitialLookup(t *testing.T) {
	const triggerError = "internal store write failure"
	for _, test := range []struct {
		name, method, path, body, failureSQL, storeError, message string
		status                                                    int
		bootstrap                                                 bool
	}{
		{"login update", http.MethodPost, "/api/auth/login", `{"email":"admin@example.com","password":"secret-password"}`, "CREATE TRIGGER fail_login BEFORE UPDATE ON users BEGIN SELECT RAISE(FAIL, '" + triggerError + "'); END", triggerError, "authentication failed", http.StatusUnauthorized, false},
		{"login session", http.MethodPost, "/api/auth/login", `{"email":"admin@example.com","password":"secret-password"}`, "DROP TABLE auth_sessions", "no such table: auth_sessions", "authentication failed", http.StatusUnauthorized, false},
		{"login permissions", http.MethodPost, "/api/auth/login", `{"email":"admin@example.com","password":"secret-password"}`, "DROP TABLE role_permissions", "no such table: role_permissions", "failed to create session", http.StatusInternalServerError, false},
		{"bootstrap create", http.MethodPost, "/api/auth/bootstrap/register", `{"email":"new@example.com","password":"secret-password"}`, "CREATE TRIGGER fail_bootstrap BEFORE INSERT ON users BEGIN SELECT RAISE(FAIL, '" + triggerError + "'); END", triggerError, "failed to register admin", http.StatusInternalServerError, true},
		{"bootstrap session", http.MethodPost, "/api/auth/bootstrap/register", `{"email":"new@example.com","password":"secret-password"}`, "DROP TABLE auth_sessions", "no such table: auth_sessions", "failed to create session", http.StatusInternalServerError, true},
		{"bootstrap permissions", http.MethodPost, "/api/auth/bootstrap/register", `{"email":"new@example.com","password":"secret-password"}`, "DROP TABLE role_permissions", "no such table: role_permissions", "failed to create session", http.StatusInternalServerError, true},
		{"patch validation", http.MethodPatch, "/api/admin/users/target", `{"role":"read"}`, "DROP TABLE role_permissions", "no such table: role_permissions", "failed to update user", http.StatusInternalServerError, false},
		{"patch write", http.MethodPatch, "/api/admin/users/target", `{"status":"inactive"}`, "CREATE TRIGGER fail_update BEFORE UPDATE ON users BEGIN SELECT RAISE(FAIL, '" + triggerError + "'); END", triggerError, "failed to update user", http.StatusInternalServerError, false},
		{"delete validation", http.MethodDelete, "/api/admin/users/target", "", "DROP TABLE role_permissions", "no such table: role_permissions", "failed to delete user", http.StatusInternalServerError, false},
		{"delete write", http.MethodDelete, "/api/admin/users/target", "", "CREATE TRIGGER fail_delete BEFORE DELETE ON users BEGIN SELECT RAISE(FAIL, '" + triggerError + "'); END", triggerError, "failed to delete user", http.StatusInternalServerError, false},
		{"password write", http.MethodPost, "/api/admin/users/password", `{"userId":"target","password":"secret-password"}`, "CREATE TRIGGER fail_password BEFORE UPDATE ON users BEGIN SELECT RAISE(FAIL, '" + triggerError + "'); END", triggerError, "failed to update password", http.StatusInternalServerError, false},
		{"invitation write", http.MethodPost, "/api/admin/invitations", `{"kind":"shared","capacity":2}`, "DROP TABLE dashboard_invitations", "no such table: dashboard_invitations", "failed to create invitation", http.StatusInternalServerError, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			if test.bootstrap {
				svc.SetAllowOpenBootstrap(true)
				previous := hashBootstrapPassword
				hashBootstrapPassword = func(*Service, string) (string, error) { return "unused", nil }
				defer func() { hashBootstrapPassword = previous }()
			} else {
				newStoreErrorTestUser(t, svc, "active")
				if _, err := svc.store.db.Exec("INSERT INTO users(id,email,name,password_hash,role,status,created_at,updated_at) VALUES('target','target@example.com','Target','unused','read','active',0,0)"); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := svc.store.db.Exec(test.failureSQL); err != nil {
				t.Fatal(err)
			}
			mux := AuthRoutes(svc)
			RegisterAdminRoutes(mux, svc)
			request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
			request = request.WithContext(WithAuthContext(request.Context(), AuthContext{UserID: "admin", Perms: map[string]bool{PermUsersManage: true, PermUsersView: true}}))
			assertAuthStoreError(t, mux, request, test.status, test.message, test.storeError)
		})
	}
}

func TestAdminUserLookupKeepsNotFoundResponse(t *testing.T) {
	svc := newTestAuthService(t)
	mux := AuthRoutes(svc)
	RegisterAdminRoutes(mux, svc)
	for _, test := range []struct{ method, path, body string }{
		{http.MethodGet, "/api/admin/users/missing", ""},
		{http.MethodPatch, "/api/admin/users/missing", `{"role":"read"}`},
		{http.MethodDelete, "/api/admin/users/missing", ""},
		{http.MethodPost, "/api/admin/users/password", `{"userId":"missing","password":"secret-password"}`},
	} {
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
		request = request.WithContext(WithAuthContext(request.Context(), AuthContext{UserID: "admin", Perms: map[string]bool{PermUsersManage: true, PermUsersView: true}}))
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound || response.Body.String() != "user not found\n" {
			t.Errorf("%s: status = %d, body = %q; want 404 with fixed message", test.method, response.Code, response.Body.String())
		}
	}
}

func TestAdminUserMutationsKeepLastManagerConflict(t *testing.T) {
	for _, test := range []struct{ method, body string }{
		{http.MethodPatch, `{"role":"read"}`},
		{http.MethodPatch, `{"status":"inactive"}`},
		{http.MethodDelete, ""},
	} {
		t.Run(test.method+test.body, func(t *testing.T) {
			svc := newTestAuthService(t)
			user := newStoreErrorTestUser(t, svc, "active")
			mux := AuthRoutes(svc)
			RegisterAdminRoutes(mux, svc)
			request := httptest.NewRequest(test.method, "/api/admin/users/"+user.ID, strings.NewReader(test.body))
			request = request.WithContext(WithAuthContext(request.Context(), AuthContext{
				UserID: "operator", Perms: map[string]bool{PermUsersManage: true},
			}))
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)
			if response.Code != http.StatusConflict || response.Body.String() != "cannot remove the last active user manager\n" {
				t.Fatalf("status = %d, body = %q; want 409 with last-manager message", response.Code, response.Body.String())
			}
		})
	}
}

func TestLoginHandlerHidesStoreErrorsMatchingBusinessMessages(t *testing.T) {
	for _, test := range []struct{ message, failureSQL string }{
		{"invalid credentials", "CREATE TRIGGER fail_login BEFORE UPDATE ON users BEGIN SELECT RAISE(FAIL, 'invalid credentials'); END"},
		{"user is not active", "CREATE TRIGGER fail_login BEFORE UPDATE ON users BEGIN SELECT RAISE(FAIL, 'user is not active'); END"},
	} {
		t.Run(test.message, func(t *testing.T) {
			svc := newTestAuthService(t)
			newStoreErrorTestUser(t, svc, "active")
			if _, err := svc.store.db.Exec(test.failureSQL); err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(`{"email":"admin@example.com","password":"secret-password"}`))
			assertAuthStoreError(t, AuthRoutes(svc), request, http.StatusUnauthorized, "authentication failed", test.message)
		})
	}
}
