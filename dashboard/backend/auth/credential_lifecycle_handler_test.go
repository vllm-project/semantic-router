package auth

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAdminUserPasswordHandlerReplaysKeyedRetryAndRejectsConflict(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-handler-idempotent@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-handler-idempotent@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)
	firstResponse := postAdminPasswordReset(
		t,
		handler,
		svc,
		actor,
		target.ID,
		"rotated-password",
		"reset-key-handler",
		"",
	)
	if firstResponse.Code != http.StatusOK {
		t.Fatalf("first status = %d, body = %q", firstResponse.Code, firstResponse.Body.String())
	}
	assertPasswordResetResponse(t, firstResponse, false)
	assertTokenUnauthorized(t, svc, targetToken)
	storedAfterFirst := userPasswordHash(t, svc.store, target.ID)

	replayResponse := postAdminPasswordReset(
		t,
		handler,
		svc,
		actor,
		target.ID,
		"rotated-password",
		"reset-key-handler",
		"",
	)
	if replayResponse.Code != http.StatusOK {
		t.Fatalf("replay status = %d, body = %q", replayResponse.Code, replayResponse.Body.String())
	}
	assertPasswordResetResponse(t, replayResponse, true)
	if storedAfterReplay := userPasswordHash(t, svc.store, target.ID); storedAfterReplay != storedAfterFirst {
		t.Fatalf("password hash changed after handler replay")
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 1 {
		t.Fatalf("audit count = %d, want 1 after handler replay", count)
	}

	conflictResponse := postAdminPasswordReset(
		t,
		handler,
		svc,
		actor,
		target.ID,
		"different-password",
		"reset-key-handler",
		"",
	)
	if conflictResponse.Code != http.StatusConflict {
		t.Fatalf("conflict status = %d, want %d; body = %q", conflictResponse.Code, http.StatusConflict, conflictResponse.Body.String())
	}
	if storedAfterConflict := userPasswordHash(t, svc.store, target.ID); storedAfterConflict != storedAfterFirst {
		t.Fatalf("password hash changed after handler idempotency conflict")
	}
}

func TestAdminUserPasswordHandlerUsesXRequestIDFallback(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-handler-xrequest@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-handler-xrequest@example.com", RoleRead, defaultUserStatusActive)
	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)

	firstResponse := postAdminPasswordReset(
		t,
		handler,
		svc,
		actor,
		target.ID,
		"rotated-password",
		"",
		"x-request-reset-key",
	)
	if firstResponse.Code != http.StatusOK {
		t.Fatalf("first status = %d, body = %q", firstResponse.Code, firstResponse.Body.String())
	}
	assertPasswordResetResponse(t, firstResponse, false)
	replayResponse := postAdminPasswordReset(
		t,
		handler,
		svc,
		actor,
		target.ID,
		"rotated-password",
		"x-request-reset-key",
		"",
	)
	if replayResponse.Code != http.StatusOK {
		t.Fatalf("replay status = %d, body = %q", replayResponse.Code, replayResponse.Body.String())
	}
	assertPasswordResetResponse(t, replayResponse, true)
}

func TestAdminUserPasswordHandlerAllowsLegacyRequestWithoutIdempotencyKey(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-handler@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-handler@example.com", RoleRead, defaultUserStatusActive)

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		newAuthenticatedRequest(
			t,
			svc,
			actor,
			http.MethodPost,
			"/api/admin/users/password",
			`{"userId":"`+target.ID+`","password":"rotated-password"}`,
		),
	)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
	}
	var response map[string]bool
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !response["ok"] || response["replayed"] {
		t.Fatalf("response = %#v, want ok true and replayed false", response)
	}
	if count := credentialLifecycleRequestCount(t, svc.store); count != 0 {
		t.Fatalf("legacy request should not create idempotency record; got %d", count)
	}
}

func TestAdminUserPasswordHandlerRecordsPostCommitResponseFailure(t *testing.T) {
	svc := newTestAuthService(t)
	actor := newTestUser(t, svc, "admin-handler-response-failure@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "target-handler-response-failure@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	before := credentialLifecycleTerminalFailureMetric(
		CredentialLifecycleAdminPasswordReset,
		credentialLifecycleFailureResponseEncode,
	)

	mux := http.NewServeMux()
	RegisterAdminRoutes(mux, svc)
	handler := AuthenticateRequest(svc)(mux)
	writer := newFailingResponseWriter()
	request := newAuthenticatedRequest(
		t,
		svc,
		actor,
		http.MethodPost,
		"/api/admin/users/password",
		`{"userId":"`+target.ID+`","password":"rotated-password"}`,
	)
	request.Header.Set("Idempotency-Key", "reset-key-response-failure")
	handler.ServeHTTP(writer, request)

	after := credentialLifecycleTerminalFailureMetric(
		CredentialLifecycleAdminPasswordReset,
		credentialLifecycleFailureResponseEncode,
	)
	if after != before+1 {
		t.Fatalf("response failure metric = %d, want %d", after, before+1)
	}
	if !svc.VerifyPassword(userPasswordHash(t, svc.store, target.ID), "rotated-password") {
		t.Fatalf("password hash was not committed before response failure")
	}
	assertTokenUnauthorized(t, svc, targetToken)
	if count := credentialLifecycleRequestCount(t, svc.store); count != 1 {
		t.Fatalf("credential lifecycle request count = %d, want 1 after post-commit response failure", count)
	}
	if count := auditLogCount(t, svc.store, legacyUserPasswordAuditAction); count != 1 {
		t.Fatalf("audit count = %d, want 1 after post-commit response failure", count)
	}
}

func postAdminPasswordReset(
	t *testing.T,
	handler http.Handler,
	svc *Service,
	actor *User,
	targetUserID string,
	password string,
	idempotencyKey string,
	xRequestID string,
) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	request := newAuthenticatedRequest(
		t,
		svc,
		actor,
		http.MethodPost,
		"/api/admin/users/password",
		`{"userId":"`+targetUserID+`","password":"`+password+`"}`,
	)
	if idempotencyKey != "" {
		request.Header.Set("Idempotency-Key", idempotencyKey)
	}
	if xRequestID != "" {
		request.Header.Set("X-Request-ID", xRequestID)
	}
	handler.ServeHTTP(recorder, request)
	return recorder
}

func assertPasswordResetResponse(t *testing.T, recorder *httptest.ResponseRecorder, wantReplayed bool) {
	t.Helper()
	var response map[string]bool
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !response["ok"] || response["replayed"] != wantReplayed {
		t.Fatalf("response = %#v, want ok true and replayed %v", response, wantReplayed)
	}
}

type failingResponseWriter struct {
	header http.Header
	status int
}

func newFailingResponseWriter() *failingResponseWriter {
	return &failingResponseWriter{header: http.Header{}}
}

func (w *failingResponseWriter) Header() http.Header {
	return w.header
}

func (w *failingResponseWriter) WriteHeader(status int) {
	w.status = status
}

func (w *failingResponseWriter) Write(_ []byte) (int, error) {
	return 0, errors.New("simulated response write failure")
}
