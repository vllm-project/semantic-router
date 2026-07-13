package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChangePasswordReauthenticatesRevokesOldSessionsAndIssuesFreshSession(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "change-password@example.com", RoleRead, defaultUserStatusActive)
	firstOldToken, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issue first token: %v", err)
	}
	secondOldToken, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issue second token: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/auth/password",
		strings.NewReader(`{"currentPassword":"secret-password","newPassword":"new unique password value 2026"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+firstOldToken)
	req.Header.Set(authResponseModeHeader, authResponseModeBearer)
	req.Header.Set("X-Forwarded-Proto", "https")
	handler := AuthenticateRequest(svc)(changePasswordHandler(svc))
	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	var response LoginResponse
	if decodeErr := json.NewDecoder(recorder.Body).Decode(&response); decodeErr != nil {
		t.Fatalf("decode response: %v", decodeErr)
	}
	if response.Token == "" || response.User == nil || response.User.ID != user.ID {
		t.Fatalf("response = %#v, want fresh token and current user", response)
	}
	if cookies := recorder.Result().Cookies(); len(cookies) != 0 {
		t.Fatalf("explicit bearer password change unexpectedly set cookies: %#v", cookies)
	}

	assertSessionTokenStatus(t, svc, firstOldToken, http.StatusUnauthorized)
	assertSessionTokenStatus(t, svc, secondOldToken, http.StatusUnauthorized)
	assertSessionTokenStatus(t, svc, response.Token, http.StatusNoContent)

	_, storedHash, err := svc.store.GetUserWithPasswordHash(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if svc.VerifyPassword(storedHash, "secret-password") {
		t.Fatal("old password still verifies")
	}
	if !svc.VerifyPassword(storedHash, "new unique password value 2026") {
		t.Fatal("new password does not verify")
	}
}

func TestChangePasswordWrongCurrentPasswordIs403AndKeepsSession(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "wrong-current@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issue token: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/auth/password",
		strings.NewReader(`{"currentPassword":"incorrect-password-value","newPassword":"new unique password value 2026"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	AuthenticateRequest(svc)(changePasswordHandler(svc)).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403; body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "current password is invalid") {
		t.Fatalf("body = %q, want generic current-password error", recorder.Body.String())
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestChangePasswordRejectsNFCEquivalentCurrentPassword(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		currentPassword string
		newPassword     string
	}{
		{
			name:            "exactly equal",
			currentPassword: "secret-password",
			newPassword:     "secret-password",
		},
		{
			name:            "canonically equivalent Unicode",
			currentPassword: strings.Repeat("e\u0301", minimumPasswordCodePoints),
			newPassword:     strings.Repeat("é", minimumPasswordCodePoints),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			email := "equivalent-password@example.com"
			hash, err := svc.HashPasswordForUser(email, test.currentPassword)
			if err != nil {
				t.Fatalf("HashPasswordForUser() error = %v", err)
			}
			user, err := svc.store.CreateUser(
				context.Background(),
				email,
				"Equivalent Password",
				hash,
				RoleRead,
				defaultUserStatusActive,
			)
			if err != nil {
				t.Fatalf("CreateUser() error = %v", err)
			}
			token, err := svc.issueToken(user)
			if err != nil {
				t.Fatalf("issueToken() error = %v", err)
			}
			body, err := json.Marshal(ChangePasswordRequest{
				CurrentPassword: test.currentPassword,
				NewPassword:     test.newPassword,
			})
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/api/auth/password", strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+token)
			AuthenticateRequest(svc)(changePasswordHandler(svc)).ServeHTTP(recorder, req)

			if recorder.Code != http.StatusUnprocessableEntity {
				t.Fatalf("status = %d, want 422; body=%s", recorder.Code, recorder.Body.String())
			}
			if !strings.Contains(recorder.Body.String(), "must differ") {
				t.Fatalf("body = %q, want safe unchanged-password error", recorder.Body.String())
			}
			assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
		})
	}
}

func authJSONRequest(
	t *testing.T,
	handler http.Handler,
	method string,
	path string,
	body string,
	authContext *AuthContext,
) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if authContext != nil {
		req = req.WithContext(WithAuthContext(req.Context(), *authContext))
	}
	handler.ServeHTTP(recorder, req)
	return recorder
}

func assertSessionTokenStatus(t *testing.T, svc *Service, token string, wantStatus int) {
	t.Helper()
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	handler.ServeHTTP(recorder, req)
	if recorder.Code != wantStatus {
		t.Fatalf("token status = %d, want %d", recorder.Code, wantStatus)
	}
}
