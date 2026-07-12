package auth

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
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
	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value != response.Token || !cookie.HttpOnly || !cookie.Secure {
		t.Fatalf("fresh cookie = %#v, want secure HttpOnly response token", cookie)
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

func TestChangePasswordCurrentVerificationIsRateLimited(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.passwordChangeLimiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  100,
		BaseDelay:               3 * time.Second,
		MaxDelay:                3 * time.Second,
	})
	user := newTestUser(t, svc, "change-rate-limit@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	verifyCalls := 0
	originalVerify := svc.verify
	svc.verify = func(hash, password string) bool {
		verifyCalls++
		return originalVerify(hash, password)
	}

	request := func() *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(
			http.MethodPost,
			"/api/auth/password",
			strings.NewReader(`{"currentPassword":"incorrect-password-value","newPassword":"new unique password value 2026"}`),
		)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+token)
		AuthenticateRequest(svc)(changePasswordHandler(svc)).ServeHTTP(recorder, req)
		return recorder
	}

	if first := request(); first.Code != http.StatusForbidden {
		t.Fatalf("first status = %d, want 403; body=%s", first.Code, first.Body.String())
	}
	second := request()
	if second.Code != http.StatusTooManyRequests {
		t.Fatalf("second status = %d, want 429; body=%s", second.Code, second.Body.String())
	}
	if second.Header().Get("Retry-After") != "3" {
		t.Fatalf("Retry-After = %q, want 3", second.Header().Get("Retry-After"))
	}
	if verifyCalls != 1 {
		t.Fatalf("password verifier calls = %d, want 1", verifyCalls)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestSuccessfulCurrentPasswordClearsAccountFailuresBeforePolicyValidation(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.passwordChangeLimiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 2,
		SourceFailureThreshold:  100,
		BaseDelay:               time.Minute,
		MaxDelay:                time.Minute,
	})
	user := newTestUser(t, svc, "reauth-reset@example.com", RoleRead, defaultUserStatusActive)

	_, _, err := svc.ChangePassword(
		context.Background(),
		user.ID,
		"incorrect-password-value",
		"new unique password value 2026",
	)
	if !errors.Is(err, ErrCurrentPasswordFailed) {
		t.Fatalf("first change error = %v, want ErrCurrentPasswordFailed", err)
	}
	_, _, err = svc.ChangePassword(
		context.Background(),
		user.ID,
		"secret-password",
		"short",
	)
	var policyErr *PasswordPolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyTooShort {
		t.Fatalf("reauthenticated change error = %#v, want too-short policy error", err)
	}
	_, _, err = svc.ChangePassword(
		context.Background(),
		user.ID,
		"incorrect-password-value",
		"new unique password value 2026",
	)
	if !errors.Is(err, ErrCurrentPasswordFailed) {
		t.Fatalf("post-reauth change error = %v, want ErrCurrentPasswordFailed", err)
	}
	if retry := svc.passwordChangeLimiter.RetryAfter(user.Email, ""); retry != 0 {
		t.Fatalf("retry after one post-reauth failure = %v, want 0", retry)
	}
}

func TestGlobalLoginBudgetCannotBlockAuthenticatedPasswordChange(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "management-reserve@example.com", RoleRead, defaultUserStatusActive)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 100,
		SourceFailureThreshold:  100,
		GlobalAttemptBurst:      1,
		GlobalRefillInterval:    time.Hour,
	})
	attacker, retry := svc.limiter.Reserve("random-attacker@example.com", "")
	if attacker == nil || retry != 0 {
		t.Fatalf("attacker Reserve() = (%#v, %v), want initial admission", attacker, retry)
	}
	attacker.Fail()
	if attempt, retry := svc.limiter.Reserve("another-random@example.com", ""); attempt != nil || retry <= 0 {
		t.Fatalf("global login budget was not exhausted: (%#v, %v)", attempt, retry)
	}

	token, _, err := svc.ChangePassword(
		context.Background(),
		user.ID,
		"secret-password",
		"management path replacement password 2026",
	)
	if err != nil {
		t.Fatalf("ChangePassword() error under exhausted login budget = %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestAnonymousLoginEvidenceCannotBlockEmergencyPasswordChange(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "emergency-rotation@example.com", RoleRead, defaultUserStatusActive)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  1,
		MaxAccounts:             1,
		MaxSources:              1,
		GlobalAttemptBurst:      10,
		BaseDelay:               time.Minute,
		MaxDelay:                time.Minute,
	})
	source := "198.51.100.20"
	svc.limiter.RecordFailure(user.Email, source)
	svc.limiter.RecordFailure("overflow-attacker@example.com", "198.51.100.21")
	if retry := svc.limiter.RetryAfter(user.Email, source); retry <= 0 {
		t.Fatal("anonymous login evidence did not block the login path fixture")
	}

	token, _, err := svc.ChangePasswordWithSource(
		context.Background(),
		user.ID,
		"secret-password",
		"emergency rotation replacement 2026",
		source,
	)
	if err != nil {
		t.Fatalf("ChangePasswordWithSource() inherited anonymous limiter state: %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestAdminPasswordResetRevokesAllTargetSessions(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "reset-admin@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "reset-target@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	adminToken, err := svc.issueToken(admin)
	if err != nil {
		t.Fatalf("issue admin token: %v", err)
	}
	adminClaims, err := svc.ParseToken(adminToken)
	if err != nil {
		t.Fatalf("parse admin token: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/admin/users/password",
		strings.NewReader(`{"userId":"`+target.ID+`","password":"admin reset password value"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(WithAuthContext(req.Context(), AuthContext{
		UserID:    admin.ID,
		SessionID: adminClaims.ID,
		Perms:     map[string]bool{PermUsersManage: true},
	}))
	adminUserPasswordHandler(svc).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	assertSessionTokenStatus(t, svc, targetToken, http.StatusUnauthorized)
	_, hash, err := svc.store.GetUserWithPasswordHash(context.Background(), target.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if !svc.VerifyPassword(hash, "admin reset password value") {
		t.Fatal("reset password does not verify")
	}
}

func TestPasswordChangeCompareAndSwapRejectsStaleHash(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "password-cas@example.com", RoleRead, defaultUserStatusActive)
	passwordState, err := svc.store.getUserPasswordStateByID(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("getUserPasswordStateByID() error = %v", err)
	}
	oldHash := passwordState.hash
	firstHash, err := svc.HashPasswordForUser(user.Email, "first replacement password")
	if err != nil {
		t.Fatalf("first hash: %v", err)
	}
	secondHash, err := svc.HashPasswordForUser(user.Email, "second replacement password")
	if err != nil {
		t.Fatalf("second hash: %v", err)
	}
	firstToken, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepare first token: %v", err)
	}
	if changeErr := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		firstHash,
		firstToken,
	); changeErr != nil {
		t.Fatalf("first change: %v", changeErr)
	}
	staleToken, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepare stale token: %v", err)
	}
	if err := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		secondHash,
		staleToken,
	); !errors.Is(err, ErrPasswordChanged) {
		t.Fatalf("stale change error = %v, want ErrPasswordChanged", err)
	}
	assertSessionTokenStatus(t, svc, firstToken.signed, http.StatusNoContent)
	assertSessionTokenStatus(t, svc, staleToken.signed, http.StatusUnauthorized)
}

func TestPasswordChangeCompareAndSwapRejectsInactiveUser(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "inactive-password-cas@example.com", RoleRead, defaultUserStatusActive)
	passwordState, err := svc.store.getUserPasswordStateByID(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("getUserPasswordStateByID() error = %v", err)
	}
	oldHash := passwordState.hash
	newHash, err := svc.HashPasswordForUser(user.Email, "inactive replacement password")
	if err != nil {
		t.Fatalf("HashPasswordForUser() error = %v", err)
	}
	issued, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepareToken() error = %v", err)
	}
	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		user.ID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate user: %v", err)
	}
	if err := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		newHash,
		issued,
	); !errors.Is(err, ErrPasswordChanged) {
		t.Fatalf("inactive change error = %v, want ErrPasswordChanged", err)
	}
	if sessionExists(t, svc.store, issued.sessionID) {
		t.Fatal("inactive password CAS inserted a session")
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
