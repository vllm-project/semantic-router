package auth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestLiveAuthorizationClosesLongLivedRequestImmediatelyOnLogout(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "live-logout@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	started := make(chan context.Context, 1)
	finished := make(chan struct{})
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		started <- r.Context()
		<-r.Context().Done()
		close(finished)
	}))
	request := httptest.NewRequest(http.MethodGet, "https://dashboard.example.com/api/openclaw/rooms/room/ws", nil)
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Origin", "https://dashboard.example.com")
	request.Header.Set("X-Forwarded-Proto", "https")

	requestDone := make(chan struct{})
	go func() {
		handler.ServeHTTP(httptest.NewRecorder(), request)
		close(requestDone)
	}()

	var liveContext context.Context
	select {
	case liveContext = <-started:
	case <-time.After(time.Second):
		t.Fatal("long-lived request did not start")
	}
	if err := svc.RevokeToken(context.Background(), token); err != nil {
		t.Fatalf("RevokeToken() error = %v", err)
	}
	select {
	case <-liveContext.Done():
		if !errors.Is(context.Cause(liveContext), ErrLiveAuthorizationInvalid) {
			t.Fatalf("context cause = %v, want live authorization invalid", context.Cause(liveContext))
		}
	case <-time.After(time.Second):
		t.Fatal("logout did not cancel long-lived request")
	}
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("handler did not observe authorization cancellation")
	}
	<-requestDone
}

func TestLiveAuthorizationClosesOnPasswordLifecycleAndUserMutation(t *testing.T) {
	// Password hashing is intentionally CPU-expensive; keep the cases serial.
	tests := []struct {
		name   string
		mutate func(*Service, *User) error
	}{
		{
			name: "self password change",
			mutate: func(svc *Service, user *User) error {
				_, _, err := svc.ChangePassword(
					context.Background(),
					user.ID,
					"secret-password",
					"replacement password for live session",
				)
				return err
			},
		},
		{
			name: "admin password reset",
			mutate: func(svc *Service, user *User) error {
				return svc.ResetPassword(context.Background(), user.ID, "admin reset live session password")
			},
		},
		{
			name: "role status or deletion notification",
			mutate: func(svc *Service, user *User) error {
				svc.invalidateUserAuthorization(user.ID)
				return nil
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			email := strings.ReplaceAll(test.name, " ", "-") + "@example.com"
			user := newTestUser(t, svc, email, RoleWrite, defaultUserStatusActive)
			token, err := svc.issueToken(user)
			if err != nil {
				t.Fatalf("issueToken() error = %v", err)
			}
			claims, err := svc.ParseToken(token)
			if err != nil {
				t.Fatalf("ParseToken() error = %v", err)
			}
			ctx, stop, err := svc.monitorAuthorization(context.Background(), claims, PermOpenClawUse)
			if err != nil {
				t.Fatalf("monitorAuthorization() error = %v", err)
			}
			defer stop()

			if err := test.mutate(svc, user); err != nil {
				t.Fatalf("mutation error = %v", err)
			}
			select {
			case <-ctx.Done():
			case <-time.After(time.Second):
				t.Fatal("credential lifecycle mutation did not cancel live authorization")
			}
		})
	}
}

func TestLiveAuthorizationPeriodicBackstopDetectsPermissionLoss(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.liveAuthorization.recheckInterval = 10 * time.Millisecond
	user := newTestUser(t, svc, "live-permission-loss@example.com", RoleWrite, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	ctx, stop, err := svc.monitorAuthorization(context.Background(), claims, PermOpenClawUse)
	if err != nil {
		t.Fatalf("monitorAuthorization() error = %v", err)
	}
	defer stop()

	// Bypass service notifications to model an out-of-process DB mutation. The
	// periodic DB check is the required backstop for this case.
	if _, err := svc.store.UpdateUserRoleOrStatus(context.Background(), user.ID, RoleRead, ""); err != nil {
		t.Fatalf("UpdateUserRoleOrStatus() error = %v", err)
	}
	select {
	case <-ctx.Done():
		if !errors.Is(context.Cause(ctx), ErrLivePermissionDenied) {
			t.Fatalf("context cause = %v, want permission denied", context.Cause(ctx))
		}
	case <-time.After(time.Second):
		t.Fatal("periodic authorization backstop did not detect permission loss")
	}
}

func TestLiveAuthorizationRegistryEnforcesAndReleasesAdmissionLimits(t *testing.T) {
	t.Parallel()

	registry := newLiveAuthorizationRegistry()
	watchers := make([]*authorizationWatcher, 0, maxLiveAuthorizationWatchersPerSession)
	for index := 0; index < maxLiveAuthorizationWatchersPerSession; index++ {
		watcher := &authorizationWatcher{
			userID:    "bounded-user",
			sessionID: "bounded-session",
			cancel:    func(error) {},
		}
		if err := registry.register(watcher); err != nil {
			t.Fatalf("register watcher %d: %v", index, err)
		}
		watchers = append(watchers, watcher)
	}
	if err := registry.register(&authorizationWatcher{
		userID:    "bounded-user",
		sessionID: "bounded-session",
		cancel:    func(error) {},
	}); !errors.Is(err, ErrLiveAuthorizationCapacity) {
		t.Fatalf("extra session watcher error = %v, want capacity", err)
	}

	registry.unregister(watchers[0].id)
	if err := registry.register(&authorizationWatcher{
		userID:    "bounded-user",
		sessionID: "bounded-session",
		cancel:    func(error) {},
	}); err != nil {
		t.Fatalf("register after release: %v", err)
	}
}

func TestLiveAuthorizationRegistryEnforcesUserAndGlobalLimits(t *testing.T) {
	t.Parallel()

	userRegistry := newLiveAuthorizationRegistry()
	for index := 0; index < maxLiveAuthorizationWatchersPerUser; index++ {
		if err := userRegistry.register(&authorizationWatcher{
			userID:    "bounded-user",
			sessionID: fmt.Sprintf("session-%d", index),
			cancel:    func(error) {},
		}); err != nil {
			t.Fatalf("register user watcher %d: %v", index, err)
		}
	}
	if err := userRegistry.register(&authorizationWatcher{
		userID:    "bounded-user",
		sessionID: "overflow-session",
		cancel:    func(error) {},
	}); !errors.Is(err, ErrLiveAuthorizationCapacity) {
		t.Fatalf("extra user watcher error = %v, want capacity", err)
	}

	globalRegistry := newLiveAuthorizationRegistry()
	for index := 0; index < maxLiveAuthorizationWatchers; index++ {
		if err := globalRegistry.register(&authorizationWatcher{
			userID:    fmt.Sprintf("user-%d", index),
			sessionID: fmt.Sprintf("session-%d", index),
			cancel:    func(error) {},
		}); err != nil {
			t.Fatalf("register global watcher %d: %v", index, err)
		}
	}
	if err := globalRegistry.register(&authorizationWatcher{
		userID:    "overflow-user",
		sessionID: "overflow-session",
		cancel:    func(error) {},
	}); !errors.Is(err, ErrLiveAuthorizationCapacity) {
		t.Fatalf("extra global watcher error = %v, want capacity", err)
	}
}

func TestLiveAuthorizationCapacityFailsClosedBeforeStreamingHandler(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "live-capacity@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	for index := 0; index < maxLiveAuthorizationWatchersPerSession; index++ {
		if err := svc.liveAuthorization.register(&authorizationWatcher{
			userID:    user.ID,
			sessionID: claims.ID,
			cancel:    func(error) {},
		}); err != nil {
			t.Fatalf("fill watcher %d: %v", index, err)
		}
	}

	handlerCalled := false
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		handlerCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))
	request := httptest.NewRequest(
		http.MethodGet,
		"/api/openclaw/rooms/room-a/stream",
		nil,
	)
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Accept", "text/event-stream")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusServiceUnavailable)
	}
	if handlerCalled {
		t.Fatal("streaming handler ran after live authorization capacity was exhausted")
	}
	if recorder.Header().Get("Retry-After") != "1" {
		t.Fatalf("Retry-After = %q, want 1", recorder.Header().Get("Retry-After"))
	}
}

func TestAdminUserMutationsInvalidateTargetLiveAuthorization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		method     string
		body       string
		wantStatus int
	}{
		{name: "role change", method: http.MethodPatch, body: `{"role":"read"}`, wantStatus: http.StatusOK},
		{name: "deactivate", method: http.MethodPatch, body: `{"status":"inactive"}`, wantStatus: http.StatusOK},
		{name: "delete", method: http.MethodDelete, wantStatus: http.StatusNoContent},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			svc := newTestAuthService(t)
			caseName := strings.ReplaceAll(test.name, " ", "-")
			admin := newTestUser(t, svc, "live-admin-"+caseName+"@example.com", RoleAdmin, defaultUserStatusActive)
			target := newTestUser(t, svc, "live-target-"+caseName+"@example.com", RoleWrite, defaultUserStatusActive)
			targetToken, err := svc.issueToken(target)
			if err != nil {
				t.Fatalf("issue target token: %v", err)
			}
			claims, err := svc.ParseToken(targetToken)
			if err != nil {
				t.Fatalf("parse target token: %v", err)
			}
			liveContext, stop, err := svc.monitorAuthorization(context.Background(), claims, PermConfigRead)
			if err != nil {
				t.Fatalf("monitor target authorization: %v", err)
			}
			defer stop()

			request := newAuthenticatedRequest(
				t,
				svc,
				admin,
				test.method,
				"/api/admin/users/"+target.ID,
				test.body,
			)
			recorder := httptest.NewRecorder()
			AuthenticateRequest(svc)(adminUserItemHandler(svc)).ServeHTTP(recorder, request)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, test.wantStatus, recorder.Body.String())
			}
			select {
			case <-liveContext.Done():
			case <-time.After(time.Second):
				t.Fatal("admin mutation did not immediately cancel target live authorization")
			}
		})
	}
}

func TestLiveAuthorizationExpiresAtJWTDeadline(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "live-expiry@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	claims.ExpiresAt = jwt.NewNumericDate(time.Now().Add(30 * time.Millisecond))
	ctx, stop, err := svc.monitorAuthorization(context.Background(), claims, PermConfigRead)
	if err != nil {
		t.Fatalf("monitorAuthorization() error = %v", err)
	}
	defer stop()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("JWT expiry did not cancel live authorization")
	}
}

func TestCredentialTransportRejectsQueryAndAmbiguityBeforeHandler(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "credential-transport@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	tests := []struct {
		name         string
		path         string
		headers      []string
		cookieValues []string
	}{
		{name: "query credential", path: "/api/status?authToken=" + token, cookieValues: []string{token}},
		{name: "case folded query credential", path: "/api/status?AUTHTOKEN=" + token, cookieValues: []string{token}},
		{name: "percent encoded query key", path: "/api/status?auth%54oken=" + token, cookieValues: []string{token}},
		{name: "malformed query value", path: "/api/status?authToken=%zz", cookieValues: []string{token}},
		{name: "bearer and cookie", path: "/api/status", headers: []string{"Bearer " + token}, cookieValues: []string{token}},
		{name: "malformed authorization and cookie", path: "/api/status", headers: []string{"Basic ignored"}, cookieValues: []string{token}},
		{name: "duplicate authorization", path: "/api/status", headers: []string{"Bearer " + token, "Bearer other"}},
		{name: "duplicate session cookie", path: "/api/status", cookieValues: []string{token, "other"}},
		{name: "empty then valid session cookie", path: "/api/status", cookieValues: []string{"", token}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nextCalled := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusNoContent)
			}))
			req := httptest.NewRequest(http.MethodGet, test.path, nil)
			for _, header := range test.headers {
				req.Header.Add("Authorization", header)
			}
			for _, value := range test.cookieValues {
				req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: value})
			}
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", recorder.Code)
			}
			if nextCalled {
				t.Fatal("invalid credential transport reached protected handler")
			}
			if strings.Contains(recorder.Body.String(), token) {
				t.Fatal("error response reflected credential")
			}
		})
	}
}

func TestCredentialQueryIsRejectedOnPublicRoutes(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	nextCalled := false
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	request := httptest.NewRequest(http.MethodGet, "/api/auth/login?authToken=must-not-enter-logs", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", recorder.Code)
	}
	if nextCalled {
		t.Fatal("public route accepted a query credential")
	}
	if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", got)
	}
}
