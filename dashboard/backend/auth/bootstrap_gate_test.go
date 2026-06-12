package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func newBootstrapService(t *testing.T, allowOpen bool, setupMode ...bool) *Service {
	t.Helper()
	svc := newTestAuthService(t)
	svc.SetAllowOpenBootstrap(allowOpen)
	if len(setupMode) > 0 {
		svc.SetSetupMode(setupMode[0])
	}
	return svc
}

func canRegister(t *testing.T, svc *Service) bool {
	t.Helper()
	rec := httptest.NewRecorder()
	bootstrapCanRegisterHandler(svc)(rec, httptest.NewRequest(http.MethodGet, "/api/auth/bootstrap/can-register", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("can-register status = %d, want 200", rec.Code)
	}
	var resp BootstrapStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode can-register: %v", err)
	}
	return resp.CanRegister
}

func postRegister(svc *Service, email string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	body := fmt.Sprintf(`{"email":%q,"password":"secret-password","name":"Admin"}`, email)
	bootstrapRegisterHandler(svc)(rec, httptest.NewRequest(http.MethodPost, "/api/auth/bootstrap/register", strings.NewReader(body)))
	return rec
}

// With open bootstrap disabled (the default), can-register must report false even
// when no users exist - both to disable the path and to avoid leaking to an
// unauthenticated caller that the instance is freshly deployed and claimable.
func TestBootstrapCanRegister_DisabledByDefaultReportsFalse(t *testing.T) {
	svc := newBootstrapService(t, false)
	if canRegister(t, svc) {
		t.Fatal("canRegister = true with open bootstrap disabled; want false")
	}
}

func TestBootstrapCanRegister_SetupModeAllowsWhenNoUsers(t *testing.T) {
	svc := newBootstrapService(t, false, true)
	if !canRegister(t, svc) {
		t.Fatal("canRegister = false with setup mode and no users; want true")
	}
}

func TestBootstrapCanRegister_SetupModeClosedAfterAdminExists(t *testing.T) {
	svc := newBootstrapService(t, false, true)
	newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	if canRegister(t, svc) {
		t.Fatal("canRegister = true with setup mode after an admin exists; want false")
	}
}

func TestBootstrapRegister_SetupModeCreatesFirstAdmin(t *testing.T) {
	svc := newBootstrapService(t, false, true)
	rec := postRegister(svc, "admin@example.com")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if n, _ := svc.store.CountUsers(context.Background()); n != 1 {
		t.Fatalf("user count = %d, want 1", n)
	}
}

func TestBootstrapCanRegister_EnabledTracksUserCount(t *testing.T) {
	svc := newBootstrapService(t, true)
	if !canRegister(t, svc) {
		t.Fatal("canRegister = false with open bootstrap enabled and no users; want true")
	}
	newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	if canRegister(t, svc) {
		t.Fatal("canRegister = true after an admin exists; want false")
	}
}

// The open register endpoint must be closed by default and must not create a user.
func TestBootstrapRegister_DisabledByDefaultForbidden(t *testing.T) {
	svc := newBootstrapService(t, false)
	rec := postRegister(svc, "admin@example.com")
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403 when open bootstrap disabled", rec.Code)
	}
	if n, _ := svc.store.CountUsers(context.Background()); n != 0 {
		t.Fatalf("created %d users while disabled; want 0", n)
	}
}

func TestBootstrapRegister_EnabledCreatesFirstAdmin(t *testing.T) {
	svc := newBootstrapService(t, true)
	rec := postRegister(svc, "admin@example.com")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if n, _ := svc.store.CountUsers(context.Background()); n != 1 {
		t.Fatalf("user count = %d, want 1", n)
	}
}

func TestBootstrapRegister_SecondAttemptConflict(t *testing.T) {
	svc := newBootstrapService(t, true)
	newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")
	rec := postRegister(svc, "second@example.com")
	if rec.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409 when an admin already exists", rec.Code)
	}
}

// The race fix: concurrent register requests must produce exactly one admin.
func TestBootstrapRegister_ConcurrentCreatesExactlyOneAdmin(t *testing.T) {
	svc := newBootstrapService(t, true)
	const n = 16
	var wg sync.WaitGroup
	codes := make([]int, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			codes[i] = postRegister(svc, fmt.Sprintf("user%d@example.com", i)).Code
		}(i)
	}
	wg.Wait()

	if count, _ := svc.store.CountUsers(context.Background()); count != 1 {
		t.Fatalf("concurrent bootstrap created %d users; want exactly 1 (race not closed)", count)
	}
	ok := 0
	for _, c := range codes {
		if c == http.StatusOK {
			ok++
		}
	}
	if ok != 1 {
		t.Fatalf("got %d successful registrations; want exactly 1", ok)
	}
}

// Once an admin exists, a register attempt must be rejected before any bcrypt
// work happens: with the endpoint enabled (setup mode or open bootstrap), an
// unauthenticated caller must not be able to burn a cost-12 bcrypt round per
// request against an already-consumed bootstrap window.
func TestBootstrapRegister_ClosedWindowRejectsBeforeHashing(t *testing.T) {
	svc := newBootstrapService(t, false, true)
	newTestUser(t, svc, "admin@example.com", RoleAdmin, "active")

	hashCalls := 0
	orig := hashBootstrapPassword
	hashBootstrapPassword = func(svc *Service, password string) (string, error) {
		hashCalls++
		return orig(svc, password)
	}
	t.Cleanup(func() { hashBootstrapPassword = orig })

	rec := postRegister(svc, "second@example.com")
	if rec.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409 when an admin already exists", rec.Code)
	}
	if hashCalls != 0 {
		t.Fatalf("bcrypt hash invoked %d times on a closed bootstrap window; want 0", hashCalls)
	}
}

// The open window still hashes exactly once and creates the admin: the
// fast-reject must not change the success path.
func TestBootstrapRegister_OpenWindowStillHashesAndCreates(t *testing.T) {
	svc := newBootstrapService(t, false, true)

	hashCalls := 0
	orig := hashBootstrapPassword
	hashBootstrapPassword = func(svc *Service, password string) (string, error) {
		hashCalls++
		return orig(svc, password)
	}
	t.Cleanup(func() { hashBootstrapPassword = orig })

	rec := postRegister(svc, "admin@example.com")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if hashCalls != 1 {
		t.Fatalf("bcrypt hash invoked %d times on the open path; want 1", hashCalls)
	}
}
