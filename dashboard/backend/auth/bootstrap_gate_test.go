package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
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

// trackHashCalls swaps hashBootstrapPassword for a counting wrapper for the
// duration of the test and returns a pointer to the live invocation count, so a
// test can assert how many bcrypt rounds a register attempt actually performed.
func trackHashCalls(t *testing.T) *int {
	t.Helper()
	calls := 0
	orig := hashBootstrapPassword
	hashBootstrapPassword = func(svc *Service, password string) (string, error) {
		calls++
		return orig(svc, password)
	}
	t.Cleanup(func() { hashBootstrapPassword = orig })
	return &calls
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

	hashCalls := trackHashCalls(t)

	rec := postRegister(svc, "second@example.com")
	if rec.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409 when an admin already exists", rec.Code)
	}
	if *hashCalls != 0 {
		t.Fatalf("bcrypt hash invoked %d times on a closed bootstrap window; want 0", *hashCalls)
	}
}

// The open window still hashes exactly once and creates the admin: the
// fast-reject must not change the success path.
func TestBootstrapRegister_OpenWindowStillHashesAndCreates(t *testing.T) {
	svc := newBootstrapService(t, false, true)

	hashCalls := trackHashCalls(t)

	rec := postRegister(svc, "admin@example.com")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if *hashCalls != 1 {
		t.Fatalf("bcrypt hash invoked %d times on the open path; want 1", *hashCalls)
	}
}

// --- Live setup-mode gate (#2795) ------------------------------------------
//
// The cases above pin setup mode to a fixed bool via SetSetupMode. The cases
// below back the gate with a real setupmode.Resolver over a config file on
// disk, which is how setupAuthRoutes wires it in production, and cover what a
// frozen bool could not express: the config changing under a running process.

const (
	setupModeConfigYAML = `version: v0.3
listeners:
  - name: http-8801
    address: 0.0.0.0
    port: 8801
    timeout: 300s
setup:
  mode: true
  state: bootstrap
`

	// activatedConfigYAML is the same file after activation strips the setup
	// block, which is what SetupActivateHandler writes to disk.
	activatedConfigYAML = `version: v0.3
listeners:
  - name: http-8801
    address: 0.0.0.0
    port: 8801
    timeout: 300s
`

	malformedConfigYAML = "::: not yaml :::\n"
)

func writeRouterConfig(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write router config %s: %v", path, err)
	}
}

// newRouterConfig writes a router config fixture to a fresh temp dir.
func newRouterConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	writeRouterConfig(t, path, content)
	return path
}

// newResolverBackedService builds a bootstrap service whose setup-mode gate
// reads a real config file through the canonical resolver, mirroring
// setupAuthRoutes. legacyFlag is the --setup-mode / DASHBOARD_SETUP_MODE value,
// which is recorded but must never decide the gate.
func newResolverBackedService(t *testing.T, configPath string, legacyFlag bool) (*Service, *setupmode.Resolver) {
	t.Helper()
	svc := newTestAuthService(t)
	svc.SetAllowOpenBootstrap(false)
	resolver := setupmode.New(configPath, legacyFlag)
	svc.SetSetupModeFunc(resolver.Active)
	return svc, resolver
}

// TestBootstrapGate_ClosesWhenActivationClearsSetupMode_WithoutRestart is the
// test that proves #2795 is fixed.
//
// Activation rewrites the router config without the setup block and does not
// restart the dashboard (it restarts only router and envoy). With setup mode
// stored as a bool captured at startup, the unauthenticated bootstrap endpoint
// stayed armed until someone restarted the container. Reading the config at
// call time closes it the moment the file stops declaring setup mode, in the
// same process, with no restart. The legacy flag stays true throughout, exactly
// as it does in a real deployment launched by the CLI.
func TestBootstrapGate_ClosesWhenActivationClearsSetupMode_WithoutRestart(t *testing.T) {
	configPath := newRouterConfig(t, setupModeConfigYAML)
	svc, resolver := newResolverBackedService(t, configPath, true)

	if !svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = false while the config declares setup mode; want true")
	}
	if !canRegister(t, svc) {
		t.Fatal("canRegister = false during first run with no users; want true")
	}

	// What SetupActivateHandler does to disk: the setup block is stripped.
	writeRouterConfig(t, configPath, activatedConfigYAML)
	resolver.Invalidate()

	if svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = true after activation cleared setup mode; the gate is still frozen at startup")
	}
	if canRegister(t, svc) {
		t.Fatal("canRegister = true after activation cleared setup mode; the open bootstrap door did not close")
	}
	if rec := postRegister(svc, "attacker@example.com"); rec.Code != http.StatusForbidden {
		t.Fatalf("register status = %d after activation, want 403; body=%s", rec.Code, rec.Body.String())
	}
	if n, _ := svc.store.CountUsers(context.Background()); n != 0 {
		t.Fatalf("created %d users through a closed bootstrap window; want 0", n)
	}
}

// Scenario 2 from the issue, and the headline security case: a stale
// DASHBOARD_SETUP_MODE=true against a fully activated config. The UI shows
// nothing because /api/setup/state reads the file, so this door was invisible.
// The config file is canonical, so the flag alone must not open it.
func TestBootstrapGate_StaleLegacyFlagCannotOpenAgainstActivatedConfig(t *testing.T) {
	configPath := newRouterConfig(t, activatedConfigYAML)
	svc, _ := newResolverBackedService(t, configPath, true)

	if svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = true from the legacy flag alone; want false, the config file is canonical")
	}
	if canRegister(t, svc) {
		t.Fatal("canRegister = true from the legacy flag alone; want false")
	}
	if rec := postRegister(svc, "attacker@example.com"); rec.Code != http.StatusForbidden {
		t.Fatalf("register status = %d, want 403; body=%s", rec.Code, rec.Body.String())
	}
}

// Scenario 3 from the issue: a config in setup mode with no environment value,
// which previously left first run locked out - the UI pinned you to /setup and
// the backend refused the registration it demanded.
func TestBootstrapGate_ConfigInSetupModeOpensWithoutLegacyFlag(t *testing.T) {
	configPath := newRouterConfig(t, setupModeConfigYAML)
	svc, _ := newResolverBackedService(t, configPath, false)

	if !svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = false while the config declares setup mode; first run is locked out")
	}
	if !canRegister(t, svc) {
		t.Fatal("canRegister = false while the config declares setup mode and no users exist; want true")
	}
	if rec := postRegister(svc, "admin@example.com"); rec.Code != http.StatusOK {
		t.Fatalf("register status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

// allowOpenBootstrap is a deliberate operator escape hatch with no config-file
// counterpart. It must keep working on its own, independently of what the
// resolver says.
func TestBootstrapGate_AllowOpenBootstrapUnaffectedByResolver(t *testing.T) {
	configPath := newRouterConfig(t, activatedConfigYAML)
	svc, _ := newResolverBackedService(t, configPath, false)
	svc.SetAllowOpenBootstrap(true)

	if !svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = false with allowOpenBootstrap set; the resolver must not override the operator flag")
	}
	if !canRegister(t, svc) {
		t.Fatal("canRegister = false with allowOpenBootstrap set and no users; want true")
	}
}

// A Service with no setup-mode source installed must keep the endpoint shut.
// NewService does not install one, so this is the state of any Service built
// without wiring - including one built when the resolver was unavailable.
func TestBootstrapGate_NilSetupModeSourceFailsClosed(t *testing.T) {
	svc := newTestAuthService(t)

	if svc.OpenBootstrapEnabled() {
		t.Fatal("OpenBootstrapEnabled = true with no setup-mode source installed; want false (fail closed)")
	}
	if canRegister(t, svc) {
		t.Fatal("canRegister = true with no setup-mode source installed; want false")
	}
}

// The error path must not reopen the door. An unreadable or unparseable config
// resolves to "not in setup mode" even with the legacy flag set; falling back
// to the flag here would reintroduce the bug through the error path.
func TestBootstrapGate_MalformedConfigFailsClosedDespiteLegacyFlag(t *testing.T) {
	t.Run("malformed config", func(t *testing.T) {
		configPath := newRouterConfig(t, malformedConfigYAML)
		svc, _ := newResolverBackedService(t, configPath, true)

		if svc.OpenBootstrapEnabled() {
			t.Fatal("OpenBootstrapEnabled = true for a malformed config with the legacy flag set; want false")
		}
		if canRegister(t, svc) {
			t.Fatal("canRegister = true for a malformed config; want false")
		}
	})

	t.Run("missing config", func(t *testing.T) {
		configPath := filepath.Join(t.TempDir(), "does-not-exist.yaml")
		svc, _ := newResolverBackedService(t, configPath, true)

		if svc.OpenBootstrapEnabled() {
			t.Fatal("OpenBootstrapEnabled = true for a missing config with the legacy flag set; want false")
		}
		if canRegister(t, svc) {
			t.Fatal("canRegister = true for a missing config; want false")
		}
	})
}

// The gate is consulted from an unauthenticated endpoint, so every request
// reads it concurrently while the activate handler may be rewriting the config.
// This asserts race-freedom under -race, not a particular value: os.WriteFile
// truncates before writing, so a reader may legitimately observe the file
// mid-write.
func TestBootstrapGate_ConcurrentGateReadsDuringConfigRewrite(t *testing.T) {
	configPath := newRouterConfig(t, setupModeConfigYAML)
	svc, resolver := newResolverBackedService(t, configPath, true)

	const (
		readers       = 50
		readsPerVisit = 40
	)

	stop := make(chan struct{})
	var writer, readerGroup sync.WaitGroup

	writer.Add(1)
	go func() {
		defer writer.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			content := setupModeConfigYAML
			if i%2 == 1 {
				content = activatedConfigYAML
			}
			if err := os.WriteFile(configPath, []byte(content), 0o600); err != nil {
				t.Errorf("write router config: %v", err)
				return
			}
			resolver.Invalidate()
		}
	}()

	for i := 0; i < readers; i++ {
		readerGroup.Add(1)
		go func() {
			defer readerGroup.Done()
			for j := 0; j < readsPerVisit; j++ {
				_ = svc.OpenBootstrapEnabled()
			}
		}()
	}

	readerGroup.Wait()
	close(stop)
	writer.Wait()
}
