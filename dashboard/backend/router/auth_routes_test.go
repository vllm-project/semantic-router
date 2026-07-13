package router

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// When the auth service fails to initialize (authSvc == nil), wrapWithAuth must
// fail CLOSED: routes that require authentication must be denied with 503 and
// their backend handlers must never run, while public/static routes stay
// reachable so the dashboard can surface the misconfiguration. This guards
// against the fail-open regression where the entire control plane was served
// unauthenticated whenever the auth store could not be opened.
func TestWrapWithAuthFailsClosedWhenAuthUnavailable(t *testing.T) {
	t.Parallel()

	protectedHit := false
	publicHit := false

	mux := http.NewServeMux()
	mux.HandleFunc("/api/router/config", func(w http.ResponseWriter, _ *http.Request) {
		protectedHit = true
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		publicHit = true
		w.WriteHeader(http.StatusOK)
	})

	handler := wrapWithAuth(mux, nil) // nil => auth store failed to initialize

	t.Run("protected route denied without executing handler", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/router/config", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("protected route status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
		if protectedHit {
			t.Fatal("protected handler executed while auth was unavailable (fail-open regression)")
		}
	})

	t.Run("public/static route still served", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("public route status = %d, want %d", rec.Code, http.StatusOK)
		}
		if !publicHit {
			t.Fatal("public handler did not run; fail-closed guard over-blocked")
		}
	})

	t.Run("protected admin route denied", func(t *testing.T) {
		mux.HandleFunc("/api/admin/users", func(w http.ResponseWriter, _ *http.Request) {
			t.Error("admin handler executed while auth was unavailable")
			w.WriteHeader(http.StatusOK)
		})
		req := httptest.NewRequest(http.MethodGet, "/api/admin/users", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("admin route status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
	})
}

func TestSetupAuthRoutesFailsClosedForInvalidPasswordBlocklist(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	cfg := &config.Config{
		AuthDBPath:            filepath.Join(t.TempDir(), "auth.db"),
		PasswordBlocklistPath: filepath.Join(t.TempDir(), "missing-password-blocklist.txt"),
	}
	svc, err := setupAuthRoutes(mux, cfg)
	if svc != nil {
		t.Fatal("setupAuthRoutes() returned a service for an invalid blocklist")
	}
	if err == nil {
		t.Fatal("setupAuthRoutes() accepted an invalid blocklist")
	}
	server, setupErr := Setup(cfg)
	if server != nil || setupErr == nil {
		t.Fatalf("Setup() = (%#v, %v), want blocklist startup failure", server, setupErr)
	}
}

func TestSetupAuthRoutesFailsClosedForWeakConfiguredJWTSecret(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	cfg := &config.Config{
		AuthDBPath: filepath.Join(t.TempDir(), "auth.db"),
		JWTSecret:  "too-short",
	}
	svc, err := setupAuthRoutes(mux, cfg)
	if svc != nil {
		t.Fatal("setupAuthRoutes() returned a service for a weak JWT secret")
	}
	if err == nil {
		t.Fatal("setupAuthRoutes() accepted a weak JWT secret")
	}
	if strings.Contains(err.Error(), cfg.JWTSecret) {
		t.Fatalf("setup error exposed configured JWT secret: %q", err)
	}

	server, setupErr := Setup(cfg)
	if server != nil || setupErr == nil {
		t.Fatalf("Setup() = (%#v, %v), want startup failure", server, setupErr)
	}
}

func TestSetupAuthRoutesExposesOnlyVerifiedPasswordPolicyMetadata(t *testing.T) {
	t.Parallel()

	blocklistPath, digest := writeRouterProductionBlocklist(t)
	mux := http.NewServeMux()
	cfg := &config.Config{
		AuthDBPath:              filepath.Join(t.TempDir(), "auth.db"),
		SecurityProfile:         config.DashboardSecurityProfileProduction,
		PasswordBlocklistPath:   blocklistPath,
		PasswordBlocklistSHA256: digest,
	}
	svc, err := setupAuthRoutes(mux, cfg)
	if err != nil {
		t.Fatalf("setupAuthRoutes() error = %v", err)
	}
	defer func() { _ = svc.Close() }()

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/auth/password-policy", nil)
	mux.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("metadata status = %d, body=%q", recorder.Code, recorder.Body.String())
	}
	var metadata map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &metadata); err != nil {
		t.Fatalf("decode metadata: %v", err)
	}
	if len(metadata) != 3 ||
		metadata["profile"] != config.DashboardSecurityProfileProduction ||
		metadata["sha256"] != digest ||
		metadata["entryCount"] != float64(auth.MinimumProductionPasswordBlocklistEntries) {
		t.Fatalf("metadata = %#v", metadata)
	}
	if strings.Contains(recorder.Body.String(), blocklistPath) ||
		strings.Contains(recorder.Body.String(), productionRouterBlocklistCanary) {
		t.Fatalf("metadata exposed corpus data: %q", recorder.Body.String())
	}
}

const productionRouterBlocklistCanary = "router metadata breached canary credential"

func writeRouterProductionBlocklist(t *testing.T) (string, string) {
	t.Helper()

	var contents strings.Builder
	contents.WriteString(productionRouterBlocklistCanary)
	contents.WriteByte('\n')
	for index := 1; index < auth.MinimumProductionPasswordBlocklistEntries; index++ {
		contents.WriteString(fmt.Sprintf("router offline corpus entry %05d\n", index))
	}
	raw := []byte(contents.String())
	path := filepath.Join(t.TempDir(), "production-passwords.txt")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write production blocklist: %v", err)
	}
	sum := sha256.Sum256(raw)
	return path, hex.EncodeToString(sum[:])
}
