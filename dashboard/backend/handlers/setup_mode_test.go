package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// createActivatedSetupConfig writes the same config without the setup block,
// the shape the file has after activation.
func createActivatedSetupConfig(t *testing.T, dir string) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	config := map[string]interface{}{
		"version": "v0.3",
		"listeners": []map[string]interface{}{
			{
				"name":    "http-8899",
				"address": "0.0.0.0",
				"port":    8899,
				"timeout": "300s",
			},
		},
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		t.Fatalf("failed to marshal activated config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("failed to write activated config: %v", err)
	}
	return configPath
}

// createCanonicallyInvalidSetupConfig writes a config with a readable setup
// block and a type error in listeners.
//
// setupmode decodes only the setup block, so it resolves this file cleanly,
// while readSetupConfigFile decodes the full schema and fails. That is the only
// way to reach SetupStateHandler's read-error branch with a good resolution.
func createCanonicallyInvalidSetupConfig(t *testing.T, dir string, setupMode bool) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	body := "version: v0.3\nlisteners: \"not-a-list\"\n"
	if setupMode {
		body += "setup:\n  mode: true\n  state: bootstrap\n"
	}
	if err := os.WriteFile(configPath, []byte(body), 0o644); err != nil {
		t.Fatalf("failed to write canonically invalid config: %v", err)
	}
	return configPath
}

// decodeSetupState also returns the raw body, because reason is omitempty and
// only the raw body can prove it is absent.
func decodeSetupState(t *testing.T, w *httptest.ResponseRecorder) (SetupStateResponse, string) {
	t.Helper()

	raw := w.Body.String()
	var resp SetupStateResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("failed to decode setup state %q: %v", raw, err)
	}
	return resp, raw
}

// getSetupState issues GET /api/setup/state against the given resolver.
func getSetupState(t *testing.T, configPath string, resolver *setupmode.Resolver) (SetupStateResponse, string) {
	t.Helper()

	w := httptest.NewRecorder()
	SetupStateHandler(configPath, resolver)(w, httptest.NewRequest(http.MethodGet, "/api/setup/state", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("setup state status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	return decodeSetupState(t, w)
}

// --- Resolved setup state on /api/setup/state (#2795) ----------------------

// The happy path must not carry a reason. Asserted on the raw body, because a
// decoded struct cannot tell an empty string from an omitted key.
func TestSetupStateHandlerOmitsReasonOnCleanResolution(t *testing.T) {
	configPath := createBootstrapSetupConfig(t, t.TempDir())

	resp, raw := getSetupState(t, configPath, setupmode.New(configPath, true))

	if !resp.SetupMode {
		t.Fatalf("setupMode = false, want true; body=%s", raw)
	}
	if resp.Reason != "" {
		t.Fatalf("reason = %q, want empty on a clean resolution", resp.Reason)
	}
	if strings.Contains(raw, "reason") {
		t.Fatalf("raw body contains a reason key, want it omitted entirely: %s", raw)
	}
}

// A stale DASHBOARD_SETUP_MODE against an activated config is the invisible
// case from the issue. The state stays false and the reason makes it visible.
func TestSetupStateHandlerExplainsStaleLegacyFlag(t *testing.T) {
	configPath := createActivatedSetupConfig(t, t.TempDir())

	resp, raw := getSetupState(t, configPath, setupmode.New(configPath, true))

	if resp.SetupMode {
		t.Fatalf("setupMode = true from the legacy flag alone, want false; body=%s", raw)
	}
	if resp.Reason == "" {
		t.Fatalf("reason is empty for a conflicting legacy flag; body=%s", raw)
	}
	if !strings.Contains(resp.Reason, "DASHBOARD_SETUP_MODE") {
		t.Fatalf("reason %q does not name the stale input", resp.Reason)
	}
	if !strings.Contains(raw, `"reason"`) {
		t.Fatalf("raw body is missing the reason key: %s", raw)
	}
}

// An unreadable config used to produce a 500, which the frontend swallowed into
// "not in setup mode" with no explanation. Answer 200 with the reason instead.
func TestSetupStateHandlerAnswers200WithReasonWhenConfigUnreadable(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "does-not-exist.yaml")

	w := httptest.NewRecorder()
	SetupStateHandler(missingPath, setupmode.New(missingPath, false))(
		w, httptest.NewRequest(http.MethodGet, "/api/setup/state", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 with a diagnostic reason; body=%s", w.Code, w.Body.String())
	}

	resp, raw := decodeSetupState(t, w)
	if resp.SetupMode {
		t.Fatalf("setupMode = true for an unreadable config, want false (fail closed); body=%s", raw)
	}
	if !strings.Contains(resp.Reason, "unreadable") {
		t.Fatalf("reason = %q, want it to mention the config being unreadable", resp.Reason)
	}
}

// The write endpoints share the gate, so on an activated config they must all
// refuse.
func TestSetupWriteEndpointsGateOnResolvedState(t *testing.T) {
	configPath := createActivatedSetupConfig(t, t.TempDir())
	resolver := setupmode.New(configPath, true)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	t.Run("validate", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupValidateHandler(configPath, resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})

	t.Run("import-remote", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupImportRemoteHandler(configPath, resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", strings.NewReader(`{"url":"https://example.com/c.yaml"}`)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})

	t.Run("activate", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupActivateHandler(configPath, false, filepath.Dir(configPath), resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})
}

// Proves the Invalidate call works.
//
// Activation writes the config then answers the request, and on coarse mtime
// granularity the identity check cannot see the change on its own. The same
// resolver is reused and nothing restarts, so a stale cached resolution would
// show up as /api/setup/state still reporting true after activation.
func TestSetupActivateHandlerFlipsSetupStateWithinOneRequest(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)
	// The legacy flag stays true across activation, as it does in a real
	// deployment: the CLI sets it at launch and nothing clears it.
	resolver := setupmode.New(configPath, true)

	before, rawBefore := getSetupState(t, configPath, resolver)
	if !before.SetupMode {
		t.Fatalf("setupMode = false before activation, want true; body=%s", rawBefore)
	}
	if before.Reason != "" {
		t.Fatalf("reason = %q before activation, want empty (flag and config agree)", before.Reason)
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	w := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, tempDir, resolver)(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activate status = %d, want 200; body=%s", w.Code, w.Body.String())
	}

	after, rawAfter := getSetupState(t, configPath, resolver)
	if after.SetupMode {
		t.Fatalf("setupMode = true after activation; the cached resolution was not invalidated; body=%s", rawAfter)
	}
	// The flag now disagrees with the config, so the response also explains
	// why the environment value lost.
	if !strings.Contains(after.Reason, "DASHBOARD_SETUP_MODE") {
		t.Fatalf("reason = %q after activation, want it to name the now-stale legacy flag", after.Reason)
	}

	// The write endpoints closed in the same moment, through the same resolver.
	vw := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(
		vw, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))
	if vw.Code != http.StatusBadRequest {
		t.Fatalf("validate status = %d after activation, want 400; body=%s", vw.Code, vw.Body.String())
	}
}

// Covers the branch where the two decoders disagree: the resolver answers
// cleanly from the setup block while this handler cannot decode the full schema.
//
// The response must carry the resolved value, not a hardcoded false. Reporting
// false while the bootstrap gate reads true is the invisible-open-door split
// this change removes. The resolution is clean, so its Reason is empty and the
// handler supplies its own.
func TestSetupStateHandlerReportsResolvedStateWhenCanonicalDecodeFails(t *testing.T) {
	t.Run("setup mode active: state must agree with the bootstrap gate", func(t *testing.T) {
		configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), true)
		// legacyFlag=true matches the config, so the resolution is clean.
		resolver := setupmode.New(configPath, true)

		if !resolver.Active() {
			t.Fatalf("precondition failed: resolver must read the setup block of a canonically invalid config")
		}
		if reason := resolver.Resolve().Reason; reason != "" {
			t.Fatalf("precondition failed: resolution should be clean, got reason %q", reason)
		}

		resp, raw := getSetupState(t, configPath, resolver)

		if !resp.SetupMode {
			t.Fatalf("setupMode = false while the bootstrap gate is open; the surfaces disagree. body=%s", raw)
		}
		if resp.Reason != unreadableConfigReason {
			t.Fatalf("reason = %q, want the handler's own explanation %q", resp.Reason, unreadableConfigReason)
		}
		if !strings.Contains(raw, `"reason"`) {
			t.Fatalf("raw body is missing the reason key: %s", raw)
		}
		// The rest of the payload is unavailable, so it must be empty.
		if resp.ListenerPort != 0 || resp.Models != 0 || resp.Decisions != 0 || resp.CanActivate {
			t.Fatalf("expected an empty payload alongside the reason, got %+v", resp)
		}
	})

	t.Run("setup mode inactive", func(t *testing.T) {
		configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), false)
		resolver := setupmode.New(configPath, false)

		resp, raw := getSetupState(t, configPath, resolver)

		if resp.SetupMode {
			t.Fatalf("setupMode = true for a config with no setup block; body=%s", raw)
		}
		if resp.Reason != unreadableConfigReason {
			t.Fatalf("reason = %q, want %q", resp.Reason, unreadableConfigReason)
		}
	})

	// The reason is served unauthenticated, so it must disclose neither the
	// config location nor its contents.
	t.Run("reason discloses neither path nor contents", func(t *testing.T) {
		dir := t.TempDir()
		configPath := createCanonicallyInvalidSetupConfig(t, dir, true)

		resp, _ := getSetupState(t, configPath, setupmode.New(configPath, true))

		for _, secret := range []string{configPath, dir, "not-a-list"} {
			if strings.Contains(resp.Reason, secret) {
				t.Fatalf("reason %q discloses %q", resp.Reason, secret)
			}
		}
	})
}

// The write endpoints gate first, then read. When the gate passes but the read
// fails they must report the read failure, not an inactive gate, which would
// send an operator looking in the wrong place.
func TestSetupWriteEndpointsReportUnreadableConfigWhileSetupModeIsActive(t *testing.T) {
	configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), true)
	resolver := setupmode.New(configPath, true)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	w := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "failed to read existing config") {
		t.Fatalf("body = %q, want it to report the read failure rather than an inactive gate", w.Body.String())
	}
}
