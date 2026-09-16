package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func TestSetupRejectsUnknownFieldsBeforeActivationPersistence(t *testing.T) {
	for _, endpoint := range []string{"validate", "activate"} {
		t.Run(endpoint, func(t *testing.T) {
			root := t.TempDir()
			configPath := createBootstrapSetupConfig(t, root)
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			patch := createValidSetupPatch()
			patch["global"] = map[string]interface{}{
				"router": map[string]interface{}{"modules": map[string]interface{}{"classifier": map[string]interface{}{}}},
			}
			resolver := setupmode.New(configPath, false)
			handler := SetupValidateHandler(configPath, resolver)
			if endpoint == "activate" {
				handler = SetupActivateHandler(configPath, false, root, resolver)
			}
			body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
			w := httptest.NewRecorder()
			handler(w, httptest.NewRequest(http.MethodPost, "/api/setup/"+endpoint, bytes.NewReader(body)))
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "modules") {
				t.Fatalf("unknown module location was silently discarded: %d %s", w.Code, w.Body.String())
			}
			after, err := os.ReadFile(configPath)
			if err != nil || !bytes.Equal(before, after) || !resolver.Active() {
				t.Fatalf("invalid setup changed active configuration: %v", err)
			}
		})
	}
}

func TestSetupRemoteImportRejectsUnknownFieldsBeforeTypedTransport(t *testing.T) {
	root := t.TempDir()
	configPath := createBootstrapSetupConfig(t, root)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("version: v0.3\nglobal:\n  router:\n    modules: {}\n"))
	}))
	defer server.Close()
	body := mustJSONRaw(t, SetupImportRemoteRequest{URL: server.URL})
	w := httptest.NewRecorder()
	SetupImportRemoteHandler(configPath, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(body)))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "modules") {
		t.Fatalf("invalid remote profile entered typed transport: %d %s", w.Code, w.Body.String())
	}
}

func TestSetupStrictDecodePreservesBootstrapMarkerStripping(t *testing.T) {
	parsed, err := parseSetupCanonicalConfig([]byte("version: v0.3\nsetup:\n  mode: true\n"))
	if err != nil || parsed.Setup != nil || parsed.Version != "v0.3" {
		t.Fatalf("valid imported bootstrap marker was not removed: config=%+v err=%v", parsed, err)
	}
	root := t.TempDir()
	configPath := createBootstrapSetupConfig(t, root)
	patch := createValidSetupPatch()
	patch["setup"] = map[string]interface{}{"mode": true}
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	candidate, err := buildSetupCandidateConfig(configPath, bytes.NewReader(body), setupmode.New(configPath, false))
	if err != nil || candidate.Setup != nil {
		t.Fatalf("valid patched bootstrap marker was not removed: config=%+v err=%v", candidate, err)
	}
	for _, invalid := range []string{
		"version: v0.3\nsetup:\n  mode: true\n  unknown: true\n",
		"version: v0.3\nsetup:\n  mode: not-a-boolean\n",
	} {
		if _, err := parseSetupCanonicalConfig([]byte(invalid)); err == nil {
			t.Fatalf("invalid setup metadata was accepted: %q", invalid)
		}
	}
}
