package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func TestNormalizePublicBaseURL(t *testing.T) {
	t.Parallel()
	for raw, want := range map[string]string{
		"https://mom.vllm-sr.ai":          "https://mom.vllm-sr.ai",
		"https://mom.vllm-sr.ai/":         "https://mom.vllm-sr.ai",
		"http://gateway.example:8899/v1/": "http://gateway.example:8899/v1",
		"https://edge.example/router":     "https://edge.example/router",
	} {
		got, err := normalizePublicBaseURL(raw)
		if err != nil || got != want {
			t.Errorf("normalizePublicBaseURL(%q) = %q, %v; want %q", raw, got, err, want)
		}
	}
	for _, raw := range []string{
		"mom.vllm-sr.ai",
		"ftp://mom.vllm-sr.ai",
		"https://",
		"https://user:secret@mom.vllm-sr.ai",
		"https://mom.vllm-sr.ai/?token=x",
		"https://mom.vllm-sr.ai/#x",
		"://bad",
	} {
		if got, err := normalizePublicBaseURL(raw); err == nil {
			t.Errorf("normalizePublicBaseURL(%q) = %q, want an error", raw, got)
		} else if strings.Contains(err.Error(), "secret") {
			t.Errorf("the error for %q quotes the value: %v", raw, err)
		}
	}
}

func TestSetupStateReportsTheDeclaredPublicEndpoint(t *testing.T) {
	t.Setenv(publicBaseURLEnv, " https://mom.vllm-sr.ai/ ")
	t.Setenv(publicRequiresKeyEnv, "true")
	configPath := createBootstrapSetupConfig(t, t.TempDir())

	resp, _ := getSetupState(t, configPath, setupmode.New(configPath, false))

	if resp.PublicBaseURL != "https://mom.vllm-sr.ai" || !resp.PublicRequiresKey {
		t.Fatalf("public endpoint = %q, requires key %t", resp.PublicBaseURL, resp.PublicRequiresKey)
	}
	if resp.ListenerPort != 8899 {
		t.Fatalf("listenerPort = %d", resp.ListenerPort)
	}

	unreadable := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(unreadable, []byte("listeners: [\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	fallback, _ := getSetupState(t, unreadable, setupmode.New(unreadable, false))
	if fallback.PublicBaseURL != "https://mom.vllm-sr.ai" || !fallback.PublicRequiresKey {
		t.Fatalf("an unreadable config dropped the public endpoint: %#v", fallback)
	}
}

func TestSetupStateOmitsAnUnsetOrInvalidPublicEndpoint(t *testing.T) {
	configPath := createBootstrapSetupConfig(t, t.TempDir())
	for _, env := range []map[string]string{
		{publicBaseURLEnv: "", publicRequiresKeyEnv: "true"},
		{publicBaseURLEnv: "https://user:secret@mom.vllm-sr.ai", publicRequiresKeyEnv: "true"},
		{publicBaseURLEnv: "gateway.example"},
	} {
		for name, value := range env {
			t.Setenv(name, value)
		}
		recorder := httptest.NewRecorder()
		SetupStateHandler(configPath, setupmode.New(configPath, false))(recorder, httptest.NewRequest(http.MethodGet, "/api/setup/state", nil))
		body := recorder.Body.String()
		if strings.Contains(body, "publicBaseUrl") || strings.Contains(body, "publicRequiresKey") || strings.Contains(body, "secret") {
			t.Fatalf("env %v exposed a public endpoint: %s", env, body)
		}
	}
}
