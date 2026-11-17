package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// listenerKeyConfig is one listener whose key settings are keys.
func listenerKeyConfig(keys string) []byte {
	return []byte(`version: v0.5
listeners:
  - name: http
    port: 8899
    ` + keys + `
providers:
  models:
    - name: gpt
      provider_model_id: gpt-4o-mini
      api_format: chat_completions
      backend_refs:
        - name: openai
          provider: openai
          endpoint: https://api.openai.com/v1
recipes:
  - name: default
`)
}

func TestListenerReadsItsKeysFromFilesAtEveryLoad(t *testing.T) {
	path := writeKeyFile(t, "gateway-key-one\n")
	document := listenerKeyConfig("api_keys: [inline-key]\n    api_key_files: [" + path + "]")
	cfg, err := ParseYAMLBytes(document)
	if err != nil {
		t.Fatal(err)
	}
	if keys := cfg.Listeners[0].ClientKeys(); strings.Join(keys, ",") != "inline-key,gateway-key-one" {
		t.Fatalf("client keys = %v, want the inline key and the file's", keys)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), path) || strings.Contains(string(exported), "gateway-key-one") {
		t.Fatalf("export must name the file, not its key:\n%s", exported)
	}

	if err = os.WriteFile(path, []byte("gateway-key-two"), 0o600); err != nil {
		t.Fatal(err)
	}
	if cfg, err = ParseYAMLBytes(document); err != nil {
		t.Fatal(err)
	}
	if keys := cfg.Listeners[0].ClientKeys(); keys[len(keys)-1] != "gateway-key-two" {
		t.Fatalf("reload kept the old key")
	}
}

func TestListenerKeyFileFailsClosed(t *testing.T) {
	path := writeKeyFile(t, "gateway-key-one")
	empty := writeKeyFile(t, " \n")
	for _, tc := range []struct{ name, keys, want string }{
		{"missing file", "api_key_files: [" + path + ".gone]", "cannot be read"},
		{"empty file", "api_key_files: [" + empty + "]", "is empty"},
		{"relative path", "api_key_files: [gateway.key]", "absolute path"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(listenerKeyConfig(tc.keys))
			if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "listener 'http': api_key_files") || strings.Contains(err.Error(), "auth.api_key_file") {
				t.Fatalf("error = %v, want %q for listener 'http' api_key_files", err, tc.want)
			}
			if strings.Contains(err.Error(), "gateway-key-one") {
				t.Fatalf("error exposes the key: %v", err)
			}
		})
	}
}

// A validation that resolves no reference checks a listener's key file paths
// and leaves the files to the Router that serves the configuration.
func TestValidationChecksListenerKeyFilePathsOnly(t *testing.T) {
	missing := listenerKeyConfig("api_key_files: [" + filepath.Join(t.TempDir(), "gone") + "]")
	if _, err := ParseYAMLBytesWithoutEnvExpansion(missing); err != nil {
		t.Fatalf("validation read the key file: %v", err)
	}
	if _, err := ParseYAMLBytesWithoutEnvExpansion(listenerKeyConfig("api_key_files: [gateway.key]")); err == nil || !strings.Contains(err.Error(), "absolute path") {
		t.Fatalf("relative path error = %v, want the path check", err)
	}
	if _, err := ParseYAMLBytes(missing); err == nil || !strings.Contains(err.Error(), "cannot be read") {
		t.Fatalf("load error = %v, want the missing file refused", err)
	}
}

func TestListenerKeyFilesNeedTheStandaloneGateway(t *testing.T) {
	cfg, err := ParseYAMLBytes(listenerKeyConfig("api_key_files: [" + writeKeyFile(t, "gateway-key-one") + "]"))
	if err != nil {
		t.Fatal(err)
	}
	if err = ValidateGatewayCapabilities(cfg, GatewayExtProc); err == nil || !strings.Contains(err.Error(), "api_key_files") {
		t.Fatalf("extproc error = %v, want api_key_files refused", err)
	}
	if err = ValidateGatewayCapabilities(cfg, GatewayStandalone); err != nil {
		t.Fatalf("standalone refused api_key_files: %v", err)
	}
}
