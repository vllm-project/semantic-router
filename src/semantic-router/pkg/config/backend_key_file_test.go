package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// keyFileConfig is one OpenAI backend whose auth block is auth.
func keyFileConfig(auth string) []byte {
	return []byte(`version: v0.5
listeners:
  - name: http
    port: 8899
providers:
  models:
    - name: gpt
      provider_model_id: gpt-4o-mini
      api_format: chat_completions
      backend_refs:
        - name: openai
          provider: openai
          endpoint: https://api.openai.com/v1
          ` + auth + `
recipes:
  - name: default
`)
}

func writeKeyFile(t *testing.T, key string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "openai.key")
	if err := os.WriteFile(path, []byte(key), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestBackendReadsItsKeyFromAFileAtEveryLoad(t *testing.T) {
	path := writeKeyFile(t, "file-key-one\n")
	document := keyFileConfig("auth: {api_key_file: " + path + "}")
	cfg, err := ParseYAMLBytes(document)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.VLLMEndpoints) != 1 || cfg.VLLMEndpoints[0].APIKey != "file-key-one" {
		t.Fatalf("endpoints = %+v, want the file's key", cfg.VLLMEndpoints)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), "api_key_file: "+path) || strings.Contains(string(exported), "file-key-one") {
		t.Fatalf("export must name the file, not the key:\n%s", exported)
	}

	// A reload reads the file again, so a rotated key applies.
	if err = os.WriteFile(path, []byte("file-key-two"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err = ParseYAMLBytes(document)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VLLMEndpoints[0].APIKey != "file-key-two" {
		t.Fatalf("reload kept the old key")
	}
}

func TestBackendKeyFileFailsClosed(t *testing.T) {
	path := writeKeyFile(t, "file-key-one")
	empty := writeKeyFile(t, " \n")
	for _, tc := range []struct{ name, auth, want string }{
		{"missing file", "auth: {api_key_file: " + path + ".gone}", "cannot be read"},
		{"empty file", "auth: {api_key_file: " + empty + "}", "is empty"},
		{"relative path", "auth: {api_key_file: openai.key}", "absolute path"},
		{"inline key too", "api_key: inline-key\n          auth: {api_key_file: " + path + "}", "set one"},
		{"environment too", "auth: {api_key_env: OPENAI_API_KEY, api_key_file: " + path + "}", "set one"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(keyFileConfig(tc.auth))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
			if strings.Contains(err.Error(), "file-key-one") || strings.Contains(err.Error(), "inline-key") {
				t.Fatalf("error exposes the key: %v", err)
			}
		})
	}
}

// A validation that resolves no reference, such as -validate-config in
// `vllm-sr serve`, runs where the key file is not mounted: it checks the
// path's shape and leaves the file to the Router that loads the config.
func TestValidationChecksAKeyFileShapeNotItsContent(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "provider-keys", "key-1")
	document := keyFileConfig("auth: {api_key_file: " + missing + "}")
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(document)
	if err != nil {
		t.Fatalf("validation read the key file: %v", err)
	}
	if cfg.VLLMEndpoints[0].APIKey != "" {
		t.Fatalf("validation resolved a key")
	}
	if _, err = ParseYAMLBytesWithoutEnvExpansion(keyFileConfig("auth: {api_key_file: provider-keys/key-1}")); err == nil || !strings.Contains(err.Error(), "absolute path") {
		t.Fatalf("relative path error = %v, want the shape check", err)
	}
	if _, err = ParseYAMLBytes(document); err == nil || !strings.Contains(err.Error(), "cannot be read") {
		t.Fatalf("load error = %v, want the missing file refused", err)
	}
}
