package handlers

import (
	"bytes"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestConfigSecretRestorationRejectsRetargeting(t *testing.T) {
	original := []byte("backends:\n  - name: primary\n    endpoint: https://old.example/v1\n    auth:\n      api_key: credential-fixture\n  - name: other\n    endpoint: https://other.example/v1\n    auth:\n      api_key: other-credential-fixture\n")
	redacted, err := redactConfigYAML(original)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, before, after string
	}{
		{"endpoint", "https://old.example/v1", "https://attacker.example/v1"},
		{"identity", "name: primary", "name: impostor"},
		{"unmatched field", "api_key:", "access_key:"},
		{"embedded marker", redactedConfigSecret, "prefix-" + redactedConfigSecret},
	} {
		t.Run(tc.name, func(t *testing.T) {
			updated := bytes.Replace(redacted, []byte(tc.before), []byte(tc.after), 1)
			if _, err := restoreConfigSecrets(original, updated); err == nil {
				t.Fatal("credential reuse was accepted after target/field change")
			}
		})
	}
	// Match named objects rather than accidentally binding credentials by index.
	node, err := configSecretDocument(redacted)
	if err != nil {
		t.Fatal(err)
	}
	list := mappingValueNode(node, "backends")
	list.Content[0], list.Content[1] = list.Content[1], list.Content[0]
	reordered, err := yaml.Marshal(node)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := restoreConfigSecrets(original, reordered)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Backends []struct {
			Name string `yaml:"name"`
			Auth struct {
				APIKey string `yaml:"api_key"`
			} `yaml:"auth"`
		} `yaml:"backends"`
	}
	if err := yaml.Unmarshal(restored, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Backends[0].Auth.APIKey != "other-credential-fixture" || decoded.Backends[1].Auth.APIKey != "credential-fixture" {
		t.Fatal("reordering changed credential ownership")
	}
}

func TestConfigSecretRedactionCoversHeadersURLsAndReferences(t *testing.T) {
	original := []byte(`api_keys: [listener-fixture]
auth:
  api_key_env: UPSTREAM_KEY
  api_key_file: /run/secrets/key
  api_key: ${UPSTREAM_KEY}
extra_headers:
  X-Custom: header-fixture
endpoint: https://user:url-fixture@example.test/v1?query-fixture=value#fragment-fixture
password: ${UNSET:-fallback-fixture}
router:
  headers:
    debug: authenticated
services:
  authz:
    gateway_credential:
      enabled: true
      header: x-gateway-credential
`)
	masked, err := redactConfigYAML(original)
	if err != nil {
		t.Fatal(err)
	}
	for _, hidden := range []string{"listener-fixture", "header-fixture", "url-fixture", "query-fixture", "fragment-fixture", "fallback-fixture"} {
		if bytes.Contains(masked, []byte(hidden)) {
			t.Errorf("secret %s remained visible", hidden)
		}
	}
	for _, visible := range []string{"${UPSTREAM_KEY}", "/run/secrets/key", "debug: authenticated", "enabled: true", "header: x-gateway-credential"} {
		if !bytes.Contains(masked, []byte(visible)) {
			t.Errorf("safe metadata %s was lost", visible)
		}
	}
	restored, err := restoreConfigSecrets(original, masked)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := configSecretDocument(original)
	after, _ := configSecretDocument(restored)
	if !sameConfigNode(before, after) {
		t.Fatal("redacted view did not round-trip to the original config")
	}
	retargeted := bytes.Replace(masked, []byte("example.test"), []byte("attacker.test"), 1)
	if _, err := restoreConfigSecrets(original, retargeted); err == nil {
		t.Fatal("URL credentials were reused for another origin")
	}
}

func TestConfigSecretRedactionRejectsYAMLIndirection(t *testing.T) {
	for _, document := range []string{
		"hidden: &secret credential-fixture\npassword: *secret\n",
		"hidden: &secret {password: credential-fixture}\nauth: {<<: *secret}\n",
		"password: first\npassword: credential-fixture\n",
	} {
		masked, err := redactConfigYAML([]byte(document))
		if err == nil && strings.Contains(string(masked), "credential-fixture") {
			t.Fatal("YAML indirection exposed a credential")
		}
	}
}

func TestConfigSecretRedactionCoversMalformedURLs(t *testing.T) {
	for _, endpoint := range []string{
		"https://user:credential-fixture@host%zz/v1",
		"https://example.test/%zz?credential-fixture",
		"https://user:credential-fixture@[::1",
	} {
		original := []byte("url: " + endpoint + "\n")
		masked, err := redactConfigYAML(original)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(masked, []byte("credential-fixture")) {
			t.Error("malformed URL retained credentials")
		}
		restored, err := restoreConfigSecrets(original, masked)
		if err != nil {
			t.Fatal(err)
		}
		before, _ := configSecretDocument(original)
		after, _ := configSecretDocument(restored)
		if !sameConfigNode(before, after) {
			t.Error("redacted malformed URL did not round-trip")
		}
	}
}
