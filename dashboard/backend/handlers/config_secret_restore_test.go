package handlers

import (
	"bytes"
	"errors"
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
			if _, restoreErr := restoreConfigSecrets(original, updated); restoreErr == nil {
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

func TestConfigSecretRestorationRejectsCrossBranchRetargeting(t *testing.T) {
	original := []byte("backends:\n  primary:\n    endpoint: https://old.example/v1\ncredentials:\n  primary:\n    api_key: credential-fixture\n")
	redacted, err := redactConfigYAML(original)
	if err != nil {
		t.Fatal(err)
	}
	updated := bytes.Replace(redacted, []byte("https://old.example/v1"), []byte("https://attacker.example/v1"), 1)
	if _, err := restoreConfigSecrets(original, updated); err == nil {
		t.Fatal("credential was restored after its endpoint changed in another config branch")
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
    delegated_credential:
      enabled: true
      header: x-delegated-credential
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
	for _, visible := range []string{"${UPSTREAM_KEY}", "/run/secrets/key", "debug: authenticated", "enabled: true", "header: x-delegated-credential"} {
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

func TestConfigSecretRestorationProtectsRedisClusterTargets(t *testing.T) {
	const nodes = "  cluster_addresses: [redis-a.example.test:6379, redis-b.example.test:6379]\n"
	const cluster = "  cluster_mode: true\n" + nodes
	for _, tc := range []struct {
		name, before, after, credential string
		wantReject                      bool
	}{
		{name: "unchanged cluster", before: cluster, after: cluster},
		{name: "replace node", before: cluster, after: strings.ReplaceAll(cluster, "redis-a.example.test", "different.example.test"), wantReject: true},
		{name: "add node", before: cluster, after: strings.ReplaceAll(cluster, "]", ", different.example.test:6379]"), wantReject: true},
		{name: "remove node", before: cluster, after: strings.ReplaceAll(cluster, ", redis-b.example.test:6379", ""), wantReject: true},
		{name: "remove nodes field", before: cluster, after: "  cluster_mode: true\n", wantReject: true},
		{name: "empty nodes", before: cluster, after: "  cluster_mode: true\n  cluster_addresses: []\n", wantReject: true},
		{name: "switch to standalone", before: cluster, after: "  cluster_mode: false\n" + nodes, wantReject: true},
		{name: "enable dormant cluster", before: nodes, after: cluster, wantReject: true},
		{name: "introduce cluster", after: cluster, wantReject: true},
		{name: "introduce dormant nodes", after: nodes, wantReject: true},
		{name: "explicit standalone default", after: "  cluster_mode: false\n  cluster_addresses: []\n"},
		{name: "remove empty defaults", before: "  cluster_mode: false\n  cluster_addresses: []\n"},
		{name: "ordinary setting", before: cluster, after: cluster + "  pool_size: 20\n"},
		{name: "replace password for new target", before: cluster, after: strings.ReplaceAll(cluster, "redis-a.example.test", "different.example.test"), credential: "replacement-fixture"},
		{name: "clear password for new target", before: cluster, after: strings.ReplaceAll(cluster, "redis-a.example.test", "different.example.test"), credential: "\"\""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prefix := "redis:\n  address: redis.example.test:6379\n  password: "
			original := []byte(prefix + configSecretFixture + "\n" + tc.before)
			credential := tc.credential
			if credential == "" {
				credential = redactedConfigSecret
			}
			updated := []byte(prefix + credential + "\n" + tc.after)
			restored, err := restoreConfigSecrets(original, updated)
			if tc.wantReject {
				if !errors.Is(err, errConfigSecretReuse) {
					t.Fatalf("target change reused a stored password: err=%v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(restored, []byte(redactedConfigSecret)) {
				t.Fatal("persisted placeholder instead of a credential")
			}
			if retained := bytes.Contains(restored, []byte(configSecretFixture)); retained != (tc.credential == "") {
				t.Fatalf("old credential retained=%v", retained)
			}
		})
	}
}

func TestConfigSecretRedactionPreservesDelegatedPolicy(t *testing.T) {
	original := []byte(`global:
  services:
    authz:
      delegated_credential:
        enabled: true
        header: x-reviewed-credential
        token: nested-fixture
  model_catalog:
    deployments:
      team:
        auth:
          credential: delegated
          api_key: deployment-fixture
      another-team:
        auth:
          credential: unknown-mode-fixture
providers:
  models:
    - backend_refs:
        - auth:
            credential: delegated
            api_key: backend-fixture
unrelated:
  auth:
    credential: delegated
  delegated_credential: unrelated-fixture
`)
	masked, err := redactConfigYAML(original)
	if err != nil {
		t.Fatal(err)
	}
	node, err := configSecretDocument(masked)
	if err != nil {
		t.Fatal(err)
	}
	global := mappingValueNode(node, "global")
	authz := mappingValueNode(mappingValueNode(global, "services"), "authz")
	policy := mappingValueNode(authz, "delegated_credential")
	if enabled := mappingValueNode(policy, "enabled"); enabled == nil || enabled.Tag != "!!bool" || enabled.Value != "true" {
		t.Error("delegated credential policy lost its boolean enabled field")
	}
	if header := mappingValueNode(policy, "header"); header == nil || header.Value != "x-reviewed-credential" {
		t.Error("delegated credential policy lost its header name")
	}
	deployments := mappingValueNode(mappingValueNode(global, "model_catalog"), "deployments")
	auth := mappingValueNode(mappingValueNode(deployments, "team"), "auth")
	if mode := mappingValueNode(auth, "credential"); mode == nil || mode.Value != "delegated" {
		t.Error("deployment credential mode was treated as a stored secret")
	}
	models := mappingValueNode(mappingValueNode(node, "providers"), "models")
	backends := mappingValueNode(models.Content[0], "backend_refs")
	backendAuth := mappingValueNode(backends.Content[0], "auth")
	if mode := mappingValueNode(backendAuth, "credential"); mode == nil || mode.Value != "delegated" {
		t.Error("backend credential mode was treated as a stored secret")
	}
	unrelated := mappingValueNode(mappingValueNode(node, "unrelated"), "auth")
	if credential := mappingValueNode(unrelated, "credential"); credential == nil || credential.Value != redactedConfigSecret {
		t.Fatal("policy exception unmasked an unrelated credential")
	}
	for _, hidden := range []string{"nested-fixture", "deployment-fixture", "backend-fixture", "unknown-mode-fixture", "unrelated-fixture"} {
		if bytes.Contains(masked, []byte(hidden)) {
			t.Fatalf("policy exception exposed %s", hidden)
		}
	}
	restored, err := restoreConfigSecrets(original, masked)
	if err != nil {
		t.Fatal(err)
	}
	before, err := configSecretDocument(original)
	if err != nil {
		t.Fatal(err)
	}
	after, err := configSecretDocument(restored)
	if err != nil {
		t.Fatal(err)
	}
	if !sameConfigNode(before, after) {
		t.Fatal("policy fields or credentials changed on round trip")
	}
}
