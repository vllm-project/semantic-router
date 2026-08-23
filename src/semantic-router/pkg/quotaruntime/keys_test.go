package quotaruntime

import (
	"strings"
	"testing"
	"time"
)

func TestEveryRuntimeKeyUsesOnePartitionHashTag(t *testing.T) {
	t.Parallel()

	partition, err := newPartitionKeys("namespace-1")
	if err != nil {
		t.Fatalf("newPartitionKeys() error = %v", err)
	}
	rules, err := compileRules("namespace-1", []RuleBinding{
		requestRule(t, "binding:a", "rule:a", "12", time.Minute, 0),
		tokenRule(t, "binding:b", "rule:b", "100", time.Minute, 1),
	})
	if err != nil {
		t.Fatalf("compileRules() error = %v", err)
	}
	keys := []string{
		partition.pendingIndex,
		partition.pending("admission:a"),
		partition.dispatches("admission:a"),
		partition.attempts("admission:a"),
		partition.terminal("admission:a"),
		partition.fence("fence:a"),
	}
	access, err := NewAccessProjectionKeyspace("namespace-1")
	if err != nil {
		t.Fatalf("NewAccessProjectionKeyspace() error = %v", err)
	}
	keys = append(keys,
		access.Credential("api-key", "kid:a"),
		access.Active("key:a"),
		access.Policy("key:a", "7"),
		access.Deny("team", "team:a"),
		access.ManagementSession("session:a"),
	)
	for _, rule := range rules {
		keys = append(keys, rule.keys.meta, rule.keys.events, rule.keys.values, rule.keys.fences)
	}
	if err := validateSingleHashTag(keys, "{namespace-1}"); err != nil {
		t.Fatalf("validateSingleHashTag() error = %v", err)
	}
	for _, key := range keys {
		if strings.Count(key, "{") != 1 || strings.Count(key, "}") != 1 {
			t.Errorf("key %q has an ambiguous Redis hash tag", key)
		}
	}

	foreign := append([]string(nil), keys...)
	foreign = append(foreign, "quota:{other}:counter")
	if err := validateSingleHashTag(foreign, "{namespace-1}"); err == nil {
		t.Fatal("validateSingleHashTag() accepted a cross-partition key")
	}
}

func TestCredentialDirectoryCannotEnterAtomicAdmission(t *testing.T) {
	t.Parallel()

	directory, err := CredentialDirectoryKey("api-key", "kid-1")
	if err != nil {
		t.Fatalf("CredentialDirectoryKey() error = %v", err)
	}
	if err := validateSingleHashTag([]string{directory}, "{namespace-1}"); err == nil {
		t.Fatal("global credential directory was accepted as an atomic admission key")
	}
}

func TestKeyComponentsCannotInjectHashTags(t *testing.T) {
	t.Parallel()

	encoded := keyComponent("binding}:{other}")
	if strings.ContainsAny(encoded, "{}:") {
		t.Fatalf("keyComponent() = %q, want URL-safe opaque component", encoded)
	}
}

func TestConfiguredKeyPrefixScopesEveryRuntimeKey(t *testing.T) {
	t.Parallel()

	const prefix = "vllm-sr:access:test"
	partition, err := newPartitionKeysWithPrefix(prefix, "namespace-1")
	if err != nil {
		t.Fatalf("newPartitionKeysWithPrefix() error = %v", err)
	}
	access, err := NewAccessProjectionKeyspaceWithPrefix(prefix, "namespace-1")
	if err != nil {
		t.Fatalf("NewAccessProjectionKeyspaceWithPrefix() error = %v", err)
	}
	directory, err := CredentialDirectoryKeyWithPrefix(prefix, "api-key", "kid-1")
	if err != nil {
		t.Fatalf("CredentialDirectoryKeyWithPrefix() error = %v", err)
	}
	keys := []string{
		partition.pendingIndex,
		partition.pending("admission"),
		partition.attempts("admission"),
		partition.terminal("admission"),
		access.Credential("api-key", "kid-1"),
		access.Active("key-1"),
		access.Policy("key-1", "1"),
		access.Deny("key", "key-1"),
		access.ManagementSession("session-1"),
		directory,
	}
	for _, key := range keys {
		if !strings.HasPrefix(key, prefix+":") {
			t.Errorf("key %q is outside prefix %q", key, prefix)
		}
	}
	if err := validateRuntimeKeys(keys[:len(keys)-1], "{namespace-1}", prefix); err != nil {
		t.Fatalf("validateRuntimeKeys() error = %v", err)
	}
	if err := validateRuntimeKeys([]string{"access:{namespace-1}:active:key"}, "{namespace-1}", prefix); err == nil {
		t.Fatal("validateRuntimeKeys() accepted an unprefixed key")
	}
	for _, invalid := range []string{"ends:", ":starts", "contains space", "has{tag}"} {
		if _, err := newPartitionKeysWithPrefix(invalid, "namespace-1"); err == nil {
			t.Errorf("newPartitionKeysWithPrefix(%q) accepted invalid prefix", invalid)
		}
	}
}
