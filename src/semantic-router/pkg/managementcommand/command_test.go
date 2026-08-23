package managementcommand

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID = "11111111-1111-4111-8111-111111111111"
	testPrincipalID = "22222222-2222-4222-8222-222222222222"
)

func TestCodecBindsKeyAndRequestToExactScope(t *testing.T) {
	codec, err := NewCodec(testKeyring("v1"))
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	bind := func(namespace, principal, endpoint string, request []byte) Command {
		t.Helper()
		command, err := codec.Bind(
			NamespaceCommandScope(namespace), principal, endpoint, "opaque-key-0123456789", request,
			now, now.Add(time.Hour),
		)
		if err != nil {
			t.Fatal(err)
		}
		return command
	}
	baseline := bind(testNamespaceID, testPrincipalID, "/management/v1/resources", []byte(`{"secret":"one"}`))
	for name, candidate := range map[string]Command{
		"namespace": bind("33333333-3333-4333-8333-333333333333", testPrincipalID, baseline.Endpoint, []byte(`{"secret":"one"}`)),
		"principal": bind(testNamespaceID, "44444444-4444-4444-8444-444444444444", baseline.Endpoint, []byte(`{"secret":"one"}`)),
		"endpoint":  bind(testNamespaceID, testPrincipalID, "/management/v1/other", []byte(`{"secret":"one"}`)),
		"request":   bind(testNamespaceID, testPrincipalID, baseline.Endpoint, []byte(`{"secret":"two"}`)),
	} {
		if baseline.ActiveDigest() == candidate.ActiveDigest() {
			t.Fatalf("%s did not alter either durable digest", name)
		}
	}
	active := baseline.ActiveDigest()
	if strings.Contains(string(active.KeyDigest[:]), "opaque-key") ||
		strings.Contains(string(active.RequestDigest[:]), "secret") {
		t.Fatal("durable command digest exposed submitted material")
	}
}

func TestCodecSeparatesClusterAndNamespaceScopes(t *testing.T) {
	codec, err := NewCodec(testKeyring("v1"))
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	bind := func(scope CommandScope) Command {
		command, bindErr := codec.Bind(scope, testPrincipalID, "/management/v1/resources",
			"opaque-key-0123456789", []byte(`{"name":"one"}`), now, now.Add(time.Hour))
		if bindErr != nil {
			t.Fatal(bindErr)
		}
		return command
	}
	cluster := bind(ClusterCommandScope())
	namespace := bind(NamespaceCommandScope(testNamespaceID))
	if cluster.ActiveDigest() == namespace.ActiveDigest() || cluster.AdvisoryLockKey() == namespace.AdvisoryLockKey() {
		t.Fatal("cluster and namespace command scopes shared a durable identity")
	}
	for _, invalid := range []CommandScope{
		{Kind: ScopeCluster, NamespaceID: testNamespaceID},
		{Kind: ScopeNamespace},
		{Kind: "unknown"},
	} {
		if _, err := codec.Bind(invalid, testPrincipalID, "/management/v1/resources",
			"opaque-key-0123456789", []byte(`{}`), now, now.Add(time.Hour)); err == nil {
			t.Fatalf("invalid scope %#v was accepted", invalid)
		}
	}
}

func TestCodecRejectsAmbiguousInputs(t *testing.T) {
	codec, err := NewCodec(testKeyring("v1"))
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	for name, key := range map[string]string{
		"short":   "short",
		"space":   "opaque key 0123456789",
		"newline": "opaque-key-012345\n",
	} {
		if _, err := codec.Bind(NamespaceCommandScope(testNamespaceID), testPrincipalID, "/management/v1/resources", key, []byte(`{}`), now, now.Add(time.Hour)); err == nil {
			t.Fatalf("%s idempotency key was accepted", name)
		}
	}
	if _, err := codec.Bind(NamespaceCommandScope(testNamespaceID), testPrincipalID, "/management//resources", "opaque-key-0123456789", []byte(`{}`), now, now.Add(time.Hour)); err == nil {
		t.Fatal("ambiguous endpoint was accepted")
	}
}

func TestCodecRejectsUnboundedOrMalformedKeyrings(t *testing.T) {
	keys := make(map[string][]byte, maximumRetainedHMACVersion+1)
	for index := 0; index <= maximumRetainedHMACVersion; index++ {
		keys[fmt.Sprintf("v%d", index)] = []byte(strings.Repeat("k", 32))
	}
	if _, err := NewCodec(securitykeyring.Symmetric{ActiveVersion: "v0", Keys: keys}); err == nil {
		t.Fatal("unbounded Management command keyring was accepted")
	}
	if _, err := NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "bad/version", Keys: map[string][]byte{"bad/version": []byte(strings.Repeat("k", 32))},
	}); err == nil {
		t.Fatal("malformed Management command key version was accepted")
	}
}

func TestCodecBindsAllRetainedVersionsAndKeepsStableAdvisoryIdentity(t *testing.T) {
	old := testKeyring("v1")
	old.Keys["v2"] = []byte(strings.Repeat("n", 32))
	rotated := testKeyring("v2")
	rotated.Keys["v1"] = []byte(strings.Repeat("k", 32))
	first, err := NewCodec(old)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewCodec(rotated)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	bind := func(codec *Codec) Command {
		command, err := codec.Bind(NamespaceCommandScope(testNamespaceID), testPrincipalID, "/management/v1/resources",
			"opaque-key-0123456789", []byte(`{"name":"one"}`), now, now.Add(time.Hour))
		if err != nil {
			t.Fatal(err)
		}
		return command
	}
	before, after := bind(first), bind(second)
	if before.AdvisoryLockKey() != after.AdvisoryLockKey() {
		t.Fatal("advisory identity changed when active HMAC version rotated")
	}
	if before.ActiveDigest().HMACVersion != "v1" || after.ActiveDigest().HMACVersion != "v2" ||
		len(before.CandidateDigests()) != 2 || len(after.CandidateDigests()) != 2 {
		t.Fatalf("versioned bindings = %#v / %#v", before.CandidateDigests(), after.CandidateDigests())
	}
}

func testKeyring(active string) securitykeyring.Symmetric {
	key := byte('k')
	if active == "v2" {
		key = 'n'
	}
	return securitykeyring.Symmetric{
		ActiveVersion: active,
		Keys:          map[string][]byte{active: []byte(strings.Repeat(string(key), 32))},
	}
}

func TestStoredResultRequiresExactlyOneKind(t *testing.T) {
	resource := &ResourceResult{ResourceType: "provider_credential", ResourceID: "33333333-3333-4333-8333-333333333333", ResourceRevision: 1, ResponseStatus: 201}
	operation := &OperationResult{OperationID: "44444444-4444-4444-8444-444444444444", ResponseStatus: 202}
	expiresAt := time.Now().UTC().Add(time.Hour)
	if err := (StoredResult{Resource: resource, ExpiresAt: expiresAt}).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (StoredResult{Operation: operation, ExpiresAt: expiresAt}).Validate(); err != nil {
		t.Fatal(err)
	}
	for _, value := range []StoredResult{
		{ExpiresAt: expiresAt},
		{Resource: resource, Operation: operation, ExpiresAt: expiresAt},
	} {
		if err := value.Validate(); err == nil {
			t.Fatal("invalid result kind was accepted")
		}
	}
}
