package providerdiscovery

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestDiscoveryClaimBindsAuthorityAndExactReturnedItems(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	codec, err := NewClaimCodec(ClaimKeyset{
		ActiveKeyID: "active", Keys: map[string][]byte{
			"active": []byte(strings.Repeat("a", 32)), "old": []byte(strings.Repeat("b", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan := testPlan()
	models, token, _, err := codec.Issue(plan, testAuthorityDigest, testCredentialVerID, []AdapterModel{
		{ProviderModelID: "model-a", DisplayName: "Model A"},
		{ProviderModelID: "model-b", DisplayName: "Model B"},
	}, now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	expectation := ClaimExpectation{
		NamespaceID: testNamespaceID, AuthorityDigest: testAuthorityDigest,
		CatalogRevision: testCatalogRevision, ProviderID: "provider-a",
	}
	verified, err := codec.VerifySelection(token, expectation, []string{models[1].CatalogItemID}, now.Add(30*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(verified.Models) != 1 || verified.Models[0].ProviderModelID != "model-b" ||
		verified.Binding.Origin != testPlan().NormalizedOrigin ||
		verified.Binding.CredentialID != testCredentialID ||
		verified.Binding.CredentialVersion != testCredentialVerID {
		t.Fatalf("verified selection = %+v", verified)
	}
	expectation.AuthorityDigest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	if _, err := codec.VerifySelection(token, expectation, []string{models[0].CatalogItemID}, now); !errors.Is(err, ErrInvalidClaim) {
		t.Fatalf("authority mismatch error = %v", err)
	}
	expectation.AuthorityDigest = testAuthorityDigest
	if _, err := codec.VerifySelection(token, expectation, []string{"pmi_invalid"}, now); !errors.Is(err, ErrInvalidClaim) {
		t.Fatalf("item mismatch error = %v", err)
	}
	if _, err := codec.VerifySelection(token, expectation, []string{models[0].CatalogItemID}, now.Add(2*time.Minute)); !errors.Is(err, ErrExpiredClaim) {
		t.Fatalf("expiry error = %v", err)
	}
	parts := strings.Split(token, ".")
	if parts[2][0] == 'A' {
		parts[2] = "B" + parts[2][1:]
	} else {
		parts[2] = "A" + parts[2][1:]
	}
	tampered := strings.Join(parts, ".")
	if _, err := codec.VerifySelection(tampered, expectation, []string{models[0].CatalogItemID}, now); !errors.Is(err, ErrInvalidClaim) {
		t.Fatalf("tamper error = %v", err)
	}
}

func TestDiscoveryClaimPreservesOnlyModelSpecificCapabilities(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	codec, err := NewClaimCodec(ClaimKeyset{
		ActiveKeyID: "active", Keys: map[string][]byte{"active": []byte(strings.Repeat("a", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	models, _, _, err := codec.Issue(testPlan(), testAuthorityDigest, testCredentialVerID, []AdapterModel{
		{ProviderModelID: "unknown-model", DisplayName: "Unknown Model"},
		{ProviderModelID: "described-model", DisplayName: "Described Model", Capabilities: []string{"image_input", "tools"}},
	}, now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if models[0].Capabilities != nil {
		t.Fatalf("unknown model capabilities = %v, want unset", models[0].Capabilities)
	}
	if got := strings.Join(models[1].Capabilities, ","); got != "image_input,tools" {
		t.Fatalf("model-specific capabilities = %q", got)
	}
}

func TestDiscoveryClaimRejectsOldCredentialVersionAndExposesCanonicalFieldDigest(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	codec, err := NewClaimCodec(ClaimKeyset{
		ActiveKeyID: "active", Keys: map[string][]byte{"active": []byte(strings.Repeat("a", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan := testPlan()
	plan.ConnectionFields = map[string]providercatalog.CanonicalConnectionValue{
		"region": {Kind: providercatalog.FieldText, Value: "us-east"},
		"batch":  {Kind: providercatalog.FieldInteger, Value: "4"},
	}
	models, token, _, err := codec.Issue(plan, testAuthorityDigest, "credential-version-old", []AdapterModel{{
		ProviderModelID: "model-a", DisplayName: "Model A",
	}}, now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := codec.VerifySelection(token, ClaimExpectation{
		NamespaceID: plan.NamespaceID, AuthorityDigest: testAuthorityDigest,
		CatalogRevision: plan.CatalogRevision, ProviderID: plan.ProviderID,
	}, []string{models[0].CatalogItemID}, now)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := providercatalog.CanonicalConnectionDigest(plan.ConnectionFields)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Binding.ConnectionDigest != digest || verified.Binding.CredentialVersion != "credential-version-old" {
		t.Fatalf("binding = %+v, digest = %s", verified.Binding, digest)
	}
	if verified.Binding.CredentialVersion == "credential-version-new" {
		t.Fatal("old discovery claim silently rebound to a new credential version")
	}
}
