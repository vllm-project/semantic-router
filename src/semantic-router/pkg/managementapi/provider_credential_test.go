package managementapi

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestProviderCredentialRepresentationsCannotSerializeSecretInternals(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	value := ProviderCredential{
		CredentialID: "33333333-3333-4333-8333-333333333333",
		Name:         "Primary", ProviderID: "provider-a", CatalogRevision: "sha256:" + strings.Repeat("a", 64),
		NormalizedOrigin: "https://api.example.com/v1", Status: "active",
		Revision: 3, CreatedAt: now, UpdatedAt: now,
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"secret", "ciphertext", "nonce", "kek", "adapter", "credentialMode", "activeVersion", "namespaceId",
	} {
		if strings.Contains(strings.ToLower(text), strings.ToLower(forbidden)) {
			t.Fatalf("safe ProviderCredential exposed %q: %s", forbidden, text)
		}
	}
	for _, required := range []string{"credentialId", "providerId", "normalizedOrigin", "revision"} {
		if !strings.Contains(text, `"`+required+`"`) {
			t.Fatalf("safe ProviderCredential omitted %q: %s", required, text)
		}
	}
}

func TestResourceMutationReceiptIsImmutableAndExclusive(t *testing.T) {
	replayed := true
	receipt := NewResourceMutationReceipt(
		"provider_credential", "33333333-3333-4333-8333-333333333333", 1, &replayed,
	)
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, expected := range []string{`"kind":"provider_credential"`, `"revision":1`, `"replayed":true`} {
		if !strings.Contains(text, expected) {
			t.Fatalf("receipt %s omitted %s", text, expected)
		}
	}
	for _, forbidden := range []string{"operation", "desiredRevision", "data", "secret"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("resource receipt exposed %q: %s", forbidden, text)
		}
	}
}

func TestOperationMutationReceiptIsExclusiveAndDefensivelyCopiesRevision(t *testing.T) {
	revision := uint64(17)
	replayed := true
	receipt := NewOperationMutationReceipt("33333333-3333-4333-8333-333333333333", &revision, &replayed)
	revision = 99
	if receipt.Resource != nil || receipt.Operation == nil || receipt.Operation.DesiredRevision == nil ||
		*receipt.Operation.DesiredRevision != 17 || receipt.Idempotency == nil || !receipt.Idempotency.Replayed {
		t.Fatalf("operation receipt = %#v", receipt)
	}
}

func TestOpenAPIProviderCredentialSchemasAreExactAndRedacted(t *testing.T) {
	schemas := GenerateOpenAPI().Components.Schemas
	credential := schemas["ProviderCredential"]
	for _, forbidden := range []string{"secret", "activeVersionId", "credentialAdapterId", "credentialMode", "ciphertext"} {
		if _, found := credential.Properties[forbidden]; found {
			t.Fatalf("ProviderCredential schema exposes %q", forbidden)
		}
	}
	mutation := schemas["MutationReceipt"]
	if len(mutation.OneOf) != 2 {
		t.Fatalf("MutationReceipt oneOf = %#v", mutation.OneOf)
	}
	for _, branch := range mutation.OneOf {
		if branch.AdditionalProperties == nil || *branch.AdditionalProperties {
			t.Fatalf("MutationReceipt branch accepts unknown fields: %#v", branch)
		}
	}
}
