package providercredential

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

const (
	testNamespaceID  = "11111111-1111-4111-8111-111111111111"
	testCredentialID = "22222222-2222-4222-8222-222222222222"
	testVersionID    = "33333333-3333-4333-8333-333333333333"
)

func TestNormalizeOrigin(t *testing.T) {
	for input, expected := range map[string]string{
		"HTTPS://API.Example.com:443/v1/": "https://api.example.com/v1",
		"http://[2001:db8::1]:80/":        "http://[2001:db8::1]",
		"https://api.example.com":         "https://api.example.com",
	} {
		actual, err := NormalizeOrigin(input)
		if err != nil || actual != expected {
			t.Fatalf("NormalizeOrigin(%q) = %q, %v; want %q", input, actual, err, expected)
		}
	}
	for _, invalid := range []string{
		"https://user:secret@example.com", "https://example.com/a/../b",
		"https://example.com/a%2fb", "https://example.com/?key=secret", "file:///tmp/socket",
	} {
		if _, err := NormalizeOrigin(invalid); err == nil {
			t.Fatalf("NormalizeOrigin(%q) succeeded", invalid)
		}
	}
}

func TestCodecBindsSecretToCredentialProviderOriginAndVersion(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 0, 0, 0, time.UTC)
	codec := testCodec()
	credential := activeCredential(now)
	version, err := codec.Seal(credential, testVersionID, []byte("provider-secret"), now)
	if err != nil {
		t.Fatal(err)
	}
	credential.ActiveVersionID = stringPointer(version.ID)
	opened, err := codec.OpenActive(credential, version, "openai", "https://api.example.com/v1", now.Add(time.Second))
	if err != nil || string(opened) != "provider-secret" {
		t.Fatalf("OpenActive() = %q, %v", opened, err)
	}
	Zero(opened)
	if _, err := codec.OpenActive(credential, version, "anthropic", credential.NormalizedOrigin, now); !errors.Is(err, ErrMismatch) {
		t.Fatalf("provider mismatch error = %v", err)
	}
	tampered := credential
	tampered.NormalizedOrigin = "https://other.example.com"
	if _, err := codec.OpenActive(tampered, version, tampered.ProviderID, tampered.NormalizedOrigin, now); err == nil {
		t.Fatal("tampered origin decrypted secret")
	}
	tampered = credential
	tampered.CredentialAdapterID = "x-api-key"
	if _, err := codec.OpenActive(tampered, version, tampered.ProviderID, tampered.NormalizedOrigin, now); err == nil {
		t.Fatal("tampered credential adapter decrypted secret")
	}
	tampered = credential
	tampered.CatalogRevision = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	if _, err := codec.OpenActive(tampered, version, tampered.ProviderID, tampered.NormalizedOrigin, now); err == nil {
		t.Fatal("tampered provider catalog revision decrypted secret")
	}
}

func TestRotationAllowsOnlyBoundedPinnedRetirement(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 0, 0, 0, time.UTC)
	codec := testCodec()
	credential := activeCredential(now)
	version, testRotationAllowsOnlyBoundedPinnedRetirementErr := codec.Seal(credential, testVersionID, []byte("old-secret"), now)
	if testRotationAllowsOnlyBoundedPinnedRetirementErr != nil {
		t.Fatal(testRotationAllowsOnlyBoundedPinnedRetirementErr)
	}
	newVersion := "44444444-4444-4444-8444-444444444444"
	credential.ActiveVersionID = &newVersion
	retireAt := now.Add(time.Minute)
	version.Status = VersionRetiring
	version.ExpiresAt = &retireAt
	if _, err := codec.OpenActive(credential, version, credential.ProviderID, credential.NormalizedOrigin, now); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("OpenActive retiring error = %v", err)
	}
	secret, testRotationAllowsOnlyBoundedPinnedRetirementErr := codec.OpenPinned(credential, version, credential.ProviderID, credential.NormalizedOrigin, now)
	if testRotationAllowsOnlyBoundedPinnedRetirementErr != nil || string(secret) != "old-secret" {
		t.Fatalf("OpenPinned() = %q, %v", secret, testRotationAllowsOnlyBoundedPinnedRetirementErr)
	}
	Zero(secret)
	if _, err := codec.OpenPinned(credential, version, credential.ProviderID, credential.NormalizedOrigin, retireAt); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expired OpenPinned error = %v", err)
	}
}

func activeCredential(now time.Time) Credential {
	return Credential{
		ID: testCredentialID, NamespaceID: testNamespaceID, Name: "Primary",
		ProviderID: "openai", CredentialMode: ModeRequired, CredentialAdapterID: "bearer",
		CatalogRevision:  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		NormalizedOrigin: "https://api.example.com/v1",
		Status:           StatusActive, ActiveVersionID: stringPointer(testVersionID), Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
}

func testCodec() Codec {
	return Codec{Keyring: accesscredential.KEKKeyring{
		ActiveVersion: "provider-kek-1",
		Keys:          map[string][]byte{"provider-kek-1": []byte(strings.Repeat("p", 32))},
	}}
}

func stringPointer(value string) *string { return &value }
