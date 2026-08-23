package accesscredential

import (
	"errors"
	"strings"
	"testing"
)

func testPepperKeyring() PepperKeyring {
	return PepperKeyring{ActiveVersion: "pepper-2", Keys: map[string][]byte{
		"pepper-1": []byte(strings.Repeat("a", 32)),
		"pepper-2": []byte(strings.Repeat("b", 32)),
	}}
}

func TestIssueAndVerifyCredentialKinds(t *testing.T) {
	keyring := testPepperKeyring()
	for _, tc := range []struct {
		kind Kind
		id   string
	}{
		{KindAPIKey, "key-12345678"},
		{KindDelegation, "session-12345678"},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			issued, err := keyring.Issue(tc.kind, tc.id)
			if err != nil {
				t.Fatalf("Issue() error = %v", err)
			}
			if strings.Contains(issued.Plaintext, "=") {
				t.Fatalf("credential is not canonical base64url: %q", issued.Plaintext)
			}
			kind, id, err := PublicID(issued.Plaintext)
			if err != nil || kind != tc.kind || id != tc.id {
				t.Fatalf("PublicID() = (%q, %q, %v)", kind, id, err)
			}
			if err := keyring.Verify(issued.Plaintext, issued.Digest); err != nil {
				t.Fatalf("Verify() error = %v", err)
			}
		})
	}
}

func TestVerifyRejectsCrossKindTamperAndWrongPepper(t *testing.T) {
	keyring := testPepperKeyring()
	issued, err := keyring.Issue(KindAPIKey, "key-12345678")
	if err != nil {
		t.Fatal(err)
	}
	tamperedSuffix := "A"
	if strings.HasSuffix(issued.Plaintext, tamperedSuffix) {
		tamperedSuffix = "B"
	}

	cases := []struct {
		name      string
		presented string
		digest    Digest
		want      error
	}{
		{"tampered secret", issued.Plaintext[:len(issued.Plaintext)-1] + tamperedSuffix, issued.Digest, ErrInvalidCredential},
		{"cross kind", strings.Replace(issued.Plaintext, "vsr_", "vsd_", 1), issued.Digest, ErrInvalidCredential},
		{"wrong id", strings.Replace(issued.Plaintext, "key-12345678", "key-87654321", 1), issued.Digest, ErrInvalidCredential},
		{"unknown pepper", issued.Plaintext, func() Digest { d := issued.Digest; d.PepperVersion = "missing"; return d }(), ErrPepperUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := keyring.Verify(tc.presented, tc.digest); !errors.Is(err, tc.want) {
				t.Fatalf("Verify() error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestPublicIDRejectsNonCanonicalAndMalformedCredentials(t *testing.T) {
	for _, presented := range []string{
		"", "vsr_key_short", "vsr_key_12345678_extra", "api_key-12345678_secret",
		"vsr_key_with_under_score_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		"vsr_key-12345678_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
	} {
		if _, _, err := PublicID(presented); !errors.Is(err, ErrInvalidCredential) {
			t.Fatalf("PublicID(%q) error = %v", presented, err)
		}
	}
}

func TestPepperRotationKeepsPinnedVersionsVerifiable(t *testing.T) {
	old := PepperKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("1", 32))}}
	issued, err := old.Issue(KindAPIKey, "key-12345678")
	if err != nil {
		t.Fatal(err)
	}
	rotated := PepperKeyring{ActiveVersion: "v2", Keys: map[string][]byte{
		"v1": []byte(strings.Repeat("1", 32)),
		"v2": []byte(strings.Repeat("2", 32)),
	}}
	if err := rotated.Verify(issued.Plaintext, issued.Digest); err != nil {
		t.Fatalf("Verify() after rotation error = %v", err)
	}
}

func TestEnvelopeRoundTripRotationAndAADBinding(t *testing.T) {
	keyring := KEKKeyring{ActiveVersion: "kek-2", Keys: map[string][]byte{
		"kek-1": []byte(strings.Repeat("1", 32)),
		"kek-2": []byte(strings.Repeat("2", 32)),
	}}
	plaintext := []byte("secret response that must never be logged")
	aad := []byte("namespace:key:credential:operation")
	envelope, err := keyring.Seal(plaintext, aad)
	if err != nil {
		t.Fatalf("Seal() error = %v", err)
	}
	opened, err := keyring.Open(envelope, aad)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if string(opened) != string(plaintext) {
		t.Fatalf("Open() = %q", opened)
	}
	if _, err := keyring.Open(envelope, []byte("different operation")); !errors.Is(err, ErrInvalidEnvelope) {
		t.Fatalf("Open() wrong AAD error = %v", err)
	}

	rotated := KEKKeyring{ActiveVersion: "kek-3", Keys: map[string][]byte{
		"kek-2": []byte(strings.Repeat("2", 32)),
		"kek-3": []byte(strings.Repeat("3", 32)),
	}}
	if _, err := rotated.Open(envelope, aad); err != nil {
		t.Fatalf("Open() after rotation error = %v", err)
	}
	withoutOld := KEKKeyring{ActiveVersion: "kek-3", Keys: map[string][]byte{
		"kek-3": []byte(strings.Repeat("3", 32)),
	}}
	if _, err := withoutOld.Open(envelope, aad); !errors.Is(err, ErrKEKUnavailable) {
		t.Fatalf("Open() without pinned key error = %v", err)
	}
}
