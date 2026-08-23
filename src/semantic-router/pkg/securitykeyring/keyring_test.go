package securitykeyring

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func encoded(value []byte) string { return base64.RawURLEncoding.EncodeToString(value) }

func TestParseSymmetricStrictVersionedDocument(t *testing.T) {
	payload := fmt.Sprintf(`{"activeVersion":"v2","keys":[{"version":"v1","key":"%s"},{"version":"v2","key":"%s"}]}`,
		encoded([]byte(strings.Repeat("1", 32))), encoded([]byte(strings.Repeat("2", 32))))
	keyring, err := ParseSymmetric([]byte(payload), 32)
	if err != nil {
		t.Fatalf("ParseSymmetric() error = %v", err)
	}
	if keyring.ActiveVersion != "v2" || len(keyring.Keys) != 2 {
		t.Fatalf("ParseSymmetric() = %+v", keyring)
	}
}

func TestParseSymmetricRejectsUnknownDuplicateAndNonCanonicalFields(t *testing.T) {
	valid := encoded([]byte(strings.Repeat("1", 32)))
	for _, payload := range []string{
		fmt.Sprintf(`{"activeVersion":"v1","unknown":true,"keys":[{"version":"v1","key":"%s"}]}`, valid),
		fmt.Sprintf(`{"activeVersion":"v1","keys":[{"version":"v1","key":"%s"},{"version":"v1","key":"%s"}]}`, valid, valid),
		fmt.Sprintf(`{"activeVersion":"missing","keys":[{"version":"v1","key":"%s"}]}`, valid),
		`{"activeVersion":"v1","keys":[{"version":"v1","key":"not-base64"}]}`,
	} {
		if _, err := ParseSymmetric([]byte(payload), 32); !errors.Is(err, ErrInvalidKeyring) {
			t.Fatalf("ParseSymmetric(%s) error = %v", payload, err)
		}
	}
}

func TestParseSigningAllowsRetainedPublicVerificationKeys(t *testing.T) {
	oldPublic, _, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	activePublic, activePrivate, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	payload := fmt.Sprintf(`{"activeVersion":"sig-2","keys":[`+
		`{"version":"sig-1","publicKey":"%s"},`+
		`{"version":"sig-2","publicKey":"%s","privateKey":"%s"}]}`,
		encoded(oldPublic), encoded(activePublic), encoded(activePrivate.Seed()))
	keyring, err := ParseSigning([]byte(payload))
	if err != nil {
		t.Fatalf("ParseSigning() error = %v", err)
	}
	if len(keyring.Public) != 2 || len(keyring.Private) != 1 || keyring.Private["sig-2"] == nil {
		t.Fatalf("ParseSigning() = %+v", keyring)
	}
}

func TestParseSigningRejectsMismatchedAndMissingActivePrivateKey(t *testing.T) {
	public, private, _ := ed25519.GenerateKey(nil)
	otherPublic, _, _ := ed25519.GenerateKey(nil)
	for _, payload := range []string{
		fmt.Sprintf(`{"activeVersion":"sig-1","keys":[{"version":"sig-1","publicKey":"%s"}]}`, encoded(public)),
		fmt.Sprintf(`{"activeVersion":"sig-1","keys":[{"version":"sig-1","publicKey":"%s","privateKey":"%s"}]}`, encoded(otherPublic), encoded(private.Seed())),
	} {
		if _, err := ParseSigning([]byte(payload)); !errors.Is(err, ErrInvalidKeyring) {
			t.Fatalf("ParseSigning() error = %v", err)
		}
	}
}
