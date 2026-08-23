package postgres

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
)

func TestParseServiceCredentialAcceptsCanonicalBase64URLUnderscores(t *testing.T) {
	secret := make([]byte, 32)
	for index := range secret {
		secret[index] = 0xff
	}
	encodedSecret := base64.RawURLEncoding.EncodeToString(secret)
	if !strings.Contains(encodedSecret, "_") {
		t.Fatal("fixture does not contain a URL-safe base64 underscore")
	}
	publicID, parsed, ok := parseServiceCredential("vsm_10000000-0000-4000-8000-000000000001_" + encodedSecret)
	if !ok || publicID != "10000000-0000-4000-8000-000000000001" || !bytes.Equal(parsed, secret) {
		t.Fatalf("canonical credential parse = (%q, %d bytes, %t)", publicID, len(parsed), ok)
	}
	zeroBytes(parsed)
}

func TestParseServiceCredentialRejectsWrongLengthAndNonCanonicalEncoding(t *testing.T) {
	const prefix = "vsm_10000000-0000-4000-8000-000000000001_"
	short := base64.RawURLEncoding.EncodeToString(make([]byte, 31))
	if _, parsed, ok := parseServiceCredential(prefix + short); ok || parsed != nil {
		t.Fatal("wrong-length service credential was accepted")
	}
	canonical := base64.RawURLEncoding.EncodeToString(make([]byte, 32))
	nonCanonical := canonical[:len(canonical)-1] + "B"
	if _, parsed, ok := parseServiceCredential(prefix + nonCanonical); ok || parsed != nil {
		t.Fatal("non-canonical service credential was accepted")
	}
}
