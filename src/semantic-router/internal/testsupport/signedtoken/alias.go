// Package signedtoken provides focused helpers for testing canonical signed-token wire encodings.
package signedtoken

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
)

const rawURLAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

// Alias returns a non-canonical base64url signature segment that decodes to
// the same 256-bit value as the canonical signature in token.
func Alias(t testing.TB, token string) string {
	t.Helper()
	separator := strings.LastIndexByte(token, '.')
	if separator < 0 || separator == len(token)-1 {
		t.Fatalf("token has no signature segment: %q", token)
	}
	signature := token[separator+1:]
	index := strings.IndexByte(rawURLAlphabet, signature[len(signature)-1])
	if index < 0 || index%4 != 0 {
		t.Fatalf("signature is not canonical 256-bit base64url: %q", signature)
	}
	alias := signature[:len(signature)-1] + string(rawURLAlphabet[index+1])
	canonicalBytes, canonicalErr := base64.RawURLEncoding.DecodeString(signature)
	aliasBytes, aliasErr := base64.RawURLEncoding.DecodeString(alias)
	if canonicalErr != nil || aliasErr != nil || len(canonicalBytes) != 32 || !bytes.Equal(canonicalBytes, aliasBytes) ||
		base64.RawURLEncoding.EncodeToString(aliasBytes) != signature {
		t.Fatalf("failed to construct a non-canonical signature alias for %q", signature)
	}
	return token[:separator+1] + alias
}
