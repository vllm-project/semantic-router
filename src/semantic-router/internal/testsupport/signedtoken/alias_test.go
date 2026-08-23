package signedtoken

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
)

func TestAliasPreservesDecodedSignatureButChangesWireEncoding(t *testing.T) {
	signature := bytes.Repeat([]byte{0x5a}, 32)
	canonical := base64.RawURLEncoding.EncodeToString(signature)
	token := "token.v1.payload." + canonical
	aliasToken := Alias(t, token)
	alias := aliasToken[strings.LastIndexByte(aliasToken, '.')+1:]
	if alias == canonical {
		t.Fatal("alias did not change the signature wire encoding")
	}
	decoded, err := base64.RawURLEncoding.DecodeString(alias)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(decoded, signature) {
		t.Fatalf("alias decoded to %x, want %x", decoded, signature)
	}
	if got := base64.RawURLEncoding.EncodeToString(decoded); got != canonical {
		t.Fatalf("canonical re-encoding = %q, want %q", got, canonical)
	}
}
