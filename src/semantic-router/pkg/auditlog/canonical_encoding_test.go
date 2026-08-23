package auditlog

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
)

func TestCursorRejectsNonCanonicalSignatureEncoding(t *testing.T) {
	codec, err := NewCursorCodec([]byte(strings.Repeat("k", 32)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.Close)
	token, err := codec.encode(cursorValue{
		Version:     1,
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		QueryDigest: strings.Repeat("a", 64),
		CreatedAt:   1,
		EventID:     "22222222-2222-4222-8222-222222222222",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := codec.decode(token); err != nil {
		t.Fatal(err)
	}
	if _, err := codec.decode(signedtoken.Alias(t, token)); err == nil {
		t.Fatal("non-canonical signature encoding was accepted")
	}
}
