package routingmanagement

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestCursorRejectsNonCanonicalSignatureEncoding(t *testing.T) {
	codec, err := newRoutingCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.close)
	token, err := codec.encode(routingCursorPayload{})
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
