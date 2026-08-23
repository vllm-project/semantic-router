package agentmanagement

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestSignedCodecRejectsNonCanonicalSignatureEncoding(t *testing.T) {
	codec, err := newSignedCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.close)
	token, err := codec.envelope("cursor", []byte(`{"v":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := codec.open("cursor", token, 4096); err != nil {
		t.Fatal(err)
	}
	if _, err := codec.open("cursor", signedtoken.Alias(t, token), 4096); err == nil {
		t.Fatal("non-canonical signature encoding was accepted")
	}
}
