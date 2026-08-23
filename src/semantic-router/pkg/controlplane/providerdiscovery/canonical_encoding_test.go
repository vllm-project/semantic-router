package providerdiscovery

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
)

func TestDiscoveryClaimRejectsNonCanonicalSignatureEncoding(t *testing.T) {
	codec, err := NewClaimCodec(ClaimKeyset{
		ActiveKeyID: "v1",
		Keys:        map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1_800_000_000, 0).UTC()
	_, token, _, err := codec.Issue(
		testPlan(),
		testAuthorityDigest,
		testCredentialVerID,
		[]AdapterModel{{ProviderModelID: "model-a", DisplayName: "Model A"}},
		now,
		time.Minute,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := codec.verify(token); err != nil {
		t.Fatal(err)
	}
	if _, err := codec.verify(signedtoken.Alias(t, token)); err == nil {
		t.Fatal("non-canonical signature encoding was accepted")
	}
}
