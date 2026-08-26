package usageledger

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
)

func TestLogCursorRejectsNonCanonicalSignatureEncoding(t *testing.T) {
	codec, err := NewLogCursorCodec([]byte(strings.Repeat("k", 32)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.Close)
	token, err := codec.encode(logCursor{
		Version:     1,
		NamespaceID: testNamespaceID,
		QueryDigest: strings.Repeat("a", 64),
		Start:       time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC).UnixNano(),
		End:         time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC).UnixNano(),
		OccurredAt:  1,
		EventID:     testEventID("canonical-signature"),
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
