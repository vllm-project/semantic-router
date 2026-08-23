package providercatalog

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestCursorCodecSignsActiveAndVerifiesRetainedVersion(t *testing.T) {
	keys := map[string][]byte{
		"cursor-v1": []byte(strings.Repeat("1", 32)),
		"cursor-v2": []byte(strings.Repeat("2", 32)),
	}
	oldCodec, err := newCursorCodec(securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	currentCodec, err := newCursorCodec(securitykeyring.Symmetric{ActiveVersion: "cursor-v2", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	want := listCursor{
		Version: 1, CatalogRevision: "sha256:" + strings.Repeat("a", 64),
		QueryDigest: strings.Repeat("b", 64), Order: 7, ProviderID: "provider-a",
	}
	encoded, err := oldCodec.encode(want)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(encoded, "pcat.cursor-v1.") {
		t.Fatalf("cursor wire = %q", encoded)
	}
	got, err := currentCodec.decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("decoded cursor = %#v, want %#v", got, want)
	}
}

func TestCursorCodecRejectsUnknownRetiredAndUnversionedWire(t *testing.T) {
	oldCodec, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v1", Keys: map[string][]byte{"cursor-v1": []byte(strings.Repeat("1", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := oldCodec.encode(listCursor{
		Version: 1, CatalogRevision: "sha256:" + strings.Repeat("a", 64),
		QueryDigest: strings.Repeat("b", 64), ProviderID: "provider-a",
	})
	if err != nil {
		t.Fatal(err)
	}
	currentCodec, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v2", Keys: map[string][]byte{"cursor-v2": []byte(strings.Repeat("2", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []string{
		encoded,
		strings.Replace(encoded, "pcat.cursor-v1.", "pcat.unknown-v9.", 1),
		strings.TrimPrefix(encoded, "pcat.cursor-v1."),
	} {
		if _, err := currentCodec.decode(invalid); err == nil {
			t.Fatalf("invalid cursor %q was accepted", invalid)
		}
	}
}
