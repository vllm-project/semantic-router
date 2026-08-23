package management

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestCursorCodecSignsWithActiveVersionAndVerifiesRetainedVersion(t *testing.T) {
	keys := map[string][]byte{
		"cursor-v1": []byte(strings.Repeat("o", 32)),
		"cursor-v2": []byte(strings.Repeat("n", 32)),
	}
	oldCodec, err := newCursorCodec(securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	rotatedCodec, err := newCursorCodec(securitykeyring.Symmetric{ActiveVersion: "cursor-v2", Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	want := listCursor{
		Version: 1, NamespaceID: serviceNamespaceID, ProviderID: "provider-a",
		Status: providercredential.StatusActive, AfterStatus: providercredential.StatusActive,
		AfterID: serviceCredentialID,
	}
	encoded, err := oldCodec.encode(want)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(encoded, "pcc.cursor-v1.") {
		t.Fatalf("cursor wire = %q", encoded)
	}
	got, err := rotatedCodec.decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("decoded cursor = %#v, want %#v", got, want)
	}
}

func TestCursorCodecRejectsUnknownOrRetiredVersion(t *testing.T) {
	oldCodec, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v1", Keys: map[string][]byte{"cursor-v1": []byte(strings.Repeat("o", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := oldCodec.encode(listCursor{
		Version: 1, NamespaceID: serviceNamespaceID, AfterStatus: providercredential.StatusActive,
		AfterID: serviceCredentialID,
	})
	if err != nil {
		t.Fatal(err)
	}
	currentCodec, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v2", Keys: map[string][]byte{"cursor-v2": []byte(strings.Repeat("n", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := currentCodec.decode(encoded); err == nil {
		t.Fatal("cursor signed by a retired key version was accepted")
	}
	unknown := strings.Replace(encoded, "pcc.cursor-v1.", "pcc.unknown-v9.", 1)
	if _, err := currentCodec.decode(unknown); err == nil {
		t.Fatal("cursor with unknown key version was accepted")
	}
}
