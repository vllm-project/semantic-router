package policybulk

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestOperationCursorRoundTripRotationFiltersAndTamperRejection(t *testing.T) {
	codec, err := newOperationCursorCodec(securitykeyring.Symmetric{ActiveVersion: "v2", Keys: map[string][]byte{
		"v1": []byte(strings.Repeat("a", 32)), "v2": []byte(strings.Repeat("b", 32)),
	}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.close)
	want := operationCursorPayload{
		NamespaceID:       "11111111-1111-4111-8111-111111111111",
		OriginPrincipalID: "22222222-2222-4222-8222-222222222222",
		Kind:              AccessBindingOperationKind, State: OperationRunning,
		CreatedAt: time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC),
		ID:        "33333333-3333-4333-8333-333333333333",
	}
	encoded, err := codec.encode(want)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(encoded, "opb.v2.") {
		t.Fatalf("cursor version = %q", encoded)
	}
	got, err := codec.decode(encoded)
	if err != nil {
		t.Fatal(err)
	}
	want.Version = 1
	if got != want {
		t.Fatalf("decoded cursor = %#v, want %#v", got, want)
	}
	last := encoded[len(encoded)-1]
	replacement := byte('A')
	if last == replacement {
		replacement = 'B'
	}
	if _, err := codec.decode(encoded[:len(encoded)-1] + string(replacement)); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("tampered cursor error = %v", err)
	}
}

func TestOperationCursorRejectsInvalidKeyring(t *testing.T) {
	_, err := newOperationCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte("short")},
	})
	if err == nil {
		t.Fatal("expected invalid keyring error")
	}
}
