package invitationmanagement

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

const tokenTestInvitationID = "11111111-1111-4111-8111-111111111111"

func TestInvitationTokenIsCanonicalAndPepperPinned(t *testing.T) {
	codec, err := newTokenCodec(accesscredential.PepperKeyring{ActiveVersion: "invite-v2", Keys: map[string][]byte{
		"invite-v1": []byte(strings.Repeat("a", 32)), "invite-v2": []byte(strings.Repeat("b", 32)),
	}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(codec.Close)
	token, digest, version, err := codec.Issue(tokenTestInvitationID)
	if err != nil || version != "invite-v2" || len(digest) != 32 {
		t.Fatalf("Issue() = version %q, digest %d, error %v", version, len(digest), err)
	}
	if id, err := tokenInvitationID(token); err != nil || id != tokenTestInvitationID {
		t.Fatalf("tokenInvitationID() = %q, %v", id, err)
	}
	if err := codec.Verify(token, digest, version); err != nil {
		t.Fatalf("Verify() error = %v", err)
	}
	tampered := token[:len(token)-1] + "A"
	if strings.HasSuffix(token, "A") {
		tampered = token[:len(token)-1] + "B"
	}
	if err := codec.Verify(tampered, digest, version); !errors.Is(err, ErrInvalidToken) {
		t.Fatalf("tampered Verify() error = %v", err)
	}
	if err := codec.Verify(token, digest, "retired"); !errors.Is(err, ErrPepperUnavailable) {
		t.Fatalf("retired pepper Verify() error = %v", err)
	}
}

func TestInvitationTokenRejectsNonCanonicalForms(t *testing.T) {
	for _, value := range []string{
		"", "vsi_" + tokenTestInvitationID, "vsi_NOT-A-UUID_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		"vsi_" + strings.ToUpper("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa") + "_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		"vsi_" + tokenTestInvitationID + "_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
	} {
		if _, err := tokenInvitationID(value); !errors.Is(err, ErrInvalidToken) {
			t.Fatalf("tokenInvitationID(%q) error = %v", value, err)
		}
	}
}
