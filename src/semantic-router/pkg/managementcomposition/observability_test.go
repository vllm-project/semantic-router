package managementcomposition

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestObservabilityCursorKeysAreStableAndDomainSeparated(t *testing.T) {
	keyring := securitykeyring.Symmetric{
		ActiveVersion: "cursor-2",
		Keys: map[string][]byte{
			"cursor-1": bytes.Repeat([]byte{1}, 32),
			"cursor-2": bytes.Repeat([]byte{2}, 32),
		},
	}
	logs, err := deriveObservabilityCursorKey(keyring, "request-log")
	if err != nil {
		t.Fatal(err)
	}
	again, err := deriveObservabilityCursorKey(keyring, "request-log")
	if err != nil {
		t.Fatal(err)
	}
	audit, err := deriveObservabilityCursorKey(keyring, "audit-event")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(logs, again) {
		t.Fatal("cursor key derivation is not deterministic")
	}
	if bytes.Equal(logs, audit) {
		t.Fatal("request-log and audit cursor keys share one authority domain")
	}
}

func TestObservabilityCursorKeyRequiresActiveVersion(t *testing.T) {
	_, err := deriveObservabilityCursorKey(securitykeyring.Symmetric{
		ActiveVersion: "missing",
		Keys:          map[string][]byte{"retained": bytes.Repeat([]byte{1}, 32)},
	}, "request-log")
	if err == nil {
		t.Fatal("missing active cursor key was accepted")
	}
}
