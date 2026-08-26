package invitationmanagement

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

func TestFirstKeyPreparerSealsRevealableCredential(t *testing.T) {
	ids := []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
	}
	nextID := func() string {
		value := ids[0]
		ids = ids[1:]
		return value
	}
	revealKEK := accesscredential.KEKKeyring{
		ActiveVersion: "reveal-v1",
		Keys: map[string][]byte{
			"reveal-v1": []byte(strings.Repeat("r", 32)),
		},
	}
	preparer, err := NewAPIKeyFirstKeyPreparer(
		accesscredential.PepperKeyring{
			ActiveVersion: "pepper-v1",
			Keys: map[string][]byte{
				"pepper-v1": []byte(strings.Repeat("p", 32)),
			},
		},
		&revealKEK,
		nextID,
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(revealKEK.Close)
	t.Cleanup(preparer.Close)

	prepared, err := preparer.PrepareFirstKey(FirstKeyRequest{
		NamespaceID: "33333333-3333-4333-8333-333333333333",
		UserID:      "44444444-4444-4444-8444-444444444444",
		Name:        "Invited user",
		Now:         time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer zero(prepared.Plaintext)
	credential := prepared.Credential
	if len(credential.SecretCiphertext) == 0 || len(credential.CiphertextNonce) == 0 ||
		credential.KEKVersion != "reveal-v1" {
		t.Fatalf("automatic first-key credential envelope = ciphertext:%t nonce:%t version:%q",
			len(credential.SecretCiphertext) > 0, len(credential.CiphertextNonce) > 0, credential.KEKVersion)
	}
	plaintext, err := revealKEK.Open(
		accesscredential.Envelope{
			KeyVersion: credential.KEKVersion,
			Nonce:      credential.CiphertextNonce,
			Ciphertext: credential.SecretCiphertext,
		},
		accesscredential.APIKeyRevealAAD(
			string(prepared.Key.NamespaceID), string(prepared.Key.ID), string(credential.ID), credential.KID,
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer zero(plaintext)
	if !bytes.Equal(plaintext, prepared.Plaintext) {
		t.Fatal("reveal envelope does not contain the issued first-key secret")
	}
	assertFirstKeyAADIsolation(t, revealKEK, prepared)
}

func assertFirstKeyAADIsolation(
	t *testing.T,
	revealKEK accesscredential.KEKKeyring,
	prepared PreparedFirstKey,
) {
	t.Helper()
	credential := prepared.Credential
	for name, aad := range map[string][]byte{
		"namespace": accesscredential.APIKeyRevealAAD(
			"55555555-5555-4555-8555-555555555555", string(prepared.Key.ID), string(credential.ID), credential.KID,
		),
		"key": accesscredential.APIKeyRevealAAD(
			string(prepared.Key.NamespaceID), "55555555-5555-4555-8555-555555555555", string(credential.ID), credential.KID,
		),
		"credential": accesscredential.APIKeyRevealAAD(
			string(prepared.Key.NamespaceID), string(prepared.Key.ID), "55555555-5555-4555-8555-555555555555", credential.KID,
		),
		"kid": accesscredential.APIKeyRevealAAD(
			string(prepared.Key.NamespaceID), string(prepared.Key.ID), string(credential.ID), "other-kid",
		),
	} {
		if _, err := revealKEK.Open(accesscredential.Envelope{
			KeyVersion: credential.KEKVersion,
			Nonce:      credential.CiphertextNonce,
			Ciphertext: credential.SecretCiphertext,
		}, aad); !errors.Is(err, accesscredential.ErrInvalidEnvelope) {
			t.Fatalf("%s-scoped AAD opened another credential: %v", name, err)
		}
	}
}

func TestFirstKeyPreparerKeepsRevealStorageOptional(t *testing.T) {
	ids := []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
	}
	preparer, err := NewAPIKeyFirstKeyPreparer(
		accesscredential.PepperKeyring{
			ActiveVersion: "pepper-v1",
			Keys: map[string][]byte{
				"pepper-v1": []byte(strings.Repeat("p", 32)),
			},
		},
		nil,
		func() string {
			value := ids[0]
			ids = ids[1:]
			return value
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(preparer.Close)

	prepared, err := preparer.PrepareFirstKey(FirstKeyRequest{
		NamespaceID: "33333333-3333-4333-8333-333333333333",
		UserID:      "44444444-4444-4444-8444-444444444444",
		Now:         time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer zero(prepared.Plaintext)
	if len(prepared.Credential.SecretCiphertext) != 0 ||
		len(prepared.Credential.CiphertextNonce) != 0 || prepared.Credential.KEKVersion != "" {
		t.Fatal("first-key credential unexpectedly retained a reveal envelope")
	}
}
