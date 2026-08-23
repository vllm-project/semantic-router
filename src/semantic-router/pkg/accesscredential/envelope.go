package accesscredential

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
	"fmt"
	"strings"
)

// Envelope is safe to persist only alongside its application-specific AAD.
// Ciphertext remains secret material and must never be serialized by list,
// audit, log, usage, metric, trace, or error APIs.
type Envelope struct {
	KeyVersion string
	Nonce      []byte
	Ciphertext []byte
}

// KEKKeyring provides versioned AES-256 key-encryption keys. Separate keyring
// instances are used for reveal, provider credentials, and temporary response
// envelopes so compromise and rotation domains stay independent.
type KEKKeyring struct {
	ActiveVersion string
	Keys          map[string][]byte
}

func (k KEKKeyring) Validate() error {
	if strings.TrimSpace(k.ActiveVersion) == "" || k.ActiveVersion != strings.TrimSpace(k.ActiveVersion) {
		return errors.New("active KEK version is required and canonical")
	}
	if len(k.Keys) == 0 {
		return errors.New("at least one KEK is required")
	}
	for version, key := range k.Keys {
		if strings.TrimSpace(version) == "" || version != strings.TrimSpace(version) {
			return errors.New("KEK versions must be non-empty and canonical")
		}
		if len(key) != 32 {
			return fmt.Errorf("KEK %q must be exactly 256 bits", version)
		}
	}
	if _, ok := k.Keys[k.ActiveVersion]; !ok {
		return fmt.Errorf("active KEK version %q is absent", k.ActiveVersion)
	}
	return nil
}

func (k KEKKeyring) Seal(plaintext, aad []byte) (Envelope, error) {
	if err := k.Validate(); err != nil {
		return Envelope{}, err
	}
	aead, err := newGCM(k.Keys[k.ActiveVersion])
	if err != nil {
		return Envelope{}, err
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return Envelope{}, fmt.Errorf("generate envelope nonce: %w", err)
	}
	ciphertext := aead.Seal(nil, nonce, plaintext, aad)
	return Envelope{KeyVersion: k.ActiveVersion, Nonce: nonce, Ciphertext: ciphertext}, nil
}

func (k KEKKeyring) Open(envelope Envelope, aad []byte) ([]byte, error) {
	if err := k.Validate(); err != nil {
		return nil, err
	}
	key, ok := k.Keys[envelope.KeyVersion]
	if !ok {
		return nil, ErrKEKUnavailable
	}
	aead, err := newGCM(key)
	if err != nil {
		return nil, err
	}
	if len(envelope.Nonce) != aead.NonceSize() || len(envelope.Ciphertext) < aead.Overhead() {
		return nil, ErrInvalidEnvelope
	}
	plaintext, err := aead.Open(nil, envelope.Nonce, envelope.Ciphertext, aad)
	if err != nil {
		return nil, ErrInvalidEnvelope
	}
	return plaintext, nil
}

var (
	ErrKEKUnavailable  = errors.New("envelope KEK is unavailable")
	ErrInvalidEnvelope = errors.New("invalid encrypted envelope")
)

func newGCM(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("initialize envelope cipher: %w", err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("initialize envelope AEAD: %w", err)
	}
	return aead, nil
}
