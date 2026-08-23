// Package securitykeyring loads strict, versioned deployment keyrings from
// secret files or environment references. It never accepts inline Router YAML.
package securitykeyring

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const maxKeyringBytes = 1 << 20

var ErrInvalidKeyring = errors.New("invalid security keyring")

type Source struct {
	File string
	Env  string
}

func (s Source) Read() ([]byte, error) {
	if (s.File == "") == (s.Env == "") {
		return nil, fmt.Errorf("%w: exactly one file or environment source is required", ErrInvalidKeyring)
	}
	if s.File != "" {
		file, err := os.Open(s.File)
		if err != nil {
			return nil, errors.New("read keyring secret file")
		}
		defer file.Close()
		payload, err := io.ReadAll(io.LimitReader(file, maxKeyringBytes+1))
		if err != nil {
			return nil, errors.New("read keyring secret file")
		}
		if len(payload) > maxKeyringBytes {
			return nil, fmt.Errorf("%w: keyring exceeds %d bytes", ErrInvalidKeyring, maxKeyringBytes)
		}
		return payload, nil
	}
	value, ok := os.LookupEnv(s.Env)
	if !ok {
		return nil, errors.New("keyring environment source is not set")
	}
	if len(value) > maxKeyringBytes {
		return nil, fmt.Errorf("%w: keyring exceeds %d bytes", ErrInvalidKeyring, maxKeyringBytes)
	}
	return []byte(value), nil
}

// Symmetric contains decoded versioned secret bytes. ActiveVersion is used for
// new HMACs/envelopes; retained versions verify or decrypt existing data.
type Symmetric struct {
	ActiveVersion string
	Keys          map[string][]byte
}

type symmetricDocument struct {
	ActiveVersion string                 `json:"activeVersion"`
	Keys          []symmetricKeyDocument `json:"keys"`
}

type symmetricKeyDocument struct {
	Version string `json:"version"`
	Key     string `json:"key"`
}

func ParseSymmetric(payload []byte, requiredBytes int) (Symmetric, error) {
	if requiredBytes < 16 {
		return Symmetric{}, fmt.Errorf("%w: required key length is unsafe", ErrInvalidKeyring)
	}
	var document symmetricDocument
	if err := decodeStrict(payload, &document); err != nil {
		return Symmetric{}, err
	}
	if !canonicalVersion(document.ActiveVersion) || len(document.Keys) == 0 {
		return Symmetric{}, fmt.Errorf("%w: activeVersion and keys are required", ErrInvalidKeyring)
	}
	result := Symmetric{ActiveVersion: document.ActiveVersion, Keys: make(map[string][]byte, len(document.Keys))}
	for _, item := range document.Keys {
		if !canonicalVersion(item.Version) {
			return Symmetric{}, fmt.Errorf("%w: key version is invalid", ErrInvalidKeyring)
		}
		if _, duplicate := result.Keys[item.Version]; duplicate {
			return Symmetric{}, fmt.Errorf("%w: duplicate key version", ErrInvalidKeyring)
		}
		decoded, err := base64.RawURLEncoding.DecodeString(item.Key)
		if err != nil || base64.RawURLEncoding.EncodeToString(decoded) != item.Key || len(decoded) != requiredBytes {
			return Symmetric{}, fmt.Errorf("%w: key %q must be canonical base64url with exactly %d bytes", ErrInvalidKeyring, item.Version, requiredBytes)
		}
		result.Keys[item.Version] = decoded
	}
	if _, ok := result.Keys[result.ActiveVersion]; !ok {
		return Symmetric{}, fmt.Errorf("%w: active version is absent", ErrInvalidKeyring)
	}
	return result, nil
}

type Signing struct {
	ActiveVersion string
	Private       map[string]ed25519.PrivateKey
	Public        map[string]ed25519.PublicKey
}

type signingDocument struct {
	ActiveVersion string               `json:"activeVersion"`
	Keys          []signingKeyDocument `json:"keys"`
}

type signingKeyDocument struct {
	Version    string `json:"version"`
	PrivateKey string `json:"privateKey,omitempty"`
	PublicKey  string `json:"publicKey"`
}

func ParseSigning(payload []byte) (Signing, error) {
	var document signingDocument
	if err := decodeStrict(payload, &document); err != nil {
		return Signing{}, err
	}
	if !canonicalVersion(document.ActiveVersion) || len(document.Keys) == 0 {
		return Signing{}, fmt.Errorf("%w: activeVersion and keys are required", ErrInvalidKeyring)
	}
	result := Signing{
		ActiveVersion: document.ActiveVersion,
		Private:       make(map[string]ed25519.PrivateKey),
		Public:        make(map[string]ed25519.PublicKey, len(document.Keys)),
	}
	for _, item := range document.Keys {
		if !canonicalVersion(item.Version) {
			return Signing{}, fmt.Errorf("%w: signing key version is invalid", ErrInvalidKeyring)
		}
		if _, duplicate := result.Public[item.Version]; duplicate {
			return Signing{}, fmt.Errorf("%w: duplicate signing key version", ErrInvalidKeyring)
		}
		publicKey, err := decodeCanonical(item.PublicKey, ed25519.PublicKeySize)
		if err != nil {
			return Signing{}, fmt.Errorf("%w: public key %q: %w", ErrInvalidKeyring, item.Version, err)
		}
		result.Public[item.Version] = ed25519.PublicKey(publicKey)
		if item.PrivateKey == "" {
			continue
		}
		privateBytes, err := base64.RawURLEncoding.DecodeString(item.PrivateKey)
		if err != nil || base64.RawURLEncoding.EncodeToString(privateBytes) != item.PrivateKey {
			return Signing{}, fmt.Errorf("%w: private key %q is not canonical base64url", ErrInvalidKeyring, item.Version)
		}
		var privateKey ed25519.PrivateKey
		switch len(privateBytes) {
		case ed25519.SeedSize:
			privateKey = ed25519.NewKeyFromSeed(privateBytes)
		case ed25519.PrivateKeySize:
			privateKey = ed25519.PrivateKey(append([]byte(nil), privateBytes...))
		default:
			return Signing{}, fmt.Errorf("%w: private key %q has invalid length", ErrInvalidKeyring, item.Version)
		}
		if !bytes.Equal(privateKey.Public().(ed25519.PublicKey), publicKey) {
			return Signing{}, fmt.Errorf("%w: signing key %q public/private mismatch", ErrInvalidKeyring, item.Version)
		}
		result.Private[item.Version] = privateKey
	}
	if _, ok := result.Public[result.ActiveVersion]; !ok {
		return Signing{}, fmt.Errorf("%w: active signing version is absent", ErrInvalidKeyring)
	}
	if _, ok := result.Private[result.ActiveVersion]; !ok {
		return Signing{}, fmt.Errorf("%w: active signing version has no private key", ErrInvalidKeyring)
	}
	return result, nil
}

func decodeStrict(payload []byte, destination any) error {
	if len(bytes.TrimSpace(payload)) == 0 {
		return fmt.Errorf("%w: document is empty", ErrInvalidKeyring)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("%w: decode document", ErrInvalidKeyring)
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return fmt.Errorf("%w: document contains trailing JSON", ErrInvalidKeyring)
	}
	return nil
}

func decodeCanonical(value string, size int) ([]byte, error) {
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil || base64.RawURLEncoding.EncodeToString(decoded) != value || len(decoded) != size {
		return nil, fmt.Errorf("must be canonical base64url with exactly %d bytes", size)
	}
	return decoded, nil
}

func canonicalVersion(value string) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 64 {
		return false
	}
	for _, character := range value {
		if (character < 'a' || character > 'z') &&
			(character < 'A' || character > 'Z') &&
			(character < '0' || character > '9') && character != '-' {
			return false
		}
	}
	return true
}
