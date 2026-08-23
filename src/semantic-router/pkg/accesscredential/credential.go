package accesscredential

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

const (
	secretBytes = 32
	digestBytes = sha256.Size
)

var publicIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9-]{11,63}$`)

// Kind gives inference API keys and delegated inference sessions disjoint wire
// formats. Neither format is accepted as a Management credential.
type Kind string

const (
	KindAPIKey     Kind = "api_key"
	KindDelegation Kind = "delegation"
)

func (k Kind) prefix() string {
	switch k {
	case KindAPIKey:
		return "vsr"
	case KindDelegation:
		return "vsd"
	default:
		return ""
	}
}

// Digest is the non-secret credential verifier stored in PostgreSQL and the
// applied runtime projection. PepperVersion selects deployment key material;
// it is not controlled by the caller.
type Digest struct {
	Kind          Kind
	PublicID      string
	PepperVersion string
	HMAC          []byte
}

// Issued is the one-time result of credential creation. Plaintext must only be
// placed in the encrypted idempotency response envelope or returned over the
// private Management transport. It must never be logged or persisted directly.
type Issued struct {
	Plaintext string
	Digest    Digest
}

// PepperKeyring holds versioned HMAC keys. ActiveVersion is used only for new
// credentials; verification always uses the version pinned in Digest.
type PepperKeyring struct {
	ActiveVersion string
	Keys          map[string][]byte
}

func (k PepperKeyring) Validate() error {
	if strings.TrimSpace(k.ActiveVersion) == "" {
		return errors.New("active pepper version is required")
	}
	if len(k.Keys) == 0 {
		return errors.New("at least one pepper key is required")
	}
	for version, key := range k.Keys {
		if strings.TrimSpace(version) == "" || version != strings.TrimSpace(version) {
			return errors.New("pepper versions must be non-empty and canonical")
		}
		if len(key) < secretBytes {
			return fmt.Errorf("pepper %q must contain at least 256 bits", version)
		}
	}
	if _, ok := k.Keys[k.ActiveVersion]; !ok {
		return fmt.Errorf("active pepper version %q is absent", k.ActiveVersion)
	}
	return nil
}

// Issue creates a 256-bit random machine credential and its HMAC verifier.
func (k PepperKeyring) Issue(kind Kind, publicID string) (Issued, error) {
	if err := k.Validate(); err != nil {
		return Issued{}, err
	}
	if err := validateIdentity(kind, publicID); err != nil {
		return Issued{}, err
	}
	secret := make([]byte, secretBytes)
	if _, err := rand.Read(secret); err != nil {
		return Issued{}, fmt.Errorf("generate credential secret: %w", err)
	}
	defer zero(secret)
	plain := format(kind, publicID, secret)
	digest := credentialHMAC(k.Keys[k.ActiveVersion], kind, publicID, secret)
	return Issued{
		Plaintext: plain,
		Digest: Digest{
			Kind:          kind,
			PublicID:      publicID,
			PepperVersion: k.ActiveVersion,
			HMAC:          digest,
		},
	}, nil
}

// Verify parses a presented credential, selects the digest-pinned pepper, and
// compares its HMAC in constant time. It deliberately does not decide status or
// time bounds; the atomic runtime admission operation owns those checks using
// Valkey server time.
func (k PepperKeyring) Verify(presented string, expected Digest) error {
	if err := k.Validate(); err != nil {
		return err
	}
	kind, publicID, secret, err := parse(presented)
	if err != nil {
		return ErrInvalidCredential
	}
	defer zero(secret)
	if kind != expected.Kind || publicID != expected.PublicID || len(expected.HMAC) != digestBytes {
		return ErrInvalidCredential
	}
	pepper, ok := k.Keys[expected.PepperVersion]
	if !ok {
		return ErrPepperUnavailable
	}
	actual := credentialHMAC(pepper, kind, publicID, secret)
	if subtle.ConstantTimeCompare(actual, expected.HMAC) != 1 {
		return ErrInvalidCredential
	}
	return nil
}

var (
	ErrInvalidCredential = errors.New("invalid inference credential")
	ErrPepperUnavailable = errors.New("credential pepper is unavailable")
)

// PublicID returns only the non-secret lookup component after strict wire
// validation. Callers may use it for the O(1) runtime projection lookup.
func PublicID(presented string) (Kind, string, error) {
	kind, publicID, secret, err := parse(presented)
	zero(secret)
	if err != nil {
		return "", "", ErrInvalidCredential
	}
	return kind, publicID, nil
}

func parse(presented string) (Kind, string, []byte, error) {
	parts := strings.SplitN(presented, "_", 3)
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return "", "", nil, ErrInvalidCredential
	}
	var kind Kind
	switch parts[0] {
	case KindAPIKey.prefix():
		kind = KindAPIKey
	case KindDelegation.prefix():
		kind = KindDelegation
	default:
		return "", "", nil, ErrInvalidCredential
	}
	if err := validateIdentity(kind, parts[1]); err != nil {
		return "", "", nil, ErrInvalidCredential
	}
	secret, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(secret) != secretBytes || base64.RawURLEncoding.EncodeToString(secret) != parts[2] {
		zero(secret)
		return "", "", nil, ErrInvalidCredential
	}
	return kind, parts[1], secret, nil
}

func format(kind Kind, publicID string, secret []byte) string {
	return kind.prefix() + "_" + publicID + "_" + base64.RawURLEncoding.EncodeToString(secret)
}

func validateIdentity(kind Kind, publicID string) error {
	if kind.prefix() == "" {
		return errors.New("credential kind is invalid")
	}
	if !publicIDPattern.MatchString(publicID) {
		return errors.New("public credential id must be 12-64 alphanumeric or hyphen characters")
	}
	return nil
}

func credentialHMAC(key []byte, kind Kind, publicID string, secret []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/inference-credential/v1\x00"))
	_, _ = mac.Write([]byte(kind))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(publicID))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(secret)
	return mac.Sum(nil)
}

func zero(value []byte) {
	for i := range value {
		value[i] = 0
	}
}
