package providercredential

import (
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

var (
	ErrUnavailable = errors.New("provider credential is unavailable")
	ErrMismatch    = errors.New("provider credential binding mismatch")
)

const maximumSecretBytes = 64 * 1024

// Codec uses the deployment's dedicated provider-credential KEK keyring.
// Its keyring must not be shared with revealable inference keys or response
// envelopes.
type Codec struct {
	Keyring accesscredential.KEKKeyring
}

func (c Codec) Seal(credential Credential, versionID string, secret []byte, now time.Time) (Version, error) {
	if credential.ID == "" || credential.NamespaceID == "" || credential.ProviderID == "" ||
		credential.CredentialMode == "" || credential.CredentialAdapterID == "" ||
		credential.CatalogRevision == "" || credential.NormalizedOrigin == "" {
		return Version{}, errors.New("provider credential identity is required")
	}
	if err := validateUUIDs(map[string]string{"credential_id": credential.ID, "namespace_id": credential.NamespaceID, "version_id": versionID}); err != nil {
		return Version{}, err
	}
	if len(secret) == 0 || len(secret) > maximumSecretBytes {
		return Version{}, errors.New("provider credential secret must contain 1-65536 bytes")
	}
	if now.IsZero() {
		return Version{}, errors.New("provider credential creation time is required")
	}
	envelope, err := c.Keyring.Seal(secret, additionalData(credential, versionID))
	if err != nil {
		return Version{}, fmt.Errorf("encrypt provider credential: %w", err)
	}
	return Version{
		ID: versionID, NamespaceID: credential.NamespaceID, CredentialID: credential.ID,
		Envelope: envelope, Status: VersionActive, NotBefore: now.UTC(), CreatedAt: now.UTC(),
	}, nil
}

// OpenActive resolves the one version available to a new dispatch. Provider
// and origin are rechecked against the pinned Model backend before decryption.
func (c Codec) OpenActive(
	credential Credential,
	version Version,
	expectedProvider string,
	expectedOrigin string,
	now time.Time,
) ([]byte, error) {
	if err := validateBinding(credential, version, expectedProvider, expectedOrigin); err != nil {
		return nil, err
	}
	if credential.Status != StatusActive || credential.ActiveVersionID == nil ||
		*credential.ActiveVersionID != version.ID || version.Status != VersionActive || !versionAvailable(version, now) {
		return nil, ErrUnavailable
	}
	return c.open(credential, version)
}

// OpenPinned permits an already journaled dispatch to finish on the exact
// version it pinned before a rotation. A retiring version is usable only until
// its bounded expiry; a revoked version is never usable.
func (c Codec) OpenPinned(
	credential Credential,
	version Version,
	expectedProvider string,
	expectedOrigin string,
	now time.Time,
) ([]byte, error) {
	if err := validateBinding(credential, version, expectedProvider, expectedOrigin); err != nil {
		return nil, err
	}
	if credential.Status != StatusActive ||
		(version.Status != VersionActive && version.Status != VersionRetiring) || !versionAvailable(version, now) {
		return nil, ErrUnavailable
	}
	return c.open(credential, version)
}

func (c Codec) open(credential Credential, version Version) ([]byte, error) {
	plaintext, err := c.Keyring.Open(version.Envelope, additionalData(credential, version.ID))
	if err != nil {
		return nil, fmt.Errorf("decrypt provider credential: %w", ErrUnavailable)
	}
	if len(plaintext) == 0 || len(plaintext) > maximumSecretBytes {
		zero(plaintext)
		return nil, ErrUnavailable
	}
	return plaintext, nil
}

func validateBinding(credential Credential, version Version, provider, origin string) error {
	if err := credential.Validate(); err != nil {
		return fmt.Errorf("%w: invalid credential metadata", ErrMismatch)
	}
	if err := version.Validate(); err != nil {
		return fmt.Errorf("%w: invalid credential version", ErrMismatch)
	}
	if version.NamespaceID != credential.NamespaceID || version.CredentialID != credential.ID ||
		provider != credential.ProviderID || origin != credential.NormalizedOrigin {
		return ErrMismatch
	}
	return nil
}

func versionAvailable(version Version, now time.Time) bool {
	if now.IsZero() || now.Before(version.NotBefore) {
		return false
	}
	return version.ExpiresAt == nil || now.Before(*version.ExpiresAt)
}

func additionalData(credential Credential, versionID string) []byte {
	return []byte("vllm-sr/provider-credential/v1\x00" + credential.NamespaceID + "\x00" +
		credential.ID + "\x00" + versionID + "\x00" + credential.ProviderID + "\x00" +
		string(credential.CredentialMode) + "\x00" + credential.CredentialAdapterID + "\x00" +
		credential.CatalogRevision + "\x00" + credential.NormalizedOrigin)
}

// Zero erases a resolved secret after the provider adapter has constructed its
// request authentication state.
func Zero(secret []byte) { zero(secret) }

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
