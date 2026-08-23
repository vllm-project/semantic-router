package invitationmanagement

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

const invitationTokenPrefix = "vsi"

type tokenCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newTokenCodec(keyring accesscredential.PepperKeyring) (*tokenCodec, error) {
	if err := keyring.Validate(); err != nil || len(keyring.Keys) > 8 {
		return nil, ErrUnavailable
	}
	codec := &tokenCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if len(key) != sha256.Size {
			codec.Close()
			return nil, ErrUnavailable
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	return codec, nil
}

func (codec *tokenCodec) Close() {
	if codec == nil {
		return
	}
	for version, key := range codec.keys {
		zero(key)
		delete(codec.keys, version)
	}
}

func (codec *tokenCodec) Versions() []string {
	versions := make([]string, 0, len(codec.keys))
	for version := range codec.keys {
		versions = append(versions, version)
	}
	return versions
}

func (codec *tokenCodec) Issue(invitationID string) (string, []byte, string, error) {
	if codec == nil || !canonicalUUID(invitationID) {
		return "", nil, "", ErrUnavailable
	}
	secret := make([]byte, sha256.Size)
	if _, err := rand.Read(secret); err != nil {
		return "", nil, "", ErrUnavailable
	}
	defer zero(secret)
	plaintext := invitationTokenPrefix + "_" + invitationID + "_" + base64.RawURLEncoding.EncodeToString(secret)
	digest := invitationHMAC(codec.keys[codec.activeVersion], invitationID, secret)
	return plaintext, digest, codec.activeVersion, nil
}

func (codec *tokenCodec) Verify(presented string, expectedDigest []byte, pepperVersion string) error {
	id, secret, err := parseToken(presented)
	if err != nil {
		return ErrInvalidToken
	}
	defer zero(secret)
	key, found := codec.keys[pepperVersion]
	if !found {
		return ErrPepperUnavailable
	}
	actual := invitationHMAC(key, id, secret)
	if len(expectedDigest) != sha256.Size || subtle.ConstantTimeCompare(actual, expectedDigest) != 1 {
		return ErrInvalidToken
	}
	return nil
}

func tokenInvitationID(presented string) (string, error) {
	id, secret, err := parseToken(presented)
	zero(secret)
	return id, err
}

func parseToken(presented string) (string, []byte, error) {
	parts := strings.SplitN(presented, "_", 3)
	if len(parts) != 3 || parts[0] != invitationTokenPrefix || !canonicalUUID(parts[1]) {
		return "", nil, ErrInvalidToken
	}
	secret, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(secret) != sha256.Size || base64.RawURLEncoding.EncodeToString(secret) != parts[2] {
		zero(secret)
		return "", nil, ErrInvalidToken
	}
	return parts[1], secret, nil
}

func invitationHMAC(key []byte, invitationID string, secret []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/management-invitation/v1\x00"))
	_, _ = mac.Write([]byte(invitationID))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(secret)
	return mac.Sum(nil)
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

func mapTokenError(err error) error {
	if errors.Is(err, ErrPepperUnavailable) {
		return ErrUnavailable
	}
	return ErrIdentityMismatch
}
