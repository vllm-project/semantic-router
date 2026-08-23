package delegationmanagement

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type cursorPayload struct {
	Version     int       `json:"v"`
	Kind        string    `json:"kind"`
	NamespaceID string    `json:"namespaceId"`
	PrincipalID string    `json:"principalId,omitempty"`
	APIKeyID    string    `json:"keyId,omitempty"`
	CreatedAt   time.Time `json:"createdAt"`
	ID          string    `json:"id"`
}

type cursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newCursorCodec(keyring securitykeyring.Symmetric) (cursorCodec, error) {
	if keyring.ActiveVersion == "" || len(keyring.Keys) < 1 || len(keyring.Keys) > 8 {
		return cursorCodec{}, ErrUnavailable
	}
	codec := cursorCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if version == "" || len(key) != sha256.Size {
			return cursorCodec{}, ErrUnavailable
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		return cursorCodec{}, ErrUnavailable
	}
	return codec, nil
}

func (codec cursorCodec) encode(payload cursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", ErrUnavailable
	}
	mac := hmac.New(sha256.New, codec.keys[codec.activeVersion])
	_, _ = mac.Write([]byte("vllm-sr/delegation-management-cursor/v1\x00" + codec.activeVersion + "\x00"))
	_, _ = mac.Write(encoded)
	return strings.Join([]string{
		"delegation", codec.activeVersion,
		base64.RawURLEncoding.EncodeToString(encoded), base64.RawURLEncoding.EncodeToString(mac.Sum(nil)),
	}, "."), nil
}

func (codec cursorCodec) decode(value string) (cursorPayload, error) {
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != "delegation" || len(value) > 4096 {
		return cursorPayload{}, ErrInvalidRequest
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return cursorPayload{}, ErrInvalidRequest
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return cursorPayload{}, ErrInvalidRequest
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[3])
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/delegation-management-cursor/v1\x00" + parts[1] + "\x00"))
	_, _ = mac.Write(payload)
	if err != nil || len(signature) != sha256.Size || base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		subtle.ConstantTimeCompare(signature, mac.Sum(nil)) != 1 {
		return cursorPayload{}, ErrInvalidRequest
	}
	var decoded cursorPayload
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&decoded) != nil || decoded.Version != 1 {
		return cursorPayload{}, ErrInvalidRequest
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return cursorPayload{}, ErrInvalidRequest
	}
	return decoded, nil
}

func (codec cursorCodec) close() {
	for _, key := range codec.keys {
		clear(key)
	}
}
