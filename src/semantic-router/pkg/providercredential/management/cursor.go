package management

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	cursorWirePrefix        = "pcc"
	maximumCursorKeyVersion = 8
)

type listCursor struct {
	Version     int                       `json:"v"`
	NamespaceID string                    `json:"namespaceId"`
	ProviderID  string                    `json:"providerId,omitempty"`
	Status      providercredential.Status `json:"status,omitempty"`
	AfterStatus providercredential.Status `json:"afterStatus"`
	AfterID     string                    `json:"afterId"`
	ScopeDigest string                    `json:"scopeDigest"`
}

type cursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newCursorCodec(keyring securitykeyring.Symmetric) (cursorCodec, error) {
	if !canonicalCursorKeyVersion(keyring.ActiveVersion) ||
		len(keyring.Keys) < 1 || len(keyring.Keys) > maximumCursorKeyVersion {
		return cursorCodec{}, fmt.Errorf("provider credential cursor keyring must contain 1-%d canonical versions", maximumCursorKeyVersion)
	}
	codec := cursorCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if !canonicalCursorKeyVersion(version) || len(key) != sha256.Size {
			return cursorCodec{}, errors.New("provider credential cursor versions require canonical names and exactly 256-bit keys")
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		return cursorCodec{}, errors.New("provider credential active cursor key version is not retained")
	}
	return codec, nil
}

func (codec cursorCodec) encode(cursor listCursor) (string, error) {
	payload, err := json.Marshal(cursor)
	if err != nil {
		return "", fmt.Errorf("encode provider credential cursor: %w", err)
	}
	key := codec.keys[codec.activeVersion]
	signature := codec.sign(key, codec.activeVersion, payload)
	return cursorWirePrefix + "." + codec.activeVersion + "." +
		base64.RawURLEncoding.EncodeToString(payload) + "." +
		base64.RawURLEncoding.EncodeToString(signature), nil
}

func (codec cursorCodec) decode(value string) (listCursor, error) {
	if len(value) < 3 || len(value) > 4096 {
		return listCursor{}, fmt.Errorf("%w: cursor is malformed", ErrInvalidRequest)
	}
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != cursorWirePrefix ||
		!canonicalCursorKeyVersion(parts[1]) || parts[2] == "" || parts[3] == "" {
		return listCursor{}, fmt.Errorf("%w: cursor is malformed", ErrInvalidRequest)
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return listCursor{}, fmt.Errorf("%w: cursor key version is unavailable", ErrInvalidRequest)
	}
	payloadEncoded, signatureEncoded := parts[2], parts[3]
	payload, err := base64.RawURLEncoding.DecodeString(payloadEncoded)
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != payloadEncoded {
		return listCursor{}, fmt.Errorf("%w: cursor is malformed", ErrInvalidRequest)
	}
	signature, err := base64.RawURLEncoding.DecodeString(signatureEncoded)
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != signatureEncoded ||
		subtle.ConstantTimeCompare(signature, codec.sign(key, parts[1], payload)) != 1 {
		return listCursor{}, fmt.Errorf("%w: cursor signature is invalid", ErrInvalidRequest)
	}
	var cursor listCursor
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cursor); err != nil || cursor.Version != 1 {
		return listCursor{}, fmt.Errorf("%w: cursor payload is invalid", ErrInvalidRequest)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return listCursor{}, fmt.Errorf("%w: cursor payload is invalid", ErrInvalidRequest)
	}
	return cursor, nil
}

func (codec cursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/provider-credential-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec *cursorCodec) close() {
	if codec == nil {
		return
	}
	for _, key := range codec.keys {
		providercredential.Zero(key)
	}
	codec.activeVersion = ""
	codec.keys = nil
}

func canonicalCursorKeyVersion(value string) bool {
	if len(value) < 1 || len(value) > 64 {
		return false
	}
	for index, character := range value {
		if (character < 'a' || character > 'z') &&
			(character < 'A' || character > 'Z') &&
			(character < '0' || character > '9') && (character != '-' || index == 0) {
			return false
		}
	}
	return true
}
