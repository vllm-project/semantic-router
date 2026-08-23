package subjectmanagement

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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const subjectCursorPrefix = "subj"

type cursorPayload struct {
	Version     int       `json:"v"`
	Kind        string    `json:"kind"`
	NamespaceID string    `json:"namespaceId"`
	OwnerID     string    `json:"ownerId,omitempty"`
	Status      string    `json:"status,omitempty"`
	Search      string    `json:"search,omitempty"`
	ScopeDigest string    `json:"scopeDigest,omitempty"`
	CreatedAt   time.Time `json:"createdAt"`
	ID          string    `json:"id"`
}

type cursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newCursorCodec(keyring securitykeyring.Symmetric) (cursorCodec, error) {
	if !validKeyVersion(keyring.ActiveVersion) || len(keyring.Keys) < 1 || len(keyring.Keys) > 8 {
		return cursorCodec{}, errors.New("subject cursor keyring must contain 1-8 canonical versions")
	}
	codec := cursorCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if !validKeyVersion(version) || len(key) != sha256.Size {
			return cursorCodec{}, errors.New("subject cursor keys must be exactly 256 bits")
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		return cursorCodec{}, errors.New("subject cursor active version is not retained")
	}
	return codec, nil
}

func (codec cursorCodec) encode(payload cursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode subject cursor: %w", err)
	}
	signature := codec.sign(codec.keys[codec.activeVersion], codec.activeVersion, encoded)
	return strings.Join([]string{
		subjectCursorPrefix,
		codec.activeVersion,
		base64.RawURLEncoding.EncodeToString(encoded),
		base64.RawURLEncoding.EncodeToString(signature),
	}, "."), nil
}

func (codec cursorCodec) decode(value string) (cursorPayload, error) {
	if len(value) < 4 || len(value) > 4096 {
		return cursorPayload{}, ErrInvalidRequest
	}
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != subjectCursorPrefix || !validKeyVersion(parts[1]) {
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
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		subtle.ConstantTimeCompare(signature, codec.sign(key, parts[1], payload)) != 1 {
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

func (codec cursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/subject-management-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec cursorCodec) close() {
	for _, key := range codec.keys {
		for index := range key {
			key[index] = 0
		}
	}
}

func validKeyVersion(value string) bool {
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
