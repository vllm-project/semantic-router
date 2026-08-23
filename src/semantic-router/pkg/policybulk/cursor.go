package policybulk

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

const operationCursorPrefix = "opb"

type operationCursorPayload struct {
	Version           int            `json:"v"`
	NamespaceID       string         `json:"namespaceId"`
	OriginPrincipalID string         `json:"originPrincipalId,omitempty"`
	Kind              string         `json:"kind,omitempty"`
	State             OperationState `json:"state,omitempty"`
	CreatedAt         time.Time      `json:"createdAt"`
	ID                string         `json:"id"`
	VisibilityDigest  string         `json:"visibilityDigest"`
}

type operationCursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newOperationCursorCodec(keyring securitykeyring.Symmetric) (operationCursorCodec, error) {
	if !validCursorKeyVersion(keyring.ActiveVersion) || len(keyring.Keys) < 1 || len(keyring.Keys) > 8 {
		return operationCursorCodec{}, errors.New("operation cursor keyring must contain 1-8 canonical versions")
	}
	codec := operationCursorCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if !validCursorKeyVersion(version) || len(key) != sha256.Size {
			codec.close()
			return operationCursorCodec{}, errors.New("operation cursor keys must be exactly 256 bits")
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		codec.close()
		return operationCursorCodec{}, errors.New("operation cursor active version is not retained")
	}
	return codec, nil
}

func (codec operationCursorCodec) encode(payload operationCursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode operation cursor: %w", err)
	}
	signature := codec.sign(codec.keys[codec.activeVersion], codec.activeVersion, encoded)
	return strings.Join([]string{
		operationCursorPrefix, codec.activeVersion,
		base64.RawURLEncoding.EncodeToString(encoded), base64.RawURLEncoding.EncodeToString(signature),
	}, "."), nil
}

func (codec operationCursorCodec) decode(value string) (operationCursorPayload, error) {
	if len(value) < 4 || len(value) > 4096 {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != operationCursorPrefix || !validCursorKeyVersion(parts[1]) {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		subtle.ConstantTimeCompare(signature, codec.sign(key, parts[1], payload)) != 1 {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	var decoded operationCursorPayload
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&decoded) != nil || decoded.Version != 1 {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return operationCursorPayload{}, ErrInvalidRequest
	}
	return decoded, nil
}

func (codec operationCursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/policy-bulk-operation-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec operationCursorCodec) close() {
	for _, key := range codec.keys {
		for index := range key {
			key[index] = 0
		}
	}
}

func validCursorKeyVersion(value string) bool {
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
