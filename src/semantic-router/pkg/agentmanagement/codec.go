package agentmanagement

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type signedCodec struct {
	active string
	keys   map[string][]byte
}

type cursorPayload struct {
	Version          int       `json:"v"`
	NamespaceID      string    `json:"namespaceId"`
	Kind             string    `json:"kind"`
	ScopeDigest      string    `json:"scopeDigest"`
	Search           string    `json:"search,omitempty"`
	OwnerPrincipalID string    `json:"ownerPrincipalId,omitempty"`
	Timestamp        time.Time `json:"timestamp,omitempty"`
	ID               string    `json:"id,omitempty"`
	Sequence         int64     `json:"sequence,omitempty"`
}

type toolCursorPayload struct {
	Version          int    `json:"v"`
	NamespaceID      string `json:"namespaceId"`
	RegistryRevision string `json:"registryRevision"`
	Search           string `json:"search,omitempty"`
	AfterName        string `json:"afterName"`
}

func newSignedCodec(keyring securitykeyring.Symmetric) (signedCodec, error) {
	if keyring.ActiveVersion == "" || len(keyring.Keys) == 0 || len(keyring.Keys) > 8 {
		return signedCodec{}, fmt.Errorf("%w: Agent keyring is invalid", ErrInvalid)
	}
	codec := signedCodec{active: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if version == "" || len(version) > 64 || len(key) != sha256.Size {
			codec.close()
			return signedCodec{}, fmt.Errorf("%w: Agent keys must be 256 bits", ErrInvalid)
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.active]; !found {
		codec.close()
		return signedCodec{}, fmt.Errorf("%w: Agent active key is missing", ErrInvalid)
	}
	return codec, nil
}

func (codec signedCodec) encodeCursor(payload cursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return codec.envelope("cursor", encoded)
}

func (codec signedCodec) decodeCursor(value string) (cursorPayload, error) {
	payload, err := codec.open("cursor", value, 4096)
	if err != nil {
		return cursorPayload{}, ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var cursor cursorPayload
	if decoder.Decode(&cursor) != nil || cursor.Version != 1 || cursor.NamespaceID == "" ||
		cursor.Kind == "" || cursor.ScopeDigest == "" {
		return cursorPayload{}, ErrInvalid
	}
	if cursor.Kind == "events" {
		if cursor.Sequence < 1 || !cursor.Timestamp.IsZero() || cursor.ID != "" {
			return cursorPayload{}, ErrInvalid
		}
	} else if cursor.Timestamp.IsZero() || cursor.ID == "" || cursor.Sequence != 0 {
		return cursorPayload{}, ErrInvalid
	}
	return cursor, nil
}

func (codec signedCodec) encodeToolCursor(payload toolCursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return codec.envelope("tool-cursor", encoded)
}

func (codec signedCodec) decodeToolCursor(value string) (toolCursorPayload, error) {
	payload, err := codec.open("tool-cursor", value, 4096)
	if err != nil {
		return toolCursorPayload{}, ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var cursor toolCursorPayload
	if decoder.Decode(&cursor) != nil || decoder.Decode(&struct{}{}) != io.EOF ||
		cursor.Version != 1 || cursor.NamespaceID == "" ||
		!validSHA256Digest(cursor.RegistryRevision) || !canonicalToolName(cursor.AfterName) {
		return toolCursorPayload{}, ErrInvalid
	}
	return cursor, nil
}

func (codec signedCodec) envelope(kind string, payload []byte) (string, error) {
	key, found := codec.keys[codec.active]
	if !found {
		return "", ErrInvalid
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/agent-" + kind + "/v1\x00"))
	_, _ = mac.Write([]byte(codec.active))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return strings.Join([]string{
		"agent", kind, codec.active,
		base64.RawURLEncoding.EncodeToString(payload),
		base64.RawURLEncoding.EncodeToString(mac.Sum(nil)),
	}, "."), nil
}

func (codec signedCodec) open(kind, value string, maximum int) ([]byte, error) {
	if len(value) > maximum {
		return nil, ErrInvalid
	}
	parts := strings.Split(value, ".")
	if len(parts) != 5 || parts[0] != "agent" || parts[1] != kind {
		return nil, ErrInvalid
	}
	key, found := codec.keys[parts[2]]
	if !found {
		return nil, ErrInvalid
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[3] {
		return nil, ErrInvalid
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[4])
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[4] {
		return nil, ErrInvalid
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/agent-" + kind + "/v1\x00"))
	_, _ = mac.Write([]byte(parts[2]))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return nil, ErrInvalid
	}
	return payload, nil
}

func (codec *signedCodec) close() {
	if codec == nil {
		return
	}
	for version, key := range codec.keys {
		clear(key)
		delete(codec.keys, version)
	}
	codec.active = ""
}
