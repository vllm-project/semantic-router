package routingmanagement

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const routingCursorPrefix = "routing"

type routingResourceKind string

const (
	routingResourceModel      routingResourceKind = "model"
	routingResourceRecipe     routingResourceKind = "recipe"
	routingResourceEntrypoint routingResourceKind = "entrypoint"
	routingResourceSnapshot   routingResourceKind = "snapshot"
)

type routingCursorPayload struct {
	Version         int                 `json:"v"`
	NamespaceID     string              `json:"namespaceId"`
	ResourceKind    routingResourceKind `json:"resourceKind"`
	Status          Status              `json:"status,omitempty"`
	Search          string              `json:"search,omitempty"`
	ScopeDigest     string              `json:"scopeDigest"`
	CreatedAt       time.Time           `json:"createdAt"`
	ID              string              `json:"id"`
	RoutingRevision int64               `json:"routingRevision,omitempty"`
}

type routingCursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newRoutingCursorCodec(keyring securitykeyring.Symmetric) (routingCursorCodec, error) {
	if !validCursorVersion(keyring.ActiveVersion) || len(keyring.Keys) < 1 || len(keyring.Keys) > 8 {
		return routingCursorCodec{}, errors.New("routing cursor keyring must contain 1-8 canonical versions")
	}
	codec := routingCursorCodec{
		activeVersion: keyring.ActiveVersion,
		keys:          make(map[string][]byte, len(keyring.Keys)),
	}
	for version, key := range keyring.Keys {
		if !validCursorVersion(version) || len(key) != sha256.Size {
			codec.close()
			return routingCursorCodec{}, errors.New("routing cursor keys must be exactly 256 bits")
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		codec.close()
		return routingCursorCodec{}, errors.New("routing cursor active version is not retained")
	}
	return codec, nil
}

func (codec routingCursorCodec) encode(payload routingCursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode routing cursor: %w", err)
	}
	key, found := codec.keys[codec.activeVersion]
	if !found {
		return "", errors.New("routing cursor active key is unavailable")
	}
	signature := codec.sign(key, codec.activeVersion, encoded)
	return strings.Join([]string{
		routingCursorPrefix,
		codec.activeVersion,
		base64.RawURLEncoding.EncodeToString(encoded),
		base64.RawURLEncoding.EncodeToString(signature),
	}, "."), nil
}

func (codec routingCursorCodec) decode(value string) (routingCursorPayload, error) {
	if len(value) < 4 || len(value) > 4096 {
		return routingCursorPayload{}, ErrInvalid
	}
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != routingCursorPrefix || !validCursorVersion(parts[1]) {
		return routingCursorPayload{}, ErrInvalid
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return routingCursorPayload{}, ErrInvalid
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return routingCursorPayload{}, ErrInvalid
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		!hmac.Equal(signature, codec.sign(key, parts[1], payload)) {
		return routingCursorPayload{}, ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var decoded routingCursorPayload
	if decoder.Decode(&decoded) != nil || decoded.Version != 1 {
		return routingCursorPayload{}, ErrInvalid
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return routingCursorPayload{}, ErrInvalid
	}
	return decoded, nil
}

func (codec routingCursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/routing-management-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec *routingCursorCodec) close() {
	if codec == nil {
		return
	}
	for version, key := range codec.keys {
		for index := range key {
			key[index] = 0
		}
		delete(codec.keys, version)
	}
	codec.activeVersion = ""
	codec.keys = nil
}

func validCursorVersion(value string) bool {
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
