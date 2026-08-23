package managementidentity

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

const workloadCursorPrefix = "wid"

type workloadCursorPayload struct {
	Version     int       `json:"v"`
	Kind        string    `json:"kind"`
	OwnerID     string    `json:"ownerId,omitempty"`
	Status      string    `json:"status,omitempty"`
	ScopeDigest string    `json:"scopeDigest"`
	CreatedAt   time.Time `json:"createdAt"`
	ID          string    `json:"id"`
}

type workloadCursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

func newWorkloadCursorCodec(keyring securitykeyring.Symmetric) (workloadCursorCodec, error) {
	if !canonicalWorkloadKeyVersion(keyring.ActiveVersion) || len(keyring.Keys) < 1 || len(keyring.Keys) > 8 {
		return workloadCursorCodec{}, errors.New("workload identity cursor keyring is invalid")
	}
	codec := workloadCursorCodec{activeVersion: keyring.ActiveVersion, keys: make(map[string][]byte, len(keyring.Keys))}
	for version, key := range keyring.Keys {
		if !canonicalWorkloadKeyVersion(version) || len(key) != sha256.Size {
			codec.close()
			return workloadCursorCodec{}, errors.New("workload identity cursor keys must be 256 bits")
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		codec.close()
		return workloadCursorCodec{}, errors.New("active workload identity cursor key is unavailable")
	}
	return codec, nil
}

func (codec workloadCursorCodec) encode(payload workloadCursorPayload) (string, error) {
	payload.Version = 1
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	signature := codec.sign(codec.keys[codec.activeVersion], codec.activeVersion, encoded)
	return strings.Join([]string{
		workloadCursorPrefix, codec.activeVersion,
		base64.RawURLEncoding.EncodeToString(encoded), base64.RawURLEncoding.EncodeToString(signature),
	}, "."), nil
}

func (codec workloadCursorCodec) decode(value string) (workloadCursorPayload, error) {
	if len(value) < 4 || len(value) > 4096 {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	parts := strings.Split(value, ".")
	if len(parts) != 4 || parts[0] != workloadCursorPrefix || !canonicalWorkloadKeyVersion(parts[1]) {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		subtle.ConstantTimeCompare(signature, codec.sign(key, parts[1], payload)) != 1 {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	var decoded workloadCursorPayload
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&decoded) != nil || decoded.Version != 1 {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return workloadCursorPayload{}, ErrInvalidWorkloadRequest
	}
	return decoded, nil
}

func (codec workloadCursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/workload-identity-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec workloadCursorCodec) close() {
	for _, key := range codec.keys {
		zeroWorkloadBytes(key)
	}
}

func canonicalWorkloadKeyVersion(value string) bool {
	if value == "" || len(value) > 64 || strings.TrimSpace(value) != value {
		return false
	}
	for index, character := range value {
		if (character < 'a' || character > 'z') && (character < 'A' || character > 'Z') &&
			(character < '0' || character > '9') && (character != '-' || index == 0) {
			return false
		}
	}
	return true
}
