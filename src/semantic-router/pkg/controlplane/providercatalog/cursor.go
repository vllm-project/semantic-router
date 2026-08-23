package providercatalog

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

var (
	ErrInvalidRequest = errors.New("invalid provider catalog request")
	ErrInvalidCursor  = errors.New("invalid provider catalog cursor")
	ErrStaleCursor    = errors.New("provider catalog cursor is stale")
	ErrNotFound       = errors.New("provider is not in the catalog")
)

const (
	cursorWirePrefix      = "pcat"
	maximumCursorVersions = 8
)

type cursorCodec struct {
	activeVersion string
	keys          map[string][]byte
}

type listCursor struct {
	Version         int    `json:"v"`
	CatalogRevision string `json:"r"`
	QueryDigest     string `json:"q"`
	Order           uint32 `json:"o"`
	ProviderID      string `json:"p"`
}

func newCursorCodec(keyring securitykeyring.Symmetric) (cursorCodec, error) {
	if !canonicalCursorVersion(keyring.ActiveVersion) || len(keyring.Keys) < 1 || len(keyring.Keys) > maximumCursorVersions {
		return cursorCodec{}, fmt.Errorf("%w: cursor keyring must contain 1-%d canonical versions", ErrInvalidRequest, maximumCursorVersions)
	}
	codec := cursorCodec{
		activeVersion: keyring.ActiveVersion,
		keys:          make(map[string][]byte, len(keyring.Keys)),
	}
	for version, key := range keyring.Keys {
		if !canonicalCursorVersion(version) || len(key) != sha256.Size {
			codec.close()
			return cursorCodec{}, fmt.Errorf("%w: cursor versions require canonical names and exactly 256-bit keys", ErrInvalidRequest)
		}
		codec.keys[version] = append([]byte(nil), key...)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		codec.close()
		return cursorCodec{}, fmt.Errorf("%w: active cursor key version is not retained", ErrInvalidRequest)
	}
	return codec, nil
}

func (codec cursorCodec) encode(value listCursor) (string, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode provider catalog cursor: %w", err)
	}
	key, found := codec.keys[codec.activeVersion]
	if !found {
		return "", fmt.Errorf("%w: active cursor key is unavailable", ErrInvalidRequest)
	}
	return cursorWirePrefix + "." + codec.activeVersion + "." +
		base64.RawURLEncoding.EncodeToString(payload) + "." +
		base64.RawURLEncoding.EncodeToString(codec.sign(key, codec.activeVersion, payload)), nil
}

func (codec cursorCodec) decode(encoded string) (listCursor, error) {
	if encoded == "" || len(encoded) > 2048 {
		return listCursor{}, ErrInvalidCursor
	}
	parts := strings.Split(encoded, ".")
	if len(parts) != 4 || parts[0] != cursorWirePrefix || !canonicalCursorVersion(parts[1]) ||
		parts[2] == "" || parts[3] == "" {
		return listCursor{}, ErrInvalidCursor
	}
	key, found := codec.keys[parts[1]]
	if !found {
		return listCursor{}, ErrInvalidCursor
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return listCursor{}, ErrInvalidCursor
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(signature) != sha256.Size || base64.RawURLEncoding.EncodeToString(signature) != parts[3] ||
		!hmac.Equal(signature, codec.sign(key, parts[1], payload)) {
		return listCursor{}, ErrInvalidCursor
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var value listCursor
	if err := decoder.Decode(&value); err != nil {
		return listCursor{}, ErrInvalidCursor
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return listCursor{}, ErrInvalidCursor
	}
	if value.Version != 1 || !strings.HasPrefix(value.CatalogRevision, "sha256:") ||
		!isSHA256Hex(value.QueryDigest) || !idPattern.MatchString(value.ProviderID) {
		return listCursor{}, ErrInvalidCursor
	}
	return value, nil
}

func (codec cursorCodec) sign(key []byte, version string, payload []byte) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/provider-catalog-cursor/v1\x00"))
	_, _ = mac.Write([]byte(version))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (codec *cursorCodec) close() {
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

func canonicalCursorVersion(value string) bool {
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

func listQueryDigest(query normalizedListQuery) string {
	payload, _ := json.Marshal(struct {
		Search     string `json:"search"`
		Category   string `json:"category"`
		Capability string `json:"capability"`
	}{Search: query.search, Category: query.category, Capability: query.capability})
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func isSHA256Hex(value string) bool {
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}
