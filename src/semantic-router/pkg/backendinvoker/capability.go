package backendinvoker

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const dispatchCapabilityPrefix = "vdc"

type DispatchCapability struct {
	NamespaceID     string                 `json:"namespaceId"`
	QuotaPartition  string                 `json:"quotaPartition"`
	PublicationID   string                 `json:"publicationId"`
	RuntimeEpoch    uint64                 `json:"runtimeEpoch"`
	RoutingRevision int64                  `json:"routingRevision"`
	RoutingDigest   string                 `json:"routingDigest"`
	AdmissionID     string                 `json:"admissionId"`
	AdmissionDigest string                 `json:"admissionDigest"`
	RequestID       string                 `json:"requestId"`
	Candidates      []DispatchCandidate    `json:"candidates"`
	Fallback        FallbackPolicy         `json:"fallback"`
	RequestDigest   string                 `json:"requestDigest"`
	Method          string                 `json:"method"`
	Path            string                 `json:"path"`
	Query           string                 `json:"query"`
	WireFormat      llmprotocol.WireFormat `json:"wireFormat"`
	Audience        string                 `json:"audience"`
	IssuedAt        int64                  `json:"issuedAt"`
	ExpiresAt       int64                  `json:"expiresAt"`
}

type SigningKeyring struct {
	ActiveVersion string
	Keys          map[string][]byte
	MaxLifetime   time.Duration
}

func (k SigningKeyring) Sign(capability DispatchCapability, now time.Time) (string, error) {
	key, ok := k.Keys[k.ActiveVersion]
	if !ok || len(key) < 32 {
		return "", fmt.Errorf("active dispatch signing key is unavailable")
	}
	if err := validateCapability(capability, now, k.MaxLifetime); err != nil {
		return "", err
	}
	payload, err := json.Marshal(capability)
	if err != nil {
		return "", fmt.Errorf("marshal dispatch capability: %w", err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	message := dispatchCapabilityPrefix + "." + k.ActiveVersion + "." + encoded
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(message))
	return message + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (k SigningKeyring) Verify(token, audience string, now time.Time) (DispatchCapability, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 4 || parts[0] != dispatchCapabilityPrefix {
		return DispatchCapability{}, fmt.Errorf("invalid dispatch capability")
	}
	key, ok := k.Keys[parts[1]]
	if !ok || len(key) < 32 {
		return DispatchCapability{}, fmt.Errorf("unknown dispatch signing key")
	}
	provided, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(provided) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(provided) != parts[3] {
		return DispatchCapability{}, fmt.Errorf("invalid dispatch capability signature")
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(strings.Join(parts[:3], ".")))
	expected := mac.Sum(nil)
	if len(provided) != len(expected) || subtle.ConstantTimeCompare(provided, expected) != 1 {
		return DispatchCapability{}, fmt.Errorf("invalid dispatch capability signature")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return DispatchCapability{}, fmt.Errorf("invalid dispatch capability payload")
	}
	var capability DispatchCapability
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&capability); err != nil {
		return DispatchCapability{}, fmt.Errorf("invalid dispatch capability payload")
	}
	if capability.Audience != audience {
		return DispatchCapability{}, fmt.Errorf("dispatch capability audience mismatch")
	}
	if err := validateCapability(capability, now, k.MaxLifetime); err != nil {
		return DispatchCapability{}, err
	}
	return capability, nil
}

func validateCapability(capability DispatchCapability, now time.Time, maxLifetime time.Duration) error {
	if capability.NamespaceID == "" || capability.QuotaPartition == "" ||
		capability.PublicationID == "" || capability.RuntimeEpoch == 0 ||
		capability.RoutingRevision <= 0 || !validSHA256Hex(capability.RoutingDigest) ||
		capability.AdmissionID == "" || !validSHA256Hex(capability.AdmissionDigest) ||
		!validBoundedIdentity(capability.RequestID) ||
		!validRequestDigest(capability.RequestDigest) || capability.Method == "" ||
		!strings.HasPrefix(capability.Path, "/") || !boundedOptionalIdentity(capability.Query, 8192) || capability.WireFormat == "" ||
		capability.Audience == "" {
		return fmt.Errorf("dispatch capability is incomplete")
	}
	if err := validateCandidateChain(capability.Candidates, capability.Fallback); err != nil {
		return fmt.Errorf("dispatch capability candidates: %w", err)
	}
	issued := time.Unix(capability.IssuedAt, 0)
	expires := time.Unix(capability.ExpiresAt, 0)
	if !expires.After(issued) || now.Before(issued.Add(-5*time.Second)) || !now.Before(expires) {
		return fmt.Errorf("dispatch capability is outside its validity window")
	}
	if maxLifetime <= 0 {
		maxLifetime = 30 * time.Second
	}
	if expires.Sub(issued) > maxLifetime {
		return fmt.Errorf("dispatch capability lifetime exceeds %s", maxLifetime)
	}
	return nil
}

func validBoundedIdentity(value string) bool {
	trimmed := strings.TrimSpace(value)
	return trimmed != "" && trimmed == value && len(value) <= 256 && !strings.ContainsRune(value, '\x00')
}

func validSHA256Hex(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func validRequestDigest(value string) bool {
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && base64.RawURLEncoding.EncodeToString(decoded) == value
}

// RequestDigest binds a capability to the exact provider-neutral HTTP request
// that leaves ExtProc. The backend invoker recomputes the same value after the
// gateway hop, before any provider credential or protocol mutation occurs.
func RequestDigest(method, path, query string, body []byte) string {
	hash := sha256.New()
	for _, part := range []string{method, path, query, strconv.Itoa(len(body))} {
		_, _ = hash.Write([]byte(strconv.Itoa(len(part))))
		_, _ = hash.Write([]byte{':'})
		_, _ = hash.Write([]byte(part))
		_, _ = hash.Write([]byte{';'})
	}
	_, _ = hash.Write(body)
	return base64.RawURLEncoding.EncodeToString(hash.Sum(nil))
}
