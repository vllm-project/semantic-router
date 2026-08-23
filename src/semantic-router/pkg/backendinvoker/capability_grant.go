package backendinvoker

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const dispatchGrantPrefix = "vdg"

// DispatchGrant authorizes an authenticated internal Router hop to mint the
// exact request-bound capability only after its final body is known. It never
// authorizes a backend call by itself.
type DispatchGrant struct {
	NamespaceID     string              `json:"namespaceId"`
	QuotaPartition  string              `json:"quotaPartition"`
	PublicationID   string              `json:"publicationId"`
	RuntimeEpoch    uint64              `json:"runtimeEpoch"`
	RoutingRevision int64               `json:"routingRevision"`
	RoutingDigest   string              `json:"routingDigest"`
	AdmissionID     string              `json:"admissionId"`
	AdmissionDigest string              `json:"admissionDigest"`
	RequestID       string              `json:"requestId"`
	Candidates      []DispatchCandidate `json:"candidates"`
	Fallback        FallbackPolicy      `json:"fallback"`
	Audience        string              `json:"audience"`
	IssuedAt        int64               `json:"issuedAt"`
	ExpiresAt       int64               `json:"expiresAt"`
}

type DispatchGrantIssueRequest struct {
	NamespaceID     string
	QuotaPartition  string
	PublicationID   string
	RuntimeEpoch    uint64
	RoutingRevision int64
	RoutingDigest   string
	AdmissionID     string
	AdmissionDigest string
	RequestID       string
	Candidates      []DispatchCandidate
	Fallback        FallbackPolicy
}

type DispatchFinalRequest struct {
	Method     string
	Path       string
	Query      string
	WireFormat llmprotocol.WireFormat
	Body       []byte
}

// VerifiedDispatchGrant is an opaque process-local proof. Callers cannot
// construct a value accepted by IssueFromGrant.
type VerifiedDispatchGrant struct {
	grant DispatchGrant
	owner *CapabilityIssuer
}

type DispatchGrantClaims struct {
	NamespaceID     string
	QuotaPartition  string
	PublicationID   string
	RuntimeEpoch    uint64
	RoutingRevision int64
	RoutingDigest   string
	RequestID       string
}

// VerifiedGrantClaims exposes only immutable routing and request identity from
// process-local verified proof. It never makes serialized grants constructible.
func VerifiedGrantClaims(verified VerifiedDispatchGrant) (DispatchGrantClaims, bool) {
	if verified.owner == nil {
		return DispatchGrantClaims{}, false
	}
	grant := verified.grant
	return DispatchGrantClaims{
		NamespaceID: grant.NamespaceID, QuotaPartition: grant.QuotaPartition,
		PublicationID: grant.PublicationID, RuntimeEpoch: grant.RuntimeEpoch,
		RoutingRevision: grant.RoutingRevision, RoutingDigest: grant.RoutingDigest,
		RequestID: grant.RequestID,
	}, true
}

// VerifiedGrantCandidates returns a defensive copy of the immutable candidate
// chain carried by process-local verified proof.
func VerifiedGrantCandidates(verified VerifiedDispatchGrant) ([]DispatchCandidate, bool) {
	if verified.owner == nil {
		return nil, false
	}
	return cloneDispatchCandidates(verified.grant.Candidates), true
}

func (issuer *CapabilityIssuer) IssueGrant(request DispatchGrantIssueRequest) (string, error) {
	if issuer == nil {
		return "", fmt.Errorf("dispatch capability issuer is unavailable")
	}
	issuer.mu.RLock()
	defer issuer.mu.RUnlock()
	if issuer.closed {
		return "", fmt.Errorf("dispatch capability issuer is closed")
	}
	now := issuer.now().UTC()
	grant := DispatchGrant{
		NamespaceID: request.NamespaceID, QuotaPartition: request.QuotaPartition,
		PublicationID: request.PublicationID, RuntimeEpoch: request.RuntimeEpoch,
		RoutingRevision: request.RoutingRevision, RoutingDigest: request.RoutingDigest,
		AdmissionID:     request.AdmissionID,
		AdmissionDigest: request.AdmissionDigest,
		RequestID:       request.RequestID,
		Candidates:      cloneDispatchCandidates(request.Candidates),
		Fallback:        cloneFallbackPolicy(request.Fallback), Audience: issuer.audience,
		IssuedAt: now.Unix(), ExpiresAt: now.Add(issuer.lifetime).Unix(),
	}
	return issuer.keyring.signGrant(grant, now)
}

func (issuer *CapabilityIssuer) VerifyGrant(token string) (VerifiedDispatchGrant, error) {
	if issuer == nil {
		return VerifiedDispatchGrant{}, fmt.Errorf("dispatch capability issuer is unavailable")
	}
	issuer.mu.RLock()
	defer issuer.mu.RUnlock()
	if issuer.closed {
		return VerifiedDispatchGrant{}, fmt.Errorf("dispatch capability issuer is closed")
	}
	grant, err := issuer.keyring.verifyGrant(token, issuer.audience, issuer.now().UTC())
	if err != nil {
		return VerifiedDispatchGrant{}, err
	}
	return VerifiedDispatchGrant{grant: grant, owner: issuer}, nil
}

func (issuer *CapabilityIssuer) IssueFromGrant(
	verified VerifiedDispatchGrant,
	request DispatchFinalRequest,
) (string, error) {
	if issuer == nil || verified.owner != issuer {
		return "", fmt.Errorf("verified dispatch grant is unavailable")
	}
	issuer.mu.RLock()
	defer issuer.mu.RUnlock()
	if issuer.closed {
		return "", fmt.Errorf("dispatch capability issuer is closed")
	}
	now := issuer.now().UTC()
	grant := verified.grant
	if err := validateDispatchGrant(grant, now, issuer.keyring.MaxLifetime); err != nil {
		return "", err
	}
	expiresAt := now.Add(issuer.lifetime)
	grantExpiry := time.Unix(grant.ExpiresAt, 0)
	if grantExpiry.Before(expiresAt) {
		expiresAt = grantExpiry
	}
	capability := DispatchCapability{
		NamespaceID: grant.NamespaceID, QuotaPartition: grant.QuotaPartition,
		PublicationID: grant.PublicationID, RuntimeEpoch: grant.RuntimeEpoch,
		RoutingRevision: grant.RoutingRevision, RoutingDigest: grant.RoutingDigest,
		AdmissionID:     grant.AdmissionID,
		AdmissionDigest: grant.AdmissionDigest,
		RequestID:       grant.RequestID,
		Candidates:      cloneDispatchCandidates(grant.Candidates),
		Fallback:        cloneFallbackPolicy(grant.Fallback),
		RequestDigest:   RequestDigest(request.Method, request.Path, request.Query, request.Body),
		Method:          request.Method, Path: request.Path, Query: request.Query, WireFormat: request.WireFormat, Audience: issuer.audience,
		IssuedAt: now.Unix(), ExpiresAt: expiresAt.Unix(),
	}
	return issuer.keyring.Sign(capability, now)
}

func (keyring SigningKeyring) signGrant(grant DispatchGrant, now time.Time) (string, error) {
	key, ok := keyring.Keys[keyring.ActiveVersion]
	if !ok || len(key) < sha256.Size {
		return "", fmt.Errorf("active dispatch signing key is unavailable")
	}
	if err := validateDispatchGrant(grant, now, keyring.MaxLifetime); err != nil {
		return "", err
	}
	payload, err := json.Marshal(grant)
	if err != nil {
		return "", fmt.Errorf("marshal dispatch grant: %w", err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	message := dispatchGrantPrefix + "." + keyring.ActiveVersion + "." + encoded
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(message))
	return message + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (keyring SigningKeyring) verifyGrant(token, audience string, now time.Time) (DispatchGrant, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 4 || parts[0] != dispatchGrantPrefix {
		return DispatchGrant{}, fmt.Errorf("invalid dispatch grant")
	}
	key, ok := keyring.Keys[parts[1]]
	if !ok || len(key) < sha256.Size {
		return DispatchGrant{}, fmt.Errorf("unknown dispatch signing key")
	}
	provided, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(provided) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(provided) != parts[3] {
		return DispatchGrant{}, fmt.Errorf("invalid dispatch grant signature")
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(strings.Join(parts[:3], ".")))
	expected := mac.Sum(nil)
	if len(provided) != len(expected) || subtle.ConstantTimeCompare(provided, expected) != 1 {
		return DispatchGrant{}, fmt.Errorf("invalid dispatch grant signature")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != parts[2] {
		return DispatchGrant{}, fmt.Errorf("invalid dispatch grant payload")
	}
	var grant DispatchGrant
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&grant); err != nil {
		return DispatchGrant{}, fmt.Errorf("invalid dispatch grant payload")
	}
	if grant.Audience != audience {
		return DispatchGrant{}, fmt.Errorf("dispatch grant audience mismatch")
	}
	if err := validateDispatchGrant(grant, now, keyring.MaxLifetime); err != nil {
		return DispatchGrant{}, err
	}
	return grant, nil
}

func validateDispatchGrant(grant DispatchGrant, now time.Time, maxLifetime time.Duration) error {
	if grant.NamespaceID == "" || grant.QuotaPartition == "" ||
		grant.PublicationID == "" || grant.RuntimeEpoch == 0 ||
		grant.RoutingRevision <= 0 || !validSHA256Hex(grant.RoutingDigest) ||
		grant.AdmissionID == "" || !validSHA256Hex(grant.AdmissionDigest) ||
		!validBoundedIdentity(grant.RequestID) || grant.Audience == "" {
		return fmt.Errorf("dispatch grant is incomplete")
	}
	if err := validateCandidateChain(grant.Candidates, grant.Fallback); err != nil {
		return fmt.Errorf("dispatch grant candidates: %w", err)
	}
	issued := time.Unix(grant.IssuedAt, 0)
	expires := time.Unix(grant.ExpiresAt, 0)
	if !expires.After(issued) || now.Before(issued.Add(-5*time.Second)) || !now.Before(expires) {
		return fmt.Errorf("dispatch grant is outside its validity window")
	}
	if maxLifetime <= 0 || expires.Sub(issued) > maxLifetime {
		return fmt.Errorf("dispatch grant lifetime is invalid")
	}
	return nil
}
