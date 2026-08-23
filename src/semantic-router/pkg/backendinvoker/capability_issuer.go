package backendinvoker

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// CapabilityIssuerOptions defines the process-owned authority used by ExtProc
// to authorize one immutable ordered dispatch chain. The issuer copies key
// material and must be closed by its owner.
type CapabilityIssuerOptions struct {
	Audience string
	Keyring  SigningKeyring
	Lifetime time.Duration
	Now      func() time.Time
}

// CapabilityIssueRequest contains the complete non-secret dispatch identity
// plus the final provider-neutral request emitted by ExtProc.
type CapabilityIssueRequest struct {
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
	Method          string
	Path            string
	Query           string
	WireFormat      llmprotocol.WireFormat
	Body            []byte
}

// CapabilityIssuer signs short-lived, audience-bound dispatch capabilities.
// It is safe for concurrent use.
type CapabilityIssuer struct {
	mu       sync.RWMutex
	audience string
	keyring  SigningKeyring
	lifetime time.Duration
	now      func() time.Time
	closed   bool
}

func NewCapabilityIssuer(options CapabilityIssuerOptions) (*CapabilityIssuer, error) {
	audience := strings.TrimSpace(options.Audience)
	if audience == "" || audience != options.Audience || len(audience) > 256 || strings.ContainsAny(audience, "\x00\r\n\t ") {
		return nil, fmt.Errorf("dispatch capability audience is invalid")
	}
	if options.Lifetime <= 0 || options.Keyring.MaxLifetime <= 0 || options.Lifetime > options.Keyring.MaxLifetime {
		return nil, fmt.Errorf("dispatch capability lifetime is invalid")
	}
	keyring, err := cloneSigningKeyring(options.Keyring)
	if err != nil {
		return nil, err
	}
	clock := options.Now
	if clock == nil {
		clock = time.Now
	}
	return &CapabilityIssuer{
		audience: audience, keyring: keyring, lifetime: options.Lifetime, now: clock,
	}, nil
}

func (issuer *CapabilityIssuer) Issue(request CapabilityIssueRequest) (string, error) {
	if issuer == nil {
		return "", fmt.Errorf("dispatch capability issuer is unavailable")
	}
	issuer.mu.RLock()
	defer issuer.mu.RUnlock()
	if issuer.closed {
		return "", fmt.Errorf("dispatch capability issuer is closed")
	}
	now := issuer.now().UTC()
	capability := DispatchCapability{
		NamespaceID: request.NamespaceID, QuotaPartition: request.QuotaPartition,
		PublicationID: request.PublicationID, RuntimeEpoch: request.RuntimeEpoch,
		RoutingRevision: request.RoutingRevision, RoutingDigest: request.RoutingDigest,
		AdmissionID:     request.AdmissionID,
		AdmissionDigest: request.AdmissionDigest,
		RequestID:       request.RequestID,
		Candidates:      cloneDispatchCandidates(request.Candidates),
		Fallback:        cloneFallbackPolicy(request.Fallback),
		RequestDigest:   RequestDigest(request.Method, request.Path, request.Query, request.Body),
		Method:          request.Method, Path: request.Path, Query: request.Query, WireFormat: request.WireFormat, Audience: issuer.audience,
		IssuedAt: now.Unix(), ExpiresAt: now.Add(issuer.lifetime).Unix(),
	}
	return issuer.keyring.Sign(capability, now)
}

// VerifyOutcome authenticates a private-dispatch result with the same
// audience, rotated keyring, and clock that issued the request capability.
// Callers must additionally bind the returned claims to their request-local
// candidate journal before treating any dispatch as attempted or served.
func (issuer *CapabilityIssuer) VerifyOutcome(token string) (DispatchOutcome, error) {
	if issuer == nil {
		return DispatchOutcome{}, fmt.Errorf("dispatch capability issuer is unavailable")
	}
	issuer.mu.RLock()
	defer issuer.mu.RUnlock()
	if issuer.closed {
		return DispatchOutcome{}, fmt.Errorf("dispatch capability issuer is closed")
	}
	return issuer.keyring.VerifyOutcome(token, issuer.audience, issuer.now().UTC())
}

func cloneDispatchCandidates(source []DispatchCandidate) []DispatchCandidate {
	return append([]DispatchCandidate(nil), source...)
}

func cloneFallbackPolicy(source FallbackPolicy) FallbackPolicy {
	return FallbackPolicy{On: append([]FallbackTrigger(nil), source.On...)}
}

func (issuer *CapabilityIssuer) Close() error {
	if issuer == nil {
		return nil
	}
	issuer.mu.Lock()
	defer issuer.mu.Unlock()
	if issuer.closed {
		return nil
	}
	issuer.closed = true
	zeroSigningKeyring(&issuer.keyring)
	issuer.audience = ""
	issuer.now = nil
	return nil
}

func cloneSigningKeyring(source SigningKeyring) (SigningKeyring, error) {
	if strings.TrimSpace(source.ActiveVersion) == "" || source.MaxLifetime <= 0 || len(source.Keys) == 0 {
		return SigningKeyring{}, fmt.Errorf("dispatch capability keyring is empty or invalid")
	}
	result := SigningKeyring{
		ActiveVersion: source.ActiveVersion,
		Keys:          make(map[string][]byte, len(source.Keys)),
		MaxLifetime:   source.MaxLifetime,
	}
	for version, key := range source.Keys {
		if strings.TrimSpace(version) == "" || len(key) < 32 {
			zeroSigningKeyring(&result)
			return SigningKeyring{}, fmt.Errorf("dispatch capability key %q is invalid", version)
		}
		result.Keys[version] = append([]byte(nil), key...)
	}
	if _, ok := result.Keys[result.ActiveVersion]; !ok {
		zeroSigningKeyring(&result)
		return SigningKeyring{}, fmt.Errorf("active dispatch capability key is unavailable")
	}
	return result, nil
}

func zeroSigningKeyring(keyring *SigningKeyring) {
	if keyring == nil {
		return
	}
	for _, key := range keyring.Keys {
		for index := range key {
			key[index] = 0
		}
	}
	*keyring = SigningKeyring{}
}
