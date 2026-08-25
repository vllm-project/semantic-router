// Package managementcommand defines the durable identity of one mutating
// Management API command. It contains no transport or storage behavior.
package managementcommand

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	digestSize                 = sha256.Size
	maximumRetainedHMACVersion = 8
)

var (
	ErrConflict               = errors.New("management command idempotency conflict")
	ErrHMACVersionUnavailable = errors.New("management command HMAC version is unavailable")

	endpointPattern     = regexp.MustCompile(`^/[a-z0-9][A-Za-z0-9._~!$&'()*+,;=:@/{}/-]{0,511}$`)
	resourceTypePattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
	hmacVersionPattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9-]{0,63}$`)
)

type ScopeKind string

const (
	ScopeCluster   ScopeKind = "cluster"
	ScopeNamespace ScopeKind = "namespace"
)

// CommandScope is an explicit sum type. Cluster commands never carry a
// Namespace identifier; Namespace commands always carry one. A sentinel UUID
// is deliberately forbidden because it would blur authorization and durable
// idempotency ownership.
type CommandScope struct {
	Kind        ScopeKind
	NamespaceID string
}

func ClusterCommandScope() CommandScope { return CommandScope{Kind: ScopeCluster} }

func NamespaceCommandScope(namespaceID string) CommandScope {
	return CommandScope{Kind: ScopeNamespace, NamespaceID: namespaceID}
}

func (scope CommandScope) Validate() error {
	switch scope.Kind {
	case ScopeCluster:
		if scope.NamespaceID != "" {
			return errors.New("cluster management command scope cannot carry a namespace")
		}
	case ScopeNamespace:
		if !canonicalUUID(scope.NamespaceID) {
			return errors.New("namespace management command scope requires a canonical namespace UUID")
		}
	default:
		return errors.New("management command scope kind is invalid")
	}
	return nil
}

// VersionedDigest is one HMAC-key-version binding of a command. The raw
// Idempotency-Key and canonical request are never retained here.
type VersionedDigest struct {
	HMACVersion   string
	KeyDigest     [digestSize]byte
	RequestDigest [digestSize]byte
}

// Command is the non-secret, HMAC-bound identity of a synchronous or
// asynchronous Management mutation. Raw Idempotency-Key values and request
// bodies are never persisted.
type Command struct {
	Scope       CommandScope
	PrincipalID string
	Endpoint    string
	ExpiresAt   time.Time

	active     VersionedDigest
	candidates []VersionedDigest
	lockDigest [digestSize]byte
}

// ResourceResult is the immutable replay value for a synchronous command.
// Mutable response bodies and submitted secrets never enter the idempotency
// table, so later resource edits cannot change the original receipt.
type ResourceResult struct {
	ResourceType     string
	ResourceID       string
	ResourceRevision uint64
	ResponseStatus   int
}

// OperationResult is the durable replay value for an asynchronous command.
type OperationResult struct {
	OperationID     string
	DesiredRevision *uint64
	ResponseStatus  int
}

// StoredResult is exactly one resource or operation result.
type StoredResult struct {
	Resource  *ResourceResult
	Operation *OperationResult
	Secret    *SecretResponse
	ExpiresAt time.Time
}

// SecretResponse is an encrypted, bounded replay body attached only to a
// synchronous resource command. Ciphertext is opaque to this package; the
// owning domain binds it to resource identity with application-specific AAD.
type SecretResponse struct {
	Ciphertext []byte
	Nonce      []byte
	KEKVersion string
	ExpiresAt  time.Time
}

func (secret SecretResponse) Validate() error {
	if len(secret.Ciphertext) == 0 || len(secret.Nonce) == 0 ||
		!hmacVersionPattern.MatchString(secret.KEKVersion) || secret.ExpiresAt.IsZero() {
		return errors.New("management command secret response is invalid")
	}
	return nil
}

func (command Command) Validate(now time.Time) error {
	if err := command.Scope.Validate(); err != nil {
		return err
	}
	if !canonicalUUID(command.PrincipalID) {
		return errors.New("management command principal must be a canonical UUID")
	}
	if !endpointPattern.MatchString(command.Endpoint) || strings.Contains(command.Endpoint, "//") {
		return errors.New("management command endpoint is invalid")
	}
	if len(command.candidates) < 1 || len(command.candidates) > maximumRetainedHMACVersion ||
		!validVersionedDigest(command.active) || zeroDigest(command.lockDigest) {
		return errors.New("management command versioned digests are invalid")
	}
	seen := make(map[string]struct{}, len(command.candidates))
	activeFound := false
	for _, candidate := range command.candidates {
		if !validVersionedDigest(candidate) {
			return errors.New("management command versioned digests are invalid")
		}
		if _, duplicate := seen[candidate.HMACVersion]; duplicate {
			return errors.New("management command HMAC versions must be unique")
		}
		seen[candidate.HMACVersion] = struct{}{}
		if candidate == command.active {
			activeFound = true
		}
	}
	if !activeFound {
		return errors.New("management command active HMAC version is not retained")
	}
	if now.IsZero() || command.ExpiresAt.IsZero() || !command.ExpiresAt.After(now) {
		return errors.New("management command expiry must be in the future")
	}
	return nil
}

// ActiveDigest is the key version used to persist a newly completed command.
func (command Command) ActiveDigest() VersionedDigest { return command.active }

// CandidateDigests returns defensive copies of every retained version used to
// recognize a replay during key rotation.
func (command Command) CandidateDigests() []VersionedDigest {
	return append([]VersionedDigest(nil), command.candidates...)
}

// AdvisoryLockKey is a stable, process-only SHA-256 binding of the raw
// Idempotency-Key and command scope. It serializes replicas across HMAC key
// rotations and must never be stored or logged.
func (command Command) AdvisoryLockKey() int64 {
	unsigned := binary.BigEndian.Uint64(command.lockDigest[:8]) & uint64(math.MaxInt64)
	// #nosec G115 -- masking the sign bit bounds the advisory-lock key to MaxInt64.
	return int64(unsigned)
}

func (result ResourceResult) Validate() error {
	if !resourceTypePattern.MatchString(result.ResourceType) {
		return errors.New("management command resource type is invalid")
	}
	if !canonicalResourceID(result.ResourceID) {
		return errors.New("management command resource ID is invalid")
	}
	if result.ResourceRevision == 0 {
		return errors.New("management command resource revision is required")
	}
	if result.ResponseStatus < 200 || result.ResponseStatus > 299 {
		return errors.New("management command response status must be successful")
	}
	return nil
}

func (result OperationResult) Validate() error {
	if !canonicalUUID(result.OperationID) {
		return errors.New("management command operation ID must be a canonical UUID")
	}
	if result.DesiredRevision != nil && *result.DesiredRevision == 0 {
		return errors.New("management command desired revision must be positive when present")
	}
	if result.ResponseStatus < 200 || result.ResponseStatus > 299 {
		return errors.New("management command response status must be successful")
	}
	return nil
}

func (result StoredResult) Validate() error {
	if result.ExpiresAt.IsZero() || (result.Resource == nil) == (result.Operation == nil) {
		return errors.New("management command result must contain exactly one result kind")
	}
	if result.Secret != nil {
		if result.Resource == nil || result.Secret.ExpiresAt.After(result.ExpiresAt) {
			return errors.New("management command secret response must belong to a live resource result")
		}
		if err := result.Secret.Validate(); err != nil {
			return err
		}
	}
	if result.Resource != nil {
		return result.Resource.Validate()
	}
	return result.Operation.Validate()
}

// SameRequest compares a stored request digest for a retained HMAC version
// without data-dependent timing.
func (command Command) SameRequest(hmacVersion string, other []byte) bool {
	for _, candidate := range command.candidates {
		if candidate.HMACVersion == hmacVersion {
			return len(other) == digestSize &&
				subtle.ConstantTimeCompare(candidate.RequestDigest[:], other) == 1
		}
	}
	return false
}

// Codec HMAC-binds raw idempotency keys and canonical requests to their exact
// namespace, principal, and endpoint. It prevents offline recovery of a
// secret-bearing request from its stored digest.
type Codec struct {
	activeVersion string
	keys          map[string][]byte
	versions      []string
}

func NewCodec(keyring securitykeyring.Symmetric) (*Codec, error) {
	if !hmacVersionPattern.MatchString(keyring.ActiveVersion) ||
		len(keyring.Keys) < 1 || len(keyring.Keys) > maximumRetainedHMACVersion {
		return nil, fmt.Errorf("management command HMAC keyring must contain 1-%d canonical versions", maximumRetainedHMACVersion)
	}
	codec := &Codec{
		activeVersion: keyring.ActiveVersion,
		keys:          make(map[string][]byte, len(keyring.Keys)),
		versions:      make([]string, 0, len(keyring.Keys)),
	}
	for version, key := range keyring.Keys {
		if !hmacVersionPattern.MatchString(version) || len(key) != digestSize {
			return nil, errors.New("management command HMAC versions require canonical names and exactly 256-bit keys")
		}
		codec.keys[version] = append([]byte(nil), key...)
		codec.versions = append(codec.versions, version)
	}
	if _, found := codec.keys[codec.activeVersion]; !found {
		return nil, errors.New("management command active HMAC version is not retained")
	}
	slices.Sort(codec.versions)
	return codec, nil
}

func (codec *Codec) Bind(
	scope CommandScope,
	principalID string,
	endpoint string,
	idempotencyKey string,
	canonicalRequest []byte,
	now time.Time,
	expiresAt time.Time,
) (Command, error) {
	if codec == nil || len(codec.keys) == 0 {
		return Command{}, errors.New("management command codec is unavailable")
	}
	if len(idempotencyKey) < 16 || len(idempotencyKey) > 200 ||
		strings.TrimSpace(idempotencyKey) != idempotencyKey || strings.ContainsAny(idempotencyKey, "\x00\r\n\t ") {
		return Command{}, errors.New("management command idempotency key is invalid")
	}
	if len(canonicalRequest) == 0 || len(canonicalRequest) > 1<<20 {
		return Command{}, errors.New("management command canonical request is invalid")
	}
	command := Command{
		Scope: scope, PrincipalID: principalID, Endpoint: endpoint,
		ExpiresAt: expiresAt.UTC(),
	}
	if err := scope.Validate(); err != nil {
		return Command{}, err
	}
	command.lockDigest = stableLockDigest(scope, principalID, endpoint, []byte(idempotencyKey))
	command.candidates = make([]VersionedDigest, 0, len(codec.versions))
	for _, version := range codec.versions {
		key := codec.keys[version]
		candidate := VersionedDigest{
			HMACVersion:   version,
			KeyDigest:     codec.digest(key, "key", scope, principalID, endpoint, []byte(idempotencyKey)),
			RequestDigest: codec.digest(key, "request", scope, principalID, endpoint, canonicalRequest),
		}
		command.candidates = append(command.candidates, candidate)
		if version == codec.activeVersion {
			command.active = candidate
		}
	}
	if err := command.Validate(now.UTC()); err != nil {
		return Command{}, fmt.Errorf("bind management command: %w", err)
	}
	return command, nil
}

// RecognizesHMACVersion reports whether this codec can validate commands
// persisted under the requested version.
func (codec *Codec) RecognizesHMACVersion(version string) bool {
	if codec == nil {
		return false
	}
	_, found := codec.keys[version]
	return found
}

// Close erases retained command HMAC material. The codec is unusable after
// Close and must be retired with its owning managed process runtime.
func (codec *Codec) Close() error {
	if codec == nil {
		return nil
	}
	for _, key := range codec.keys {
		for index := range key {
			key[index] = 0
		}
	}
	codec.activeVersion = ""
	codec.keys = nil
	codec.versions = nil
	return nil
}

func (codec *Codec) digest(key []byte, kind string, scope CommandScope, principalID, endpoint string, value []byte) [digestSize]byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte("vllm-sr/management-command/v1\x00"))
	for _, field := range []string{kind, string(scope.Kind), scope.NamespaceID, principalID, endpoint} {
		_, _ = mac.Write([]byte(field))
		_, _ = mac.Write([]byte{0})
	}
	_, _ = mac.Write(value)
	var result [digestSize]byte
	copy(result[:], mac.Sum(nil))
	return result
}

func stableLockDigest(scope CommandScope, principalID, endpoint string, rawKey []byte) [digestSize]byte {
	hash := sha256.New()
	_, _ = hash.Write([]byte("vllm-sr/management-command-lock/v1\x00"))
	for _, field := range []string{string(scope.Kind), scope.NamespaceID, principalID, endpoint} {
		_, _ = hash.Write([]byte(field))
		_, _ = hash.Write([]byte{0})
	}
	_, _ = hash.Write(rawKey)
	var result [digestSize]byte
	copy(result[:], hash.Sum(nil))
	return result
}

func validVersionedDigest(digest VersionedDigest) bool {
	return hmacVersionPattern.MatchString(digest.HMACVersion) &&
		!zeroDigest(digest.KeyDigest) && !zeroDigest(digest.RequestDigest)
}

func zeroDigest(value [digestSize]byte) bool {
	var zero [digestSize]byte
	return subtle.ConstantTimeCompare(value[:], zero[:]) == 1
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func canonicalResourceID(value string) bool {
	if len(value) < 1 || len(value) > 512 || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x21 || character > 0x7e {
			return false
		}
	}
	return true
}
