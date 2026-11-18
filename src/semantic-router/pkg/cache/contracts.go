package cache

import (
	"context"
	"errors"
	"strings"
	"time"
)

var ErrUnsupported = errors.New("response cache operation is unsupported by this backend")

type HitKind string

const (
	HitKindMiss     HitKind = "miss"
	HitKindExact    HitKind = "exact"
	HitKindSemantic HitKind = "semantic"
)

type CacheSource string

const (
	CacheSourceL1 CacheSource = "l1"
	CacheSourceL2 CacheSource = "l2"
)

// NegationGuardOutcome reports whether the English lexical guard could judge
// the question pair behind a served semantic hit.
type NegationGuardOutcome string

const (
	// NegationGuardChecked means both questions use the same words, or differ
	// only in negation cues.
	NegationGuardChecked NegationGuardOutcome = "checked"
	// NegationGuardNotApplicable means another word changed, so the hit rests
	// on vector similarity alone.
	NegationGuardNotApplicable NegationGuardOutcome = "not_applicable"
)

// CachePartition is the trusted hard-isolation boundary. Namespace contains
// only an opaque HMAC-derived user/team/tenant scope.
type CachePartition struct {
	Recipe                    string `json:"recipe,omitempty"`
	Decision                  string `json:"decision,omitempty"`
	RequestModel              string `json:"request_model,omitempty"`
	SelectedModel             string `json:"selected_model,omitempty"`
	Protocol                  string `json:"protocol,omitempty"`
	Namespace                 string `json:"namespace,omitempty"`
	Epoch                     string `json:"epoch,omitempty"`
	epochResolved             bool
	configuredEpoch           string
	resolvedEmbeddingIdentity string
}

func (p CachePartition) Key() string {
	return joinPartitionParts(
		p.Recipe,
		p.Decision,
		p.RequestModel,
		p.SelectedModel,
		p.Protocol,
		p.Namespace,
		p.Epoch,
	)
}

func joinPartitionParts(parts ...string) string {
	escaped := make([]string, len(parts))
	for index, part := range parts {
		// Escape '%' before ':' so the encoding stays injective: a part holding
		// the literal "%3A" must not produce the same token as a part holding ':'.
		escaped[index] = strings.ReplaceAll(strings.ReplaceAll(part, "%", "%25"), ":", "%3A")
	}
	return strings.Join(escaped, ":")
}

type CacheIdentity struct {
	Partition                CachePartition `json:"partition"`
	ExactFingerprint         string         `json:"exact_fingerprint"`
	CompatibilityFingerprint string         `json:"compatibility_fingerprint,omitempty"`
	SemanticQuery            string         `json:"-"`
}

func (i CacheIdentity) SemanticPartitionKey() string {
	return CombineFingerprints("semantic", i.Partition.Key(), i.CompatibilityFingerprint)
}

type TTLPolicy struct {
	UseDefault bool
	NoStore    bool
	Duration   time.Duration
}

func DefaultTTL() TTLPolicy {
	return TTLPolicy{UseDefault: true}
}

func NoStoreTTL() TTLPolicy {
	return TTLPolicy{NoStore: true}
}

func TTL(duration time.Duration) TTLPolicy {
	if duration <= 0 {
		return NoStoreTTL()
	}
	return TTLPolicy{Duration: duration}
}

func TTLPolicyFromLegacySeconds(seconds int) TTLPolicy {
	switch {
	case seconds < 0:
		return DefaultTTL()
	case seconds == 0:
		return NoStoreTTL()
	default:
		return TTL(time.Duration(seconds) * time.Second)
	}
}

func (p TTLPolicy) LegacySeconds() int {
	switch {
	case p.NoStore:
		return 0
	case p.UseDefault:
		return -1
	default:
		return int(p.Duration / time.Second)
	}
}

type ExactLookup struct {
	Identity CacheIdentity
	MaxAge   *time.Duration
}

type SemanticLookup struct {
	Identity CacheIdentity
	// Bounds bound the cached requests the lookup recalls.
	Bounds SimilarityBounds
	MaxAge *time.Duration
}

// SemanticRecall is what a semantic lookup recalled: the fresh cached entries
// within its bounds, most similar first, each with the cached request it
// answered, and the closest similarity it saw, kept or not.
type SemanticRecall struct {
	Candidates []CacheResult
	Closest    float32
}

type CacheWrite struct {
	Identity     CacheIdentity
	RequestID    string
	RequestBody  []byte
	ResponseBody []byte
	TTL          TTLPolicy
}

type CacheResult struct {
	ResponseBody  []byte               `json:"-"`
	Found         bool                 `json:"found"`
	HitKind       HitKind              `json:"hit_kind"`
	Source        CacheSource          `json:"source,omitempty"`
	Similarity    float32              `json:"similarity,omitempty"`
	Query         string               `json:"-"`
	NegationGuard NegationGuardOutcome `json:"negation_guard,omitempty"`
	Age           time.Duration        `json:"-"`
	AgeKnown      bool                 `json:"-"`
	ExpiresAt     time.Time            `json:"expires_at,omitempty"`
	// RequestID names the request that wrote a semantic hit's entry, so
	// the store can count the hit its caller serves.
	RequestID string `json:"-"`
}

type BackendCapabilities struct {
	Backend          CacheBackendType `json:"backend"`
	Exact            bool             `json:"exact"`
	Semantic         bool             `json:"semantic"`
	PerEntryTTL      bool             `json:"per_entry_ttl"`
	Stats            bool             `json:"stats"`
	Invalidate       bool             `json:"invalidate"`
	Flush            bool             `json:"flush"`
	Shared           bool             `json:"shared"`
	ConsistencyModel string           `json:"consistency_model"`
}

type ExactStore interface {
	LookupExact(context.Context, ExactLookup) (CacheResult, error)
	StoreExact(context.Context, CacheWrite) error
}

type SemanticStore interface {
	LookupSemantic(context.Context, SemanticLookup) (SemanticRecall, error)
	StoreSemantic(context.Context, CacheWrite) error
}

type CacheLifecycle interface {
	Health(context.Context) error
	Close() error
	Stats(context.Context) (CacheStats, error)
	Capabilities() BackendCapabilities
}

type CacheSelector struct {
	Partition *CachePartition `json:"partition,omitempty"`
	Epoch     string          `json:"epoch,omitempty"`
}

type InvalidationResult struct {
	DryRun          bool  `json:"dry_run"`
	MatchedEstimate int64 `json:"matched_estimate"`
	Deleted         int64 `json:"deleted"`
	// CountUnknown reports that the backend cannot count the entries the
	// call reaches, so the zero counts do not mean that it reaches none.
	CountUnknown bool `json:"count_unknown,omitempty"`
}

type CacheAdmin interface {
	Invalidate(context.Context, CacheSelector, bool) (InvalidationResult, error)
	Flush(context.Context, CacheSelector) (InvalidationResult, error)
}

type TypedCacheStore interface {
	ExactStore
	SemanticStore
	CacheLifecycle
}
