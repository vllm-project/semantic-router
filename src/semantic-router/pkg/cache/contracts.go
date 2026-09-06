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

// CachePartition is the trusted hard-isolation boundary. Namespace contains
// only an opaque HMAC-derived user/team/tenant scope.
type CachePartition struct {
	Recipe        string `json:"recipe,omitempty"`
	Decision      string `json:"decision,omitempty"`
	RequestModel  string `json:"request_model,omitempty"`
	SelectedModel string `json:"selected_model,omitempty"`
	Protocol      string `json:"protocol,omitempty"`
	Namespace     string `json:"namespace,omitempty"`
	Epoch         string `json:"epoch,omitempty"`
	epochResolved bool
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
		escaped[index] = strings.ReplaceAll(part, ":", "%3A")
	}
	return strings.Join(escaped, ":")
}

type CacheIdentity struct {
	Partition                CachePartition `json:"partition"`
	ExactFingerprint         string         `json:"exact_fingerprint"`
	CompatibilityFingerprint string         `json:"compatibility_fingerprint,omitempty"`
	SemanticQuery            string         `json:"-"`
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
	Identity  CacheIdentity
	Threshold float32
}

type CacheWrite struct {
	Identity     CacheIdentity
	RequestID    string
	RequestBody  []byte
	ResponseBody []byte
	TTL          TTLPolicy
}

type CacheResult struct {
	ResponseBody []byte        `json:"-"`
	Found        bool          `json:"found"`
	HitKind      HitKind       `json:"hit_kind"`
	Source       CacheSource   `json:"source,omitempty"`
	Similarity   float32       `json:"similarity,omitempty"`
	Age          time.Duration `json:"-"`
	AgeKnown     bool          `json:"-"`
	ExpiresAt    time.Time     `json:"expires_at,omitempty"`
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
	LookupSemantic(context.Context, SemanticLookup) (CacheResult, error)
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
