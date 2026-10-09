package config

const (
	// ToolSessionStoreBackendLocal keeps sticky tool-set state in the router
	// process's memory only. Lost on restart/reload; that is documented,
	// intended behavior, not a defect (see PL-0042).
	ToolSessionStoreBackendLocal = "local"

	// ToolSessionStoreBackendRedis shares sticky tool-set state across
	// router replicas via Redis.
	ToolSessionStoreBackendRedis = "redis"

	// ToolSessionStoreDefaultTTLSeconds is the sliding idle-expiry default:
	// every successful reuse/growth update refreshes it.
	ToolSessionStoreDefaultTTLSeconds = 1800
	ToolSessionStoreMinTTLSeconds     = 1
	ToolSessionStoreMaxTTLSeconds     = 604800

	// ToolSessionStoreDefaultMaxSessions bounds total tracked sessions
	// regardless of backend, so memory/keyspace usage stays bounded even
	// under high session cardinality.
	ToolSessionStoreDefaultMaxSessions = 10000
	ToolSessionStoreMinMaxSessions     = 1
	ToolSessionStoreMaxMaxSessions     = 100000

	// ToolSessionStoreDefaultMaxSessionsByIdentity bounds sessions per
	// authenticated principal, independent of the global cap.
	ToolSessionStoreDefaultMaxSessionsByIdentity = 128
	ToolSessionStoreMinMaxSessionsByIdentity     = 1

	// ToolSessionStoreDefaultMaxStateBytes bounds one session's encoded
	// state size — identities and bounded metadata only, never full tool
	// definitions (see StickyToolSelectionConfig).
	ToolSessionStoreDefaultMaxStateBytes = 16384
	ToolSessionStoreMinMaxStateBytes     = 1024
	ToolSessionStoreMaxMaxStateBytes     = 65536

	// ToolSessionStoreDefaultTimeoutMs bounds one store operation
	// (including CAS retries for the Redis backend) so a degraded store
	// never blocks the inference request it's attached to.
	ToolSessionStoreDefaultTimeoutMs = 50
	ToolSessionStoreMinTimeoutMs     = 1
	ToolSessionStoreMaxTimeoutMs     = 1000

	// ToolSessionStoreDefaultRedisKeyPrefix namespaces this feature's Redis
	// keys from every other store sharing the same database.
	ToolSessionStoreDefaultRedisKeyPrefix = "vsr:session-tools:v1:"
)

// ToolSessionStoreConfig configures the shared storage backend for
// session-scoped sticky tool-set selection (issue #3347,
// global.stores.tool_sessions). Constructing no store at all — not even the
// local backend — unless at least one decision's tool_selection.sticky is
// enabled is a caller responsibility (extproc wiring), not this package's.
//
// Numeric fields are pointers, not plain ints: for every one of them, 0 is
// outside the documented valid range and must be rejected by Validate, not
// silently treated as "unset -> apply the default". A plain int's zero
// value cannot carry that distinction from "the field was omitted". See
// StickyToolSelectionConfig's MaxTools/MaxNewToolsPerTurn for the same
// pattern established in TASK-01, applied here from the start.
type ToolSessionStoreConfig struct {
	Backend               string                  `json:"backend,omitempty" yaml:"backend,omitempty"`
	TTLSeconds            *int                    `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`
	MaxSessions           *int                    `json:"max_sessions,omitempty" yaml:"max_sessions,omitempty"`
	MaxSessionsByIdentity *int                    `json:"max_sessions_per_identity,omitempty" yaml:"max_sessions_per_identity,omitempty"`
	MaxStateBytes         *int                    `json:"max_state_bytes,omitempty" yaml:"max_state_bytes,omitempty"`
	TimeoutMs             *int                    `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
	Redis                 *ToolSessionRedisConfig `json:"redis,omitempty" yaml:"redis,omitempty"`
}

// ToolSessionRedisConfig configures the Redis backend for
// global.stores.tool_sessions. Forbidden (must be nil) when
// ToolSessionStoreConfig.Backend is "local"; required with a non-empty
// Address when Backend is "redis".
type ToolSessionRedisConfig struct {
	Address   string `json:"address,omitempty" yaml:"address,omitempty"`
	Password  string `json:"password,omitempty" yaml:"password,omitempty"`
	Database  int    `json:"database,omitempty" yaml:"database,omitempty"`
	KeyPrefix string `json:"key_prefix,omitempty" yaml:"key_prefix,omitempty"`
}

// EffectiveBackend returns the configured backend, or
// ToolSessionStoreBackendLocal when unset.
func (s *ToolSessionStoreConfig) EffectiveBackend() string {
	if s == nil || s.Backend == "" {
		return ToolSessionStoreBackendLocal
	}
	return s.Backend
}

// EffectiveTTLSeconds returns the configured ttl_seconds, or
// ToolSessionStoreDefaultTTLSeconds when unset.
func (s *ToolSessionStoreConfig) EffectiveTTLSeconds() int {
	if s == nil || s.TTLSeconds == nil {
		return ToolSessionStoreDefaultTTLSeconds
	}
	return *s.TTLSeconds
}

// EffectiveMaxSessions returns the configured max_sessions, or
// ToolSessionStoreDefaultMaxSessions when unset.
func (s *ToolSessionStoreConfig) EffectiveMaxSessions() int {
	if s == nil || s.MaxSessions == nil {
		return ToolSessionStoreDefaultMaxSessions
	}
	return *s.MaxSessions
}

// EffectiveMaxSessionsByIdentity returns the configured
// max_sessions_per_identity, or ToolSessionStoreDefaultMaxSessionsByIdentity
// when unset.
func (s *ToolSessionStoreConfig) EffectiveMaxSessionsByIdentity() int {
	if s == nil || s.MaxSessionsByIdentity == nil {
		return ToolSessionStoreDefaultMaxSessionsByIdentity
	}
	return *s.MaxSessionsByIdentity
}

// EffectiveMaxStateBytes returns the configured max_state_bytes, or
// ToolSessionStoreDefaultMaxStateBytes when unset.
func (s *ToolSessionStoreConfig) EffectiveMaxStateBytes() int {
	if s == nil || s.MaxStateBytes == nil {
		return ToolSessionStoreDefaultMaxStateBytes
	}
	return *s.MaxStateBytes
}

// EffectiveTimeoutMs returns the configured timeout_ms, or
// ToolSessionStoreDefaultTimeoutMs when unset.
func (s *ToolSessionStoreConfig) EffectiveTimeoutMs() int {
	if s == nil || s.TimeoutMs == nil {
		return ToolSessionStoreDefaultTimeoutMs
	}
	return *s.TimeoutMs
}

// EffectiveRedisKeyPrefix returns the configured redis.key_prefix, or
// ToolSessionStoreDefaultRedisKeyPrefix when unset or the Redis block is
// absent.
func (s *ToolSessionStoreConfig) EffectiveRedisKeyPrefix() string {
	if s == nil || s.Redis == nil || s.Redis.KeyPrefix == "" {
		return ToolSessionStoreDefaultRedisKeyPrefix
	}
	return s.Redis.KeyPrefix
}
