package config

import (
	"fmt"
	"strings"
)

// validateGlobalToolSessionsContracts is the globalConfigContractValidators
// admission-time entry for global.stores.tool_sessions. Config parsing
// (ParseYAMLBytes) fails closed on an invalid block, rather than only
// failing later when TASK-03+'s sessiontools store actually gets
// constructed — most entries in globalConfigContractValidators are
// admission-time; VectorStoreConfig.Validate() is the outlier that's
// called only at store-construction time
// (routerruntime/vectorstore_runtime.go), not from this list.
func validateGlobalToolSessionsContracts(cfg *RouterConfig) error {
	if cfg == nil || cfg.ToolSessions == nil {
		return nil
	}
	return cfg.ToolSessions.Validate()
}

// Validate enforces global.stores.tool_sessions's bounds and backend
// contract. A nil receiver (the block omitted entirely) is valid — the
// store defaults to disabled/unconstructed at the caller level.
func (s *ToolSessionStoreConfig) Validate() error {
	if s == nil {
		return nil
	}
	backend, err := normalizeToolSessionStoreBackend(s.Backend)
	if err != nil {
		return err
	}
	if err := s.validateBackendRedisContract(backend); err != nil {
		return err
	}
	if err := s.validateBounds(); err != nil {
		return err
	}
	return nil
}

func normalizeToolSessionStoreBackend(backend string) (string, error) {
	if backend == "" {
		return ToolSessionStoreBackendLocal, nil
	}
	switch backend {
	case ToolSessionStoreBackendLocal, ToolSessionStoreBackendRedis:
		return backend, nil
	default:
		return "", fmt.Errorf(
			"tool_sessions store: backend must be %q or %q",
			ToolSessionStoreBackendLocal, ToolSessionStoreBackendRedis,
		)
	}
}

func (s *ToolSessionStoreConfig) validateBackendRedisContract(backend string) error {
	if backend == ToolSessionStoreBackendLocal {
		if s.Redis != nil {
			return fmt.Errorf("tool_sessions store: redis config is not allowed when backend is %q", ToolSessionStoreBackendLocal)
		}
		return nil
	}
	// backend == ToolSessionStoreBackendRedis
	if s.Redis == nil || strings.TrimSpace(s.Redis.Address) == "" {
		return fmt.Errorf("tool_sessions store: redis.address is required when backend is %q", ToolSessionStoreBackendRedis)
	}
	return nil
}

func (s *ToolSessionStoreConfig) validateBounds() error {
	if err := validateToolSessionStoreIntBound(
		s.TTLSeconds, "ttl_seconds", ToolSessionStoreMinTTLSeconds, ToolSessionStoreMaxTTLSeconds,
	); err != nil {
		return err
	}
	if err := validateToolSessionStoreIntBound(
		s.MaxSessions, "max_sessions", ToolSessionStoreMinMaxSessions, ToolSessionStoreMaxMaxSessions,
	); err != nil {
		return err
	}
	maxSessions := s.EffectiveMaxSessions()
	if err := validateToolSessionStoreIntBound(
		s.MaxSessionsByIdentity, "max_sessions_per_identity", ToolSessionStoreMinMaxSessionsByIdentity, maxSessions,
	); err != nil {
		return err
	}
	if err := validateToolSessionStoreIntBound(
		s.MaxStateBytes, "max_state_bytes", ToolSessionStoreMinMaxStateBytes, ToolSessionStoreMaxMaxStateBytes,
	); err != nil {
		return err
	}
	if err := validateToolSessionStoreIntBound(
		s.TimeoutMs, "timeout_ms", ToolSessionStoreMinTimeoutMs, ToolSessionStoreMaxTimeoutMs,
	); err != nil {
		return err
	}
	return s.validateRedisDatabase()
}

// validateRedisDatabase rejects a negative redis.database. Unlike the
// pointer-typed fields above, Database is a plain int with no explicit-vs-
// unset ambiguity to guard against: 0 is both the zero value and Redis's
// own default database, so there is no distinct "unset" state to default
// away from — only a lower bound (no configured upper bound) to enforce.
func (s *ToolSessionStoreConfig) validateRedisDatabase() error {
	if s.Redis != nil && s.Redis.Database < 0 {
		return fmt.Errorf("tool_sessions store: redis.database must be greater than or equal to 0")
	}
	return nil
}

// validateToolSessionStoreIntBound rejects an explicitly configured value
// outside [min, max]. A nil pointer (field omitted) always passes — the
// Effective* helper applies the default, which is always in range.
func validateToolSessionStoreIntBound(value *int, field string, min, max int) error {
	if value == nil {
		return nil
	}
	if v := *value; v < min || v > max {
		return fmt.Errorf("tool_sessions store: %s must be between %d and %d", field, min, max)
	}
	return nil
}
