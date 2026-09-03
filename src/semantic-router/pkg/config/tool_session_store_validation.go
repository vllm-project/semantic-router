package config

import "fmt"

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
	if s.Redis == nil || s.Redis.Address == "" {
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
	return validateToolSessionStoreIntBound(
		s.TimeoutMs, "timeout_ms", ToolSessionStoreMinTimeoutMs, ToolSessionStoreMaxTimeoutMs,
	)
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
