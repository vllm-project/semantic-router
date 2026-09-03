package config

import (
	"strings"
	"testing"
)

func TestToolSessionStoreEffectiveDefaults_NilAndOmitted(t *testing.T) {
	var nilCfg *ToolSessionStoreConfig
	if got := nilCfg.EffectiveBackend(); got != ToolSessionStoreBackendLocal {
		t.Fatalf("nil backend = %q", got)
	}
	if got := nilCfg.EffectiveTTLSeconds(); got != ToolSessionStoreDefaultTTLSeconds {
		t.Fatalf("nil ttl_seconds = %d", got)
	}

	empty := &ToolSessionStoreConfig{}
	if got := empty.EffectiveBackend(); got != ToolSessionStoreBackendLocal {
		t.Fatalf("empty backend = %q", got)
	}
	if got := empty.EffectiveTTLSeconds(); got != ToolSessionStoreDefaultTTLSeconds {
		t.Fatalf("effective ttl_seconds = %d", got)
	}
	if got := empty.EffectiveMaxSessions(); got != ToolSessionStoreDefaultMaxSessions {
		t.Fatalf("effective max_sessions = %d", got)
	}
	if got := empty.EffectiveMaxSessionsByIdentity(); got != ToolSessionStoreDefaultMaxSessionsByIdentity {
		t.Fatalf("effective max_sessions_per_identity = %d", got)
	}
	if got := empty.EffectiveMaxStateBytes(); got != ToolSessionStoreDefaultMaxStateBytes {
		t.Fatalf("effective max_state_bytes = %d", got)
	}
	if got := empty.EffectiveTimeoutMs(); got != ToolSessionStoreDefaultTimeoutMs {
		t.Fatalf("effective timeout_ms = %d", got)
	}
	if got := empty.EffectiveRedisKeyPrefix(); got != ToolSessionStoreDefaultRedisKeyPrefix {
		t.Fatalf("effective redis key_prefix = %q", got)
	}
	if err := empty.Validate(); err != nil {
		t.Fatalf("omitted config should validate cleanly: %v", err)
	}
	if err := (*ToolSessionStoreConfig)(nil).Validate(); err != nil {
		t.Fatalf("nil receiver should validate cleanly: %v", err)
	}
}

func TestToolSessionStoreEffective_ExplicitValuesOverrideDefaults(t *testing.T) {
	cfg := &ToolSessionStoreConfig{
		Backend:               ToolSessionStoreBackendRedis,
		TTLSeconds:            intPtr(60),
		MaxSessions:           intPtr(500),
		MaxSessionsByIdentity: intPtr(5),
		MaxStateBytes:         intPtr(2048),
		TimeoutMs:             intPtr(200),
		Redis:                 &ToolSessionRedisConfig{Address: "redis:6379", KeyPrefix: "custom:"},
	}
	if got := cfg.EffectiveBackend(); got != ToolSessionStoreBackendRedis {
		t.Fatalf("backend = %q", got)
	}
	if got := cfg.EffectiveTTLSeconds(); got != 60 {
		t.Fatalf("ttl_seconds = %d", got)
	}
	if got := cfg.EffectiveMaxSessions(); got != 500 {
		t.Fatalf("max_sessions = %d", got)
	}
	if got := cfg.EffectiveMaxSessionsByIdentity(); got != 5 {
		t.Fatalf("max_sessions_per_identity = %d", got)
	}
	if got := cfg.EffectiveMaxStateBytes(); got != 2048 {
		t.Fatalf("max_state_bytes = %d", got)
	}
	if got := cfg.EffectiveTimeoutMs(); got != 200 {
		t.Fatalf("timeout_ms = %d", got)
	}
	if got := cfg.EffectiveRedisKeyPrefix(); got != "custom:" {
		t.Fatalf("redis key_prefix = %q", got)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSessionStoreValidate_InvalidBackend_Err(t *testing.T) {
	cfg := &ToolSessionStoreConfig{Backend: "postgres"}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error for unknown backend")
	}
}

func TestToolSessionStoreValidate_LocalBackendRejectsRedisBlock_Err(t *testing.T) {
	cfg := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendLocal,
		Redis:   &ToolSessionRedisConfig{Address: "redis:6379"},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error: redis config present under backend: local")
	}
}

func TestToolSessionStoreValidate_LocalBackendDefault_RedisOmitted_OK(t *testing.T) {
	cfg := &ToolSessionStoreConfig{}
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSessionStoreValidate_RedisBackendRequiresAddress_Err(t *testing.T) {
	for name, cfg := range map[string]*ToolSessionStoreConfig{
		"redis block absent":       {Backend: ToolSessionStoreBackendRedis},
		"redis block empty":        {Backend: ToolSessionStoreBackendRedis, Redis: &ToolSessionRedisConfig{}},
		"address explicitly empty": {Backend: ToolSessionStoreBackendRedis, Redis: &ToolSessionRedisConfig{Address: ""}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := cfg.Validate(); err == nil {
				t.Fatal("expected error: redis backend requires redis.address")
			}
		})
	}
}

func TestToolSessionStoreValidate_RedisBackendWithAddress_OK(t *testing.T) {
	cfg := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendRedis,
		Redis:   &ToolSessionRedisConfig{Address: "redis:6379"},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSessionStoreValidate_NumericBoundsRejectOutOfRange_Err(t *testing.T) {
	cases := map[string]*ToolSessionStoreConfig{
		"ttl_seconds explicit zero":               {TTLSeconds: intPtr(0)},
		"ttl_seconds negative":                    {TTLSeconds: intPtr(-1)},
		"ttl_seconds above ceiling":               {TTLSeconds: intPtr(ToolSessionStoreMaxTTLSeconds + 1)},
		"max_sessions explicit zero":              {MaxSessions: intPtr(0)},
		"max_sessions above ceiling":              {MaxSessions: intPtr(ToolSessionStoreMaxMaxSessions + 1)},
		"max_sessions_per_identity explicit zero": {MaxSessionsByIdentity: intPtr(0)},
		"max_state_bytes below floor":             {MaxStateBytes: intPtr(ToolSessionStoreMinMaxStateBytes - 1)},
		"max_state_bytes above ceiling":           {MaxStateBytes: intPtr(ToolSessionStoreMaxMaxStateBytes + 1)},
		"timeout_ms explicit zero":                {TimeoutMs: intPtr(0)},
		"timeout_ms above ceiling":                {TimeoutMs: intPtr(ToolSessionStoreMaxTimeoutMs + 1)},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if err := cfg.Validate(); err == nil {
				t.Fatalf("expected error for %s", name)
			}
		})
	}
}

func TestToolSessionStoreValidate_RedisAddressWhitespaceOnly_Err(t *testing.T) {
	cfg := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendRedis,
		Redis:   &ToolSessionRedisConfig{Address: "   "},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error: whitespace-only redis.address must be rejected, not treated as present")
	}
}

func TestToolSessionStoreValidate_RedisDatabaseNegative_Err(t *testing.T) {
	cfg := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendRedis,
		Redis:   &ToolSessionRedisConfig{Address: "redis:6379", Database: -1},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error: negative redis.database")
	}
	const want = "tool_sessions store: redis.database must be greater than or equal to 0"
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err.Error(), want)
	}
}

func TestToolSessionStoreValidate_RedisDatabaseZeroAndPositive_OK(t *testing.T) {
	for _, db := range []int{0, 1, 15} {
		cfg := &ToolSessionStoreConfig{
			Backend: ToolSessionStoreBackendRedis,
			Redis:   &ToolSessionRedisConfig{Address: "redis:6379", Database: db},
		}
		if err := cfg.Validate(); err != nil {
			t.Fatalf("database=%d: %v", db, err)
		}
	}
}

// TestToolSessionStoreValidate_MaxSessionsByIdentityBoundedByMaxSessions
// covers the cross-field bound: max_sessions_per_identity must not exceed
// the *effective* max_sessions (explicit or defaulted), the same pattern
// StickyToolSelectionConfig's MaxNewToolsPerTurn/MaxTools already
// established.
func TestToolSessionStoreValidate_MaxSessionsByIdentityBoundedByMaxSessions(t *testing.T) {
	t.Run("exceeds explicit max_sessions", func(t *testing.T) {
		cfg := &ToolSessionStoreConfig{
			MaxSessions:           intPtr(10),
			MaxSessionsByIdentity: intPtr(11),
		}
		if err := cfg.Validate(); err == nil {
			t.Fatal("expected error: max_sessions_per_identity exceeds max_sessions")
		}
	})
	t.Run("exceeds default max_sessions", func(t *testing.T) {
		cfg := &ToolSessionStoreConfig{
			MaxSessionsByIdentity: intPtr(ToolSessionStoreDefaultMaxSessions + 1),
		}
		if err := cfg.Validate(); err == nil {
			t.Fatal("expected error: max_sessions_per_identity exceeds the default max_sessions")
		}
	})
	t.Run("equal to max_sessions is allowed", func(t *testing.T) {
		cfg := &ToolSessionStoreConfig{
			MaxSessions:           intPtr(10),
			MaxSessionsByIdentity: intPtr(10),
		}
		if err := cfg.Validate(); err != nil {
			t.Fatal(err)
		}
	})
}

// TestToolSessionStoreCanonicalRoundTrip covers the canonical
// export/import boundary: applyCanonicalGlobal followed by the export path
// (exercised indirectly via cloneToolSessionStoreConfig, since
// exportGlobal itself needs a full RouterConfig) must not alias pointers
// between the original and the round-tripped copy.
func TestToolSessionStoreCanonicalRoundTrip(t *testing.T) {
	original := &ToolSessionStoreConfig{
		Backend:               ToolSessionStoreBackendRedis,
		TTLSeconds:            intPtr(900),
		MaxSessions:           intPtr(200),
		MaxSessionsByIdentity: intPtr(20),
		MaxStateBytes:         intPtr(4096),
		TimeoutMs:             intPtr(100),
		Redis:                 &ToolSessionRedisConfig{Address: "redis:6379", KeyPrefix: "vsr:test:"},
	}

	cloned := cloneToolSessionStoreConfig(original)
	if cloned == original {
		t.Fatal("clone must not return the same pointer")
	}
	if cloned.TTLSeconds == original.TTLSeconds {
		t.Fatal("clone must not alias the TTLSeconds pointer")
	}
	if cloned.Redis == original.Redis {
		t.Fatal("clone must not alias the Redis pointer")
	}
	if *cloned.TTLSeconds != *original.TTLSeconds || cloned.Redis.Address != original.Redis.Address {
		t.Fatalf("clone diverged in value: %+v vs %+v", cloned, original)
	}

	// Mutating the clone must not affect the original — the actual bug a
	// shallow *cfg copy (like cloneVectorStoreConfig's, safe there only
	// because that struct has no pointer fields) would introduce here.
	*cloned.TTLSeconds = 1
	cloned.Redis.Address = "mutated"
	if *original.TTLSeconds != 900 || original.Redis.Address != "redis:6379" {
		t.Fatal("mutating the clone leaked into the original: clone is not independent")
	}

	if got := cloneToolSessionStoreConfig(nil); got != nil {
		t.Fatalf("cloning nil must return nil, got %+v", got)
	}
}

// TestToolSessionStoreAdmission_InvalidConfigFailsConfigContracts covers
// Finding 1: an invalid global.stores.tool_sessions must fail admission,
// not only a direct call to ToolSessionStoreConfig.Validate(). Exercises
// validateConfigContracts(cfg) directly (the same package-level unit-test
// pattern config_test.go already uses for other contract validators)
// rather than round-tripping through a full ParseYAMLBytes document: traced
// the call chain (ParseYAMLBytes -> finalizeParsedConfig ->
// validateConfigStructure -> validateConfigContracts ->
// runConfigContractValidators(cfg, globalConfigContractValidators)) to
// confirm this is the exact function ParseYAMLBytes itself calls, so
// exercising it directly is equivalent without needing a large, fragile
// fully-parseable YAML fixture just to reach one nested field.
func TestToolSessionStoreAdmission_InvalidConfigFailsConfigContracts(t *testing.T) {
	t.Run("invalid tool_sessions fails admission", func(t *testing.T) {
		cfg := &RouterConfig{
			ToolSessions: &ToolSessionStoreConfig{Backend: "postgres"},
		}
		err := validateConfigContracts(cfg)
		if err == nil {
			t.Fatal("expected admission to fail for an invalid tool_sessions backend")
		}
		if !strings.Contains(err.Error(), "tool_sessions store") {
			t.Fatalf("error = %q, want it to name tool_sessions store", err.Error())
		}
	})

	t.Run("valid tool_sessions passes admission", func(t *testing.T) {
		cfg := &RouterConfig{
			ToolSessions: &ToolSessionStoreConfig{
				Backend: ToolSessionStoreBackendRedis,
				Redis:   &ToolSessionRedisConfig{Address: "redis:6379"},
			},
		}
		if err := validateConfigContracts(cfg); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("omitted tool_sessions passes admission", func(t *testing.T) {
		if err := validateConfigContracts(&RouterConfig{}); err != nil {
			t.Fatal(err)
		}
	})
}
