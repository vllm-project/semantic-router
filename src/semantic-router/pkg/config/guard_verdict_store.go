package config

import (
	"fmt"
	"regexp"
	"strings"
)

var guardVerdictEnvName = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)

// GuardVerdictStoreConfig optionally shares complete request-stage guard
// window answers. The process-local cache is always consulted first.
type GuardVerdictStoreConfig struct {
	Backend string `yaml:"backend,omitempty" json:"backend,omitempty" jsonschema:"enum=local,enum=redis"`
	// HMACKeyEnv names an environment variable containing a 32-byte secret
	// encoded as 64 hex digits. Only the Routers may hold this secret.
	HMACKeyEnv string                   `yaml:"hmac_key_env,omitempty" json:"hmac_key_env,omitempty" jsonschema:"pattern=^([A-Z_][A-Z0-9_]*)?$"`
	TimeoutMs  *int                     `yaml:"timeout_ms,omitempty" json:"timeout_ms,omitempty" jsonschema:"minimum=1,maximum=1000"`
	Redis      *GuardVerdictRedisConfig `yaml:"redis,omitempty" json:"redis,omitempty"`
}

// GuardVerdictRedisConfig names an authenticated, Router-private Redis store.
// Secrets are resolved at runtime, never copied into exported configuration.
type GuardVerdictRedisConfig struct {
	Address     string `yaml:"address" json:"address"`
	Username    string `yaml:"username,omitempty" json:"username,omitempty"`
	PasswordEnv string `yaml:"password_env" json:"password_env" jsonschema:"pattern=^[A-Z_][A-Z0-9_]*$"`
	Database    int    `yaml:"database,omitempty" json:"database,omitempty" jsonschema:"minimum=0"`
	TLS         bool   `yaml:"tls,omitempty" json:"tls,omitempty"`
}

func (s *GuardVerdictStoreConfig) Shared() bool {
	return s != nil && s.Backend == "redis"
}

func (s *GuardVerdictStoreConfig) EffectiveTimeoutMs() int {
	if s == nil || s.TimeoutMs == nil {
		return 50
	}
	return *s.TimeoutMs
}

func (s *GuardVerdictStoreConfig) Validate() error {
	if s == nil {
		return nil
	}
	const at = "global.stores.guard_verdicts"
	if s.TimeoutMs != nil && (*s.TimeoutMs < 1 || *s.TimeoutMs > 1000) {
		return fmt.Errorf("%s.timeout_ms must be between 1 and 1000", at)
	}
	switch s.Backend {
	case "", "local":
		if s.Redis != nil || s.HMACKeyEnv != "" {
			return fmt.Errorf("%s: redis and hmac_key_env require backend: redis", at)
		}
	case "redis":
		if strings.TrimSpace(s.HMACKeyEnv) == "" {
			return fmt.Errorf("%s.hmac_key_env is required for redis", at)
		}
		if s.Redis == nil || strings.TrimSpace(s.Redis.Address) == "" || strings.TrimSpace(s.Redis.PasswordEnv) == "" {
			return fmt.Errorf("%s.redis.address and password_env are required", at)
		}
		if !guardVerdictEnvName.MatchString(s.HMACKeyEnv) || !guardVerdictEnvName.MatchString(s.Redis.PasswordEnv) {
			return fmt.Errorf("%s.hmac_key_env and redis.password_env must be environment variable names matching [A-Z_][A-Z0-9_]*, without interpolation or whitespace", at)
		}
		if s.Redis.Database < 0 {
			return fmt.Errorf("%s.redis.database must not be negative", at)
		}
		if s.HMACKeyEnv == s.Redis.PasswordEnv {
			return fmt.Errorf("%s.hmac_key_env must be independent of redis.password_env", at)
		}
	default:
		return fmt.Errorf("%s.backend must be local or redis", at)
	}
	return nil
}

func validateGuardVerdictStore(cfg *RouterConfig) error {
	return cfg.GuardVerdicts.Validate()
}

// Check names before interpolation can replace a mistaken ${SECRET} with the
// secret itself. Neither validation errors nor exported config may expose it.
func checkRawGuardVerdictEnvNames(raw map[string]interface{}) error {
	stores := nestedStringMap(nestedStringMap(raw["global"])["stores"])
	guard := nestedStringMap(stores["guard_verdicts"])
	for field, value := range map[string]interface{}{
		"hmac_key_env":       guard["hmac_key_env"],
		"redis.password_env": nestedStringMap(guard["redis"])["password_env"],
	} {
		if value == nil {
			continue
		}
		name, ok := value.(string)
		if !ok || (name != "" && !guardVerdictEnvName.MatchString(name)) {
			return fmt.Errorf("global.stores.guard_verdicts.%s must be an environment variable name without interpolation or whitespace", field)
		}
	}
	return nil
}

func cloneGuardVerdictStoreConfig(s *GuardVerdictStoreConfig) *GuardVerdictStoreConfig {
	if s == nil {
		return nil
	}
	copy := *s
	if s.TimeoutMs != nil {
		timeout := *s.TimeoutMs
		copy.TimeoutMs = &timeout
	}
	if s.Redis != nil {
		redis := *s.Redis
		copy.Redis = &redis
	}
	return &copy
}
