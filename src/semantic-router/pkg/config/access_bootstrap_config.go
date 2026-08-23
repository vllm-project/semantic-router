package config

import (
	"fmt"
	"strings"
	"time"
)

const (
	AccessStoreTypePostgres      = "postgres"
	AccessRuntimeStoreTypeRedis  = "redis"
	defaultAccessKeyPrefix       = "vllm-sr:access"
	defaultAccessMaxConnections  = 40
	defaultAccessUsageBacklog    = 1_000_000
	defaultTenantContextStartAge = "30s"
	defaultUsageCreateAhead      = 2
	defaultUsageMaintenance      = "5m"
)

// AccessStoreConfig configures the authoritative managed control-plane store.
// Credentials are references only; literal DSNs are deliberately not part of
// the public type.
type AccessStoreConfig struct {
	Type     string                    `yaml:"type,omitempty"`
	Postgres PostgresAccessStoreConfig `yaml:"postgres,omitempty"`
}

type PostgresAccessStoreConfig struct {
	DSNFile        string `yaml:"dsn_file,omitempty"`
	DSNEnv         string `yaml:"dsn_env,omitempty"`
	MaxConnections int    `yaml:"max_connections,omitempty"`
}

// AccessRuntimeStoreConfig configures shared applied state and global quota
// counters. "redis" denotes the Redis protocol and is valid for Valkey.
type AccessRuntimeStoreConfig struct {
	Type  string                        `yaml:"type,omitempty"`
	Redis RedisAccessRuntimeStoreConfig `yaml:"redis,omitempty"`
}

type RedisAccessRuntimeStoreConfig struct {
	URLFile   string `yaml:"url_file,omitempty"`
	URLEnv    string `yaml:"url_env,omitempty"`
	KeyPrefix string `yaml:"key_prefix,omitempty"`
}

// AccessServiceConfig contains only access-runtime bootstrap semantics. Users,
// keys, policies, bindings, and usage are managed resources and never belong in
// Router YAML.
type AccessServiceConfig struct {
	Enabled       bool                     `yaml:"enabled"`
	Credentials   AccessCredentialsConfig  `yaml:"credentials,omitempty"`
	TenantContext TenantContextConfig      `yaml:"tenant_context,omitempty"`
	Enforcement   AccessEnforcementConfig  `yaml:"enforcement,omitempty"`
	UsageStorage  AccessUsageStorageConfig `yaml:"usage_storage,omitempty"`
}

// AccessUsageStorageConfig controls the physical lifecycle of immutable usage
// facts. Monthly partitioning is fixed; an empty RawRetention keeps facts
// indefinitely and is the safe default.
type AccessUsageStorageConfig struct {
	CreateAheadMonths   int    `yaml:"create_ahead_months,omitempty"`
	MaintenanceInterval string `yaml:"maintenance_interval,omitempty"`
	RawRetention        string `yaml:"raw_retention,omitempty"`
}

type AccessCredentialsConfig struct {
	APIKeyHMACKeyringFile     string                       `yaml:"api_key_hmac_keyring_file,omitempty"`
	APIKeyHMACKeyringEnv      string                       `yaml:"api_key_hmac_keyring_env,omitempty"`
	DelegationHMACKeyringFile string                       `yaml:"delegation_hmac_keyring_file,omitempty"`
	DelegationHMACKeyringEnv  string                       `yaml:"delegation_hmac_keyring_env,omitempty"`
	Reveal                    AccessCredentialRevealConfig `yaml:"reveal,omitempty"`
}

type AccessCredentialRevealConfig struct {
	Enabled        bool   `yaml:"enabled"`
	KEKKeyringFile string `yaml:"kek_keyring_file,omitempty"`
	KEKKeyringEnv  string `yaml:"kek_keyring_env,omitempty"`
}

type TenantContextConfig struct {
	SigningKeyFile string `yaml:"signing_key_file,omitempty"`
	SigningKeyEnv  string `yaml:"signing_key_env,omitempty"`
	MaxStartAge    string `yaml:"max_start_age,omitempty"`
}

type AccessEnforcementConfig struct {
	FailureMode        string `yaml:"failure_mode,omitempty"`
	RequestAccounting  string `yaml:"request_accounting,omitempty"`
	TokenAccounting    string `yaml:"token_accounting,omitempty"`
	UnknownUsageAction string `yaml:"unknown_usage_action,omitempty"`
	SettleOn           string `yaml:"settle_on,omitempty"`
	DeduplicateBy      string `yaml:"deduplicate_by,omitempty"`
	MaxUsageBacklog    int64  `yaml:"max_usage_backlog,omitempty"`
}

func DefaultAccessServiceConfig() AccessServiceConfig {
	return AccessServiceConfig{
		Enabled: false,
		TenantContext: TenantContextConfig{
			MaxStartAge: defaultTenantContextStartAge,
		},
		Enforcement: AccessEnforcementConfig{
			FailureMode:        "deny",
			RequestAccounting:  "admission",
			TokenAccounting:    "response_actual",
			UnknownUsageAction: "freeze",
			SettleOn:           "stream_done",
			DeduplicateBy:      "admission_id",
			MaxUsageBacklog:    defaultAccessUsageBacklog,
		},
		UsageStorage: AccessUsageStorageConfig{
			CreateAheadMonths:   defaultUsageCreateAhead,
			MaintenanceInterval: defaultUsageMaintenance,
		},
	}
}

func applyAccessStoreDefaults(access *AccessStoreConfig, runtime *AccessRuntimeStoreConfig) {
	if access != nil {
		if access.Type == "" {
			access.Type = AccessStoreTypePostgres
		}
		if access.Postgres.MaxConnections == 0 {
			access.Postgres.MaxConnections = defaultAccessMaxConnections
		}
	}
	if runtime != nil {
		if runtime.Type == "" {
			runtime.Type = AccessRuntimeStoreTypeRedis
		}
		if runtime.Redis.KeyPrefix == "" {
			runtime.Redis.KeyPrefix = defaultAccessKeyPrefix
		}
	}
}

func applyAccessServiceDefaults(access *AccessServiceConfig) {
	if access == nil {
		return
	}
	defaults := DefaultAccessServiceConfig()
	if access.TenantContext.MaxStartAge == "" {
		access.TenantContext.MaxStartAge = defaults.TenantContext.MaxStartAge
	}
	if access.Enforcement.FailureMode == "" {
		access.Enforcement.FailureMode = defaults.Enforcement.FailureMode
	}
	if access.Enforcement.RequestAccounting == "" {
		access.Enforcement.RequestAccounting = defaults.Enforcement.RequestAccounting
	}
	if access.Enforcement.TokenAccounting == "" {
		access.Enforcement.TokenAccounting = defaults.Enforcement.TokenAccounting
	}
	if access.Enforcement.UnknownUsageAction == "" {
		access.Enforcement.UnknownUsageAction = defaults.Enforcement.UnknownUsageAction
	}
	if access.Enforcement.SettleOn == "" {
		access.Enforcement.SettleOn = defaults.Enforcement.SettleOn
	}
	if access.Enforcement.DeduplicateBy == "" {
		access.Enforcement.DeduplicateBy = defaults.Enforcement.DeduplicateBy
	}
	if access.Enforcement.MaxUsageBacklog == 0 {
		access.Enforcement.MaxUsageBacklog = defaults.Enforcement.MaxUsageBacklog
	}
	if access.UsageStorage.CreateAheadMonths == 0 {
		access.UsageStorage.CreateAheadMonths = defaults.UsageStorage.CreateAheadMonths
	}
	if access.UsageStorage.MaintenanceInterval == "" {
		access.UsageStorage.MaintenanceInterval = defaults.UsageStorage.MaintenanceInterval
	}
}

func validateAccessStore(store *AccessStoreConfig) error {
	if store == nil {
		return nil
	}
	if store.Type != AccessStoreTypePostgres {
		return fmt.Errorf("global.stores.access.type must be postgres")
	}
	if err := validateSecretSource("global.stores.access.postgres.dsn", store.Postgres.DSNFile, store.Postgres.DSNEnv, true); err != nil {
		return err
	}
	if store.Postgres.MaxConnections < 1 || store.Postgres.MaxConnections > 1000 {
		return fmt.Errorf("global.stores.access.postgres.max_connections must be between 1 and 1000")
	}
	return nil
}

func validateAccessRuntimeStore(store *AccessRuntimeStoreConfig) error {
	if store == nil {
		return nil
	}
	if store.Type != AccessRuntimeStoreTypeRedis {
		return fmt.Errorf("global.stores.access_runtime.type must be redis")
	}
	if err := validateSecretSource("global.stores.access_runtime.redis.url", store.Redis.URLFile, store.Redis.URLEnv, true); err != nil {
		return err
	}
	prefix := strings.TrimSpace(store.Redis.KeyPrefix)
	if prefix == "" || prefix != store.Redis.KeyPrefix || strings.ContainsAny(prefix, "\r\n\t ") {
		return fmt.Errorf("global.stores.access_runtime.redis.key_prefix must be a non-empty whitespace-free prefix")
	}
	return nil
}

func validateAccessService(access AccessServiceConfig) error {
	if err := validateSecretSource(
		"global.services.access.credentials.api_key_hmac_keyring",
		access.Credentials.APIKeyHMACKeyringFile,
		access.Credentials.APIKeyHMACKeyringEnv,
		access.Enabled,
	); err != nil {
		return err
	}
	if err := validateSecretSource(
		"global.services.access.credentials.delegation_hmac_keyring",
		access.Credentials.DelegationHMACKeyringFile,
		access.Credentials.DelegationHMACKeyringEnv,
		access.Enabled,
	); err != nil {
		return err
	}
	if err := validateSecretSource(
		"global.services.access.tenant_context.signing_key",
		access.TenantContext.SigningKeyFile,
		access.TenantContext.SigningKeyEnv,
		access.Enabled,
	); err != nil {
		return err
	}
	if err := validateSecretSource(
		"global.services.access.credentials.reveal.kek_keyring",
		access.Credentials.Reveal.KEKKeyringFile,
		access.Credentials.Reveal.KEKKeyringEnv,
		access.Credentials.Reveal.Enabled,
	); err != nil {
		return err
	}
	if !access.Credentials.Reveal.Enabled && (access.Credentials.Reveal.KEKKeyringFile != "" || access.Credentials.Reveal.KEKKeyringEnv != "") {
		return fmt.Errorf("global.services.access.credentials.reveal keyring requires enabled=true")
	}
	if !access.Enabled {
		if access.Credentials.APIKeyHMACKeyringFile != "" || access.Credentials.APIKeyHMACKeyringEnv != "" ||
			access.Credentials.DelegationHMACKeyringFile != "" || access.Credentials.DelegationHMACKeyringEnv != "" ||
			access.Credentials.Reveal.Enabled || access.TenantContext.SigningKeyFile != "" || access.TenantContext.SigningKeyEnv != "" {
			return fmt.Errorf("global.services.access credentials require enabled=true")
		}
		return nil
	}
	startAge, err := time.ParseDuration(access.TenantContext.MaxStartAge)
	if err != nil || startAge <= 0 || startAge > 5*time.Minute {
		return fmt.Errorf("global.services.access.tenant_context.max_start_age must be a positive duration no greater than 5m")
	}
	values := []struct {
		path     string
		value    string
		expected string
	}{
		{"failure_mode", access.Enforcement.FailureMode, "deny"},
		{"request_accounting", access.Enforcement.RequestAccounting, "admission"},
		{"token_accounting", access.Enforcement.TokenAccounting, "response_actual"},
		{"unknown_usage_action", access.Enforcement.UnknownUsageAction, "freeze"},
		{"settle_on", access.Enforcement.SettleOn, "stream_done"},
		{"deduplicate_by", access.Enforcement.DeduplicateBy, "admission_id"},
	}
	for _, item := range values {
		if item.value != item.expected {
			return fmt.Errorf("global.services.access.enforcement.%s must be %s", item.path, item.expected)
		}
	}
	if access.Enforcement.MaxUsageBacklog < 1 {
		return fmt.Errorf("global.services.access.enforcement.max_usage_backlog must be positive")
	}
	if access.UsageStorage.CreateAheadMonths < 1 || access.UsageStorage.CreateAheadMonths > 24 {
		return fmt.Errorf("global.services.access.usage_storage.create_ahead_months must be between 1 and 24")
	}
	maintenance, err := time.ParseDuration(access.UsageStorage.MaintenanceInterval)
	if err != nil || maintenance < time.Minute || maintenance > 24*time.Hour {
		return fmt.Errorf("global.services.access.usage_storage.maintenance_interval must be between 1m and 24h")
	}
	if access.UsageStorage.RawRetention != "" {
		retention, err := time.ParseDuration(access.UsageStorage.RawRetention)
		if err != nil || retention < time.Hour || retention > 10*365*24*time.Hour {
			return fmt.Errorf("global.services.access.usage_storage.raw_retention must be empty or between 1h and 87600h")
		}
	}
	return nil
}

func cloneAccessStoreConfig(source *AccessStoreConfig) *AccessStoreConfig {
	if source == nil {
		return nil
	}
	cloned := *source
	return &cloned
}

func cloneAccessRuntimeStoreConfig(source *AccessRuntimeStoreConfig) *AccessRuntimeStoreConfig {
	if source == nil {
		return nil
	}
	cloned := *source
	return &cloned
}
