package config

import (
	"reflect"
	"strings"
	"testing"
)

const sharedGuardConfig = `  stores:
    guard_verdicts:
      backend: redis
      hmac_key_env: GUARD_HMAC
      timeout_ms: 25
      redis:
        address: redis:6379
        username: router
        password_env: REDIS_PASSWORD
        database: 3
        tls: true
`

func guardVerdictDocument(block string) []byte {
	base := string(routerHeadersConfig("    list_backend_models: false\n"))
	return []byte(base + block)
}

func TestGuardVerdictConfigCanonicalRoundTrip(t *testing.T) {
	cfg, err := ParseYAMLBytes(guardVerdictDocument(sharedGuardConfig))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.GuardVerdicts.Shared() || cfg.GuardVerdicts.EffectiveTimeoutMs() != 25 {
		t.Fatalf("shared store was not configured: %+v", cfg.GuardVerdicts)
	}
	exported := CanonicalGlobalFromRouterConfig(cfg).Stores.GuardVerdicts
	if !reflect.DeepEqual(exported, cfg.GuardVerdicts) {
		t.Fatalf("exported store changed: %+v", exported)
	}
	exported.Redis.Address = "other:6379"
	*exported.TimeoutMs = 100
	if cfg.GuardVerdicts.Redis.Address != "redis:6379" || cfg.GuardVerdicts.EffectiveTimeoutMs() != 25 {
		t.Fatal("exported store aliases the running config")
	}
	local, err := ParseYAMLBytes(guardVerdictDocument(""))
	if err != nil || local.GuardVerdicts.Shared() {
		t.Fatalf("omitting the store must keep process-local behavior: %v", err)
	}
}

func TestGuardVerdictConfigRejectsInvalidSharedStores(t *testing.T) {
	for name, replacement := range map[string][2]string{
		"backend":          {"backend: redis", "backend: unknown"},
		"local with redis": {"backend: redis", "backend: local"},
		"signing key":      {"hmac_key_env: GUARD_HMAC", "hmac_key_env: ''"},
		"authentication":   {"password_env: REDIS_PASSWORD", "password_env: ''"},
		"shared secret":    {"password_env: REDIS_PASSWORD", "password_env: GUARD_HMAC"},
		"address":          {"address: redis:6379", "address: ''"},
		"database":         {"database: 3", "database: -1"},
		"zero timeout":     {"timeout_ms: 25", "timeout_ms: 0"},
		"large timeout":    {"timeout_ms: 25", "timeout_ms: 1001"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes(guardVerdictDocument(strings.ReplaceAll(sharedGuardConfig, replacement[0], replacement[1])))
			if err == nil || !strings.Contains(err.Error(), "guard_verdicts") {
				t.Fatalf("expected guard_verdicts validation error, got %v", err)
			}
		})
	}
}

func TestGuardVerdictSecretNamesRejectInterpolationBeforeExpansion(t *testing.T) {
	secret := strings.Repeat("A", 64)
	t.Setenv("GUARD_HMAC", secret)
	t.Setenv("REDIS_PASSWORD", secret)
	for _, field := range []string{"GUARD_HMAC", "REDIS_PASSWORD"} {
		for _, name := range []string{" " + field, field + " ", "${" + field + "}", "$" + field, "key-name", "lowercase", "1KEY"} {
			document := guardVerdictDocument(strings.ReplaceAll(sharedGuardConfig, ": "+field+"\n", ": '"+name+"'\n"))
			_, err := ParseYAMLBytes(document)
			if err == nil || !strings.Contains(err.Error(), "guard_verdicts") || strings.Contains(err.Error(), secret) {
				t.Fatalf("invalid name was expanded, accepted, or exposed: %v", err)
			}
		}
	}
}

func TestGuardVerdictLocalStoreAllowsEmptySecretName(t *testing.T) {
	for _, backend := range []string{"local", ""} {
		document := guardVerdictDocument("  stores:\n    guard_verdicts:\n      backend: '" + backend + "'\n      hmac_key_env: ''\n")
		cfg, err := ParseYAMLBytes(document)
		if err != nil {
			t.Fatal(err)
		}
		if cfg.GuardVerdicts.Shared() || cfg.GuardVerdicts.HMACKeyEnv != "" {
			t.Fatalf("empty local secret name changed the backend: %+v", cfg.GuardVerdicts)
		}
	}
}
