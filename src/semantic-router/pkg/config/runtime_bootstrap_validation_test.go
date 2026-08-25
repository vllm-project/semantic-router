package config

import (
	"strings"
	"testing"
)

func TestRuntimeBootstrapDerivesCapabilitiesFromServicesAndStores(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*RouterConfig)
		wantErr string
	}{
		{name: "file routing", mutate: func(*RouterConfig) {}},
		{
			name:    "runtime store requires Management store",
			mutate:  func(cfg *RouterConfig) { cfg.AccessRuntimeStore = validAccessRuntimeStore() },
			wantErr: "global.stores.runtime.redis requires global.stores.management.postgres",
		},
		{
			name:    "native access requires both stores",
			mutate:  func(cfg *RouterConfig) { cfg.Access.Enabled = true },
			wantErr: "global.services.access.enabled requires",
		},
		{
			name:    "Management API requires durable authority",
			mutate:  func(cfg *RouterConfig) { cfg.ManagementAPI.Enabled = true },
			wantErr: "global.services.management_api.enabled requires",
		},
		{
			name:   "Management store without Management API",
			mutate: configureValidDurableRouting,
		},
		{
			name:   "durable access",
			mutate: configureValidDurableAccess,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := DefaultGlobalConfig()
			test.mutate(&cfg)
			err := cfg.ValidateRuntimeBootstrap()
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ValidateRuntimeBootstrap() error = %v, want fragment %q", err, test.wantErr)
			}
		})
	}
}

func TestRuntimeBootstrapSeparatesRoutingAndManagementAPISecurity(t *testing.T) {
	t.Run("durable routing requires only routing authorities", func(t *testing.T) {
		cfg := DefaultGlobalConfig()
		configureValidDurableRouting(&cfg)
		cfg.ManagementAPI = DefaultManagementAPIConfig()
		if err := cfg.ValidateRuntimeBootstrap(); err != nil {
			t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
		}
	})

	t.Run("routing HMAC is required by durable routing", func(t *testing.T) {
		cfg := DefaultGlobalConfig()
		configureValidDurableRouting(&cfg)
		cfg.RoutingSecurity = RoutingSecurityConfig{}
		err := cfg.ValidateRuntimeBootstrap()
		if err == nil || !strings.Contains(err.Error(), "routing_security.hmac_keyring requires exactly one") {
			t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
		}
	})

	t.Run("file routing rejects a durable routing authority", func(t *testing.T) {
		cfg := DefaultGlobalConfig()
		cfg.RoutingSecurity.HMACKeyringEnv = "VLLM_SR_ROUTING_HMAC_KEYRING"
		err := cfg.ValidateRuntimeBootstrap()
		if err == nil || !strings.Contains(err.Error(), "requires global.stores.management.postgres") {
			t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
		}
	})

	t.Run("native access does not imply Management API", func(t *testing.T) {
		cfg := DefaultGlobalConfig()
		configureValidDurableAccess(&cfg)
		cfg.ManagementAPI = DefaultManagementAPIConfig()
		if err := cfg.ValidateRuntimeBootstrap(); err != nil {
			t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
		}
	})

	t.Run("enabled Management API requires API-only TLS and keyrings", func(t *testing.T) {
		cfg := DefaultGlobalConfig()
		configureValidDurableRouting(&cfg)
		cfg.ManagementAPI.Enabled = true
		cfg.ManagementAPI.Auth.Mode = ManagementAuthModeRouter
		cfg.ManagementAPI.Auth.Roles = nil
		err := cfg.ValidateRuntimeBootstrap()
		if err == nil || !strings.Contains(err.Error(), "management_api.tls.certificate requires exactly one") {
			t.Fatalf("ValidateRuntimeBootstrap() error = %v", err)
		}
	})
}

func TestParseDurableBootstrapUsesTypedStoreChildren(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(validDurableBootstrapYAML))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.AccessStore == nil || cfg.AccessStore.Type != AccessStoreTypePostgres ||
		cfg.AccessStore.Postgres.MaxConnections != defaultAccessMaxConnections {
		t.Fatalf("Management store = %#v", cfg.AccessStore)
	}
	if cfg.AccessRuntimeStore == nil || cfg.AccessRuntimeStore.Type != AccessRuntimeStoreTypeRedis ||
		cfg.AccessRuntimeStore.Redis.KeyPrefix != defaultAccessKeyPrefix {
		t.Fatalf("runtime store = %#v", cfg.AccessRuntimeStore)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Global == nil || exported.Global.Stores.Management == nil ||
		exported.Global.Stores.Management.Postgres == nil || exported.Global.Stores.Runtime == nil ||
		exported.Global.Stores.Runtime.Redis == nil {
		t.Fatalf("canonical stores = %#v", exported.Global)
	}
}

func TestParseRejectsRemovedAuthoritySelectorsAndStoreAliases(t *testing.T) {
	for _, fragment := range []string{
		"  control_plane:\n    mode: managed\n",
		"  stores:\n    access:\n      postgres: {dsn_env: DATABASE_URL}\n",
		"  stores:\n    access_runtime:\n      redis: {url_env: REDIS_URL}\n",
	} {
		manifest := "version: v0.3\nglobal:\n" + fragment
		if _, err := ParseYAMLBytes([]byte(manifest)); err == nil {
			t.Fatalf("ParseYAMLBytes() accepted removed public authority:\n%s", fragment)
		}
	}
}

func validAccessStore() *AccessStoreConfig {
	return &AccessStoreConfig{
		Type: AccessStoreTypePostgres,
		Postgres: PostgresAccessStoreConfig{
			DSNFile: "/run/secrets/access-dsn", MaxConnections: defaultAccessMaxConnections,
		},
	}
}

func validAccessRuntimeStore() *AccessRuntimeStoreConfig {
	return &AccessRuntimeStoreConfig{
		Type: AccessRuntimeStoreTypeRedis,
		Redis: RedisAccessRuntimeStoreConfig{
			URLFile: "/run/secrets/access-redis-url", KeyPrefix: defaultAccessKeyPrefix,
		},
	}
}

func configureValidDurableAccess(cfg *RouterConfig) {
	configureValidDurableRouting(cfg)
	cfg.AccessRuntimeStore = validAccessRuntimeStore()
	cfg.Access.Enabled = true
	cfg.Agent.PublicInferenceEndpoint = "http://public-inference.internal/v1/chat/completions"
	cfg.ManagementAPI.Enabled = true
	cfg.ManagementAPI.Auth.Mode = ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Tokens = nil
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.Access.Credentials.APIKeyHMACKeyringFile = "/run/secrets/api-key-peppers"
	cfg.Access.Credentials.DelegationHMACKeyringFile = "/run/secrets/delegation-peppers"
	cfg.Access.TenantContext.SigningKeyFile = "/run/secrets/tenant-context-keys"

	cfg.ManagementAPI.TLS.CertificateFile = "/run/secrets/management-cert"
	cfg.ManagementAPI.TLS.PrivateKeyFile = "/run/secrets/management-key"
	cfg.ManagementAPI.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing-keys"
	cfg.ManagementAPI.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/management-peppers"
	cfg.ManagementAPI.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-peppers"
	cfg.RoutingSecurity.HMACKeyringFile = "/run/secrets/routing-hmac-keys"
	cfg.ManagementAPI.Auth.ResponseKEKKeyringFile = "/run/secrets/response-keks"
}

func configureValidDurableRouting(cfg *RouterConfig) {
	cfg.AccessStore = validAccessStore()
	cfg.BackendCredentials.ProviderKEKKeyringFile = "/run/secrets/provider-keks"
	cfg.RoutingSecurity.HMACKeyringFile = "/run/secrets/routing-hmac-keys"
	cfg.BackendEgress.PolicyFile = "/etc/vllm-sr/backend-egress-policy.yaml"
}

const validDurableBootstrapYAML = `
version: v0.3
global:
  stores:
    management:
      postgres:
        dsn_env: VLLM_SR_ACCESS_DATABASE_URL
    runtime:
      redis:
        url_file: /run/secrets/access-redis-url
  services:
    agent:
      public_inference_endpoint: http://public-inference.internal/v1/chat/completions
    access:
      enabled: true
      credentials:
        api_key_hmac_keyring_file: /run/secrets/api-key-peppers
        delegation_hmac_keyring_file: /run/secrets/delegation-peppers
      tenant_context:
        signing_key_file: /run/secrets/tenant-context-keys
    backend_credentials:
      provider_kek_keyring_file: /run/secrets/provider-keks
    backend_egress:
      policy_file: /etc/vllm-sr/backend-egress-policy.yaml
    routing_security:
      hmac_keyring_file: /run/secrets/routing-hmac-keys
    management_api:
      enabled: true
      tls:
        certificate_file: /run/secrets/management-cert
        private_key_file: /run/secrets/management-key
      auth:
        mode: router
        token_signing_keyring_file: /run/secrets/management-signing-keys
        service_account_hmac_keyring_file: /run/secrets/management-peppers
        invitation_hmac_keyring_file: /run/secrets/invitation-peppers
        response_kek_keyring_file: /run/secrets/response-keks
        bootstrap:
          token_file: /run/secrets/bootstrap-token
          disable_after_first_cluster_admin: true
`
