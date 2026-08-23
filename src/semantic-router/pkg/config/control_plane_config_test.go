package config

import (
	"strings"
	"testing"
)

func TestControlPlaneModeMatrix(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*RouterConfig)
		wantErr string
	}{
		{
			name:   "standalone defaults are valid",
			mutate: func(*RouterConfig) {},
		},
		{
			name: "standalone rejects authoritative stores",
			mutate: func(cfg *RouterConfig) {
				cfg.AccessStore = validAccessStore()
			},
			wantErr: "standalone rejects",
		},
		{
			name: "standalone rejects access enforcement",
			mutate: func(cfg *RouterConfig) {
				cfg.Access.Enabled = true
			},
			wantErr: "standalone requires global.services.access.enabled=false",
		},
		{
			name: "standalone rejects disabled access credentials",
			mutate: func(cfg *RouterConfig) {
				cfg.Access.Credentials.APIKeyHMACKeyringFile = "/run/secrets/unused-pepper"
			},
			wantErr: "access credentials require enabled=true",
		},
		{
			name: "managed access requires postgres",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.AccessStore = nil
			},
			wantErr: "requires global.stores.access.type=postgres",
		},
		{
			name: "managed access requires redis",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.AccessRuntimeStore = nil
			},
			wantErr: "requires global.stores.access_runtime.type=redis",
		},
		{
			name: "managed routing still requires stores and management security",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.Access = DefaultAccessServiceConfig()
				cfg.ControlPlane.PublicNamespaceID = "11111111-1111-4111-8111-111111111111"
			},
		},
		{
			name: "managed routing rejects missing management tls",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.Access = DefaultAccessServiceConfig()
				cfg.ControlPlane.PublicNamespaceID = "11111111-1111-4111-8111-111111111111"
				cfg.ManagementAPI.TLS = ManagementAPITLSConfig{}
			},
			wantErr: "management_api.tls.certificate requires exactly one",
		},
		{
			name: "managed access accepts complete bootstrap",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
			},
		},
		{
			name: "usage partition lookahead is bounded",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.Access.UsageStorage.CreateAheadMonths = 25
			},
			wantErr: "usage_storage.create_ahead_months",
		},
		{
			name: "usage maintenance interval is operationally bounded",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.Access.UsageStorage.MaintenanceInterval = "30s"
			},
			wantErr: "usage_storage.maintenance_interval",
		},
		{
			name: "raw usage retention requires an explicit safe duration",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.Access.UsageStorage.RawRetention = "30m"
			},
			wantErr: "usage_storage.raw_retention",
		},
		{
			name: "managed routing requires provider credential encryption",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.BackendCredentials.ProviderKEKKeyringFile = ""
			},
			wantErr: "backend_credentials.provider_kek_keyring requires exactly one",
		},
		{
			name: "managed routing requires an egress policy",
			mutate: func(cfg *RouterConfig) {
				configureValidManagedAccess(cfg)
				cfg.BackendEgress.PolicyFile = ""
			},
			wantErr: "requires global.services.backend_egress.policy_file",
		},
		{
			name: "standalone rejects catalog rollout state",
			mutate: func(cfg *RouterConfig) {
				cfg.ControlPlane.ProviderCatalog.ReplicaIDEnv = "VLLM_SR_REPLICA_ID"
			},
			wantErr: "provider_catalog is managed-only",
		},
		{
			name: "standalone rejects control-plane HMAC authority",
			mutate: func(cfg *RouterConfig) {
				cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac-keys"
			},
			wantErr: "control_plane_hmac_keyring is managed-only",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := DefaultGlobalConfig()
			test.mutate(&cfg)
			err := cfg.ValidateControlPlaneBootstrap()
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateControlPlaneBootstrap() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ValidateControlPlaneBootstrap() error = %v, want fragment %q", err, test.wantErr)
			}
		})
	}
}

func TestControlPlaneSecretReferenceConstraints(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*RouterConfig)
		wantErr string
	}{
		{
			name: "postgres file and env are exclusive",
			mutate: func(cfg *RouterConfig) {
				cfg.AccessStore.Postgres.DSNEnv = "VLLM_SR_ACCESS_DSN"
			},
			wantErr: "mutually exclusive",
		},
		{
			name: "access keyring file and env are exclusive",
			mutate: func(cfg *RouterConfig) {
				cfg.Access.Credentials.APIKeyHMACKeyringEnv = "VLLM_SR_API_KEY_PEPPERS"
			},
			wantErr: "api_key_hmac_keyring_file",
		},
		{
			name: "control-plane keyring file and env are exclusive",
			mutate: func(cfg *RouterConfig) {
				cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringEnv = "VLLM_SR_CONTROL_PLANE_HMAC_KEYS"
			},
			wantErr: "control_plane_hmac_keyring_file",
		},
		{
			name: "reveal requires a kek keyring",
			mutate: func(cfg *RouterConfig) {
				cfg.Access.Credentials.Reveal.Enabled = true
				cfg.Access.Credentials.Reveal.KEKKeyringFile = ""
			},
			wantErr: "reveal.kek_keyring requires exactly one",
		},
		{
			name: "recovery requires a token",
			mutate: func(cfg *RouterConfig) {
				cfg.ManagementAPI.Auth.Recovery.Enabled = true
				cfg.ManagementAPI.Auth.Recovery.TokenFile = ""
			},
			wantErr: "recovery.token requires exactly one",
		},
		{
			name: "recovery stays loopback only",
			mutate: func(cfg *RouterConfig) {
				cfg.ManagementAPI.Auth.Recovery.Enabled = true
				cfg.ManagementAPI.Auth.Recovery.TokenFile = "/run/secrets/recovery"
				cfg.ManagementAPI.Auth.Recovery.LoopbackOnly = false
			},
			wantErr: "recovery.loopback_only must be true",
		},
		{
			name: "recovery token differs from bootstrap",
			mutate: func(cfg *RouterConfig) {
				cfg.ManagementAPI.Auth.Bootstrap.TokenFile = "/run/secrets/same"
				cfg.ManagementAPI.Auth.Recovery.Enabled = true
				cfg.ManagementAPI.Auth.Recovery.TokenFile = "/run/secrets/same"
			},
			wantErr: "separate token references",
		},
		{
			name: "secret files are absolute",
			mutate: func(cfg *RouterConfig) {
				cfg.AccessStore.Postgres.DSNFile = "relative/dsn"
			},
			wantErr: "absolute secret-file path",
		},
		{
			name: "secret references reject surrounding whitespace",
			mutate: func(cfg *RouterConfig) {
				cfg.AccessStore.Postgres.DSNFile = " /run/secrets/access-dsn"
			},
			wantErr: "surrounding whitespace",
		},
		{
			name: "disabled access still rejects ambiguous secret sources",
			mutate: func(cfg *RouterConfig) {
				cfg.Access.Enabled = false
				cfg.ControlPlane.PublicNamespaceID = "11111111-1111-4111-8111-111111111111"
				cfg.Access.Credentials.APIKeyHMACKeyringFile = "/run/secrets/api-key-peppers"
				cfg.Access.Credentials.APIKeyHMACKeyringEnv = "VLLM_SR_API_KEY_PEPPERS"
			},
			wantErr: "mutually exclusive",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := DefaultGlobalConfig()
			configureValidManagedAccess(&cfg)
			test.mutate(&cfg)
			err := cfg.ValidateControlPlaneBootstrap()
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ValidateControlPlaneBootstrap() error = %v, want fragment %q", err, test.wantErr)
			}
		})
	}
}

func TestParseV04ManagedBootstrapAndApplyDefaults(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(validManagedBootstrapYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.ControlPlane.Mode != ControlPlaneModeManaged {
		t.Fatalf("control-plane mode = %q", cfg.ControlPlane.Mode)
	}
	if cfg.Agent.PublicInferenceEndpoint != "http://public-inference.internal/v1/chat/completions" {
		t.Fatalf("Agent public inference endpoint = %q", cfg.Agent.PublicInferenceEndpoint)
	}
	if cfg.AccessStore == nil || cfg.AccessStore.Postgres.MaxConnections != defaultAccessMaxConnections {
		t.Fatalf("postgres defaults were not applied: %#v", cfg.AccessStore)
	}
	if cfg.AccessRuntimeStore == nil || cfg.AccessRuntimeStore.Redis.KeyPrefix != defaultAccessKeyPrefix {
		t.Fatalf("redis defaults were not applied: %#v", cfg.AccessRuntimeStore)
	}
	if cfg.Access.Enforcement.TokenAccounting != "response_actual" || cfg.Access.Enforcement.MaxUsageBacklog != defaultAccessUsageBacklog {
		t.Fatalf("access enforcement defaults were not applied: %#v", cfg.Access.Enforcement)
	}
	if cfg.Access.UsageStorage.CreateAheadMonths != defaultUsageCreateAhead ||
		cfg.Access.UsageStorage.MaintenanceInterval != defaultUsageMaintenance ||
		cfg.Access.UsageStorage.RawRetention != "" {
		t.Fatalf("usage storage defaults were not applied: %#v", cfg.Access.UsageStorage)
	}

	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported.ControlPlane.Mode != ControlPlaneModeManaged || exported.Stores.Access == nil || exported.Stores.AccessRuntime == nil {
		t.Fatalf("canonical export lost control-plane state: %#v", exported)
	}
	if exported.Services.Agent != cfg.Agent {
		t.Fatalf("canonical export lost Agent service state: %#v", exported.Services.Agent)
	}
	if version := CanonicalConfigFromRouterConfig(cfg).Version; version != "v0.4" {
		t.Fatalf("canonical version = %q, want v0.4", version)
	}
}

func TestCanonicalConfigRejectsUnknownVersion(t *testing.T) {
	_, err := ParseYAMLBytes([]byte("version: v9\nglobal: {}\n"))
	if err == nil || !strings.Contains(err.Error(), "version must be v0.4") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestParseV04RejectsSecretLiteralsAndUnknownBootstrapFields(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "literal postgres dsn",
			yaml: `version: v0.4
global:
  stores:
    access:
      postgres:
        dsn: postgres://user:password@db/access
`,
			wantErr: "dsn is forbidden",
		},
		{
			name: "literal management token",
			yaml: `version: v0.4
global:
  services:
    management_api:
      auth:
        bootstrap:
          token: plaintext
`,
			wantErr: "token is forbidden",
		},
		{
			name: "literal control-plane HMAC keyring",
			yaml: `version: v0.4
global:
  services:
    management_api:
      auth:
        control_plane_hmac_keyring: plaintext
`,
			wantErr: "control_plane_hmac_keyring is forbidden",
		},
		{
			name: "unknown access field",
			yaml: `version: v0.4
global:
  services:
    access:
      enabled: false
      unexpected_field: true
`,
			wantErr: "unsupported fields in global.services.access",
		},
		{
			name: "removed mutable calendar database version",
			yaml: `version: v0.4
global:
  services:
    access:
      enabled: false
      calendar_tzdb_version: iana-2025b
`,
			wantErr: "unsupported fields in global.services.access",
		},
		{
			name: "unknown usage storage field",
			yaml: `version: v0.4
global:
  services:
    access:
      enabled: false
      usage_storage:
        partition_interval: weekly
`,
			wantErr: "unsupported fields in global.services.access.usage_storage",
		},
		{
			name: "removed provider pack bootstrap",
			yaml: `version: v0.4
global:
  control_plane:
    provider_packs: [/etc/vllm-sr/providers.yaml]
`,
			wantErr: "unsupported fields in global.control_plane",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(test.yaml))
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ParseYAMLBytes() error = %v, want fragment %q", err, test.wantErr)
			}
		})
	}
}

func TestStandaloneBackendCredentialReferences(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.BackendCredentials.Standalone = map[string]BackendCredentialConfig{
		"private_provider": {CredentialAdapterID: "bearer", SecretEnv: "PRIVATE_PROVIDER_API_KEY"},
	}
	if err := cfg.ValidateControlPlaneBootstrap(); err != nil {
		t.Fatalf("standalone secret reference should validate: %v", err)
	}
	cfg.BackendCredentials.ProviderKEKKeyringFile = "/run/secrets/provider-kek"
	if err := cfg.ValidateControlPlaneBootstrap(); err == nil || !strings.Contains(err.Error(), "managed-only") {
		t.Fatalf("standalone managed KEK error = %v", err)
	}
}

func TestParseV04StandaloneBackendCredentialReference(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(`
version: v0.4
models:
  - name: worker
    card: {}
    connections:
      - provider: private-test
        endpoint: http://models.example
        model: worker
        credential: private_provider
recipes:
  - name: worker
    document:
      decisions:
        - name: worker
          rules: {}
entrypoints:
  - name: router/worker
    recipe: worker
    assignments:
      worker:
        models: [{model: worker}]
global:
  control_plane:
    mode: standalone
  services:
    access:
      enabled: false
    backend_egress:
      policy_file: /app/config/backend-egress-policy.yaml
    backend_credentials:
      private_provider:
        credential_adapter_id: bearer
        secret_env: PRIVATE_PROVIDER_API_KEY
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	credential, found := cfg.BackendCredentials.Standalone["private_provider"]
	if !found || credential.SecretEnv != "PRIVATE_PROVIDER_API_KEY" {
		t.Fatalf("standalone credential was not decoded: %#v", cfg.BackendCredentials.Standalone)
	}
}

func TestParseV04RejectsUnknownServiceFields(t *testing.T) {
	_, err := ParseYAMLBytes([]byte("version: v0.4\nglobal:\n  services:\n    unknown_service:\n      enabled: true\n"))
	if err == nil || !strings.Contains(err.Error(), "field unknown_service not found") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestParseV04RejectsRemovedProviderLayoutWithoutInspectingCredentials(t *testing.T) {
	for _, field := range []string{"api_key", "api_key_env", "access_key", "access_key_env"} {
		t.Run(field, func(t *testing.T) {
			document := `
version: v0.4
providers:
  models:
    - name: worker
      backend_refs:
        - endpoint: 127.0.0.1:8000
          ` + field + `: forbidden
routing:
  modelCards:
    - name: worker
`
			_, err := ParseYAMLBytes([]byte(document))
			if err == nil || !strings.Contains(err.Error(), "current v0.4") {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}
		})
	}
}

func TestStandaloneBackendCredentialReferenceMustExist(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.RoutingSnapshot = compileManagedSnapshotFixture(t)
	if err := cfg.ValidateControlPlaneBootstrap(); err == nil || !strings.Contains(err.Error(), "references undefined") {
		t.Fatalf("ValidateControlPlaneBootstrap() error = %v", err)
	}
}

func TestManagedModeAcceptsOpaquePublishedProviderCredentialID(t *testing.T) {
	cfg := DefaultGlobalConfig()
	configureValidManagedAccess(&cfg)
	cfg.RoutingSnapshot = compileManagedSnapshotFixture(t)
	if err := cfg.ValidateControlPlaneBootstrap(); err != nil {
		t.Fatalf("published provider credential ID should remain opaque at bootstrap: %v", err)
	}
}

func validAccessStore() *AccessStoreConfig {
	return &AccessStoreConfig{
		Type: AccessStoreTypePostgres,
		Postgres: PostgresAccessStoreConfig{
			DSNFile:        "/run/secrets/access-dsn",
			MaxConnections: defaultAccessMaxConnections,
		},
	}
}

func validAccessRuntimeStore() *AccessRuntimeStoreConfig {
	return &AccessRuntimeStoreConfig{
		Type: AccessRuntimeStoreTypeRedis,
		Redis: RedisAccessRuntimeStoreConfig{
			URLFile:   "/run/secrets/access-redis-url",
			KeyPrefix: defaultAccessKeyPrefix,
		},
	}
}

func configureValidManagedAccess(cfg *RouterConfig) {
	cfg.ControlPlane.Mode = ControlPlaneModeManaged
	cfg.Agent.PublicInferenceEndpoint = "http://public-inference.internal/v1/chat/completions"
	cfg.ManagementAPI.Auth.Mode = ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Tokens = nil
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.ControlPlane.ProviderCatalog = ProviderCatalogBootstrapConfig{
		ReplicaIDEnv: "VLLM_SR_REPLICA_ID", Lease: defaultProviderCatalogLease,
		RenewInterval: defaultProviderCatalogRenewInterval,
		RolloutGroups: []ProviderCatalogRolloutGroupConfig{
			{Plane: "control", ID: "management"}, {Plane: "data", ID: "router"},
		},
		RequiredRolloutGroups: []ProviderCatalogRolloutGroupConfig{
			{Plane: "control", ID: "management"}, {Plane: "data", ID: "router"},
		},
	}
	cfg.AccessStore = validAccessStore()
	cfg.AccessRuntimeStore = validAccessRuntimeStore()
	cfg.Access.Enabled = true
	cfg.Access.Credentials.APIKeyHMACKeyringFile = "/run/secrets/api-key-peppers"
	cfg.Access.Credentials.DelegationHMACKeyringFile = "/run/secrets/delegation-peppers"
	cfg.Access.TenantContext.SigningKeyFile = "/run/secrets/tenant-context-keys"
	cfg.BackendCredentials.ProviderKEKKeyringFile = "/run/secrets/provider-keks"
	cfg.BackendEgress.PolicyFile = "/etc/vllm-sr/backend-egress-policy.yaml"
	cfg.ManagementAPI.TLS.CertificateFile = "/run/secrets/management-cert"
	cfg.ManagementAPI.TLS.PrivateKeyFile = "/run/secrets/management-key"
	cfg.ManagementAPI.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing-keys"
	cfg.ManagementAPI.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/management-peppers"
	cfg.ManagementAPI.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-peppers"
	cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac-keys"
	cfg.ManagementAPI.Auth.ResponseKEKKeyringFile = "/run/secrets/response-keks"
}

const validManagedBootstrapYAML = `
version: v0.4
global:
  control_plane:
    mode: managed
    provider_catalog:
      replica_id_env: VLLM_SR_REPLICA_ID
      rollout_groups:
        - {plane: control, id: management}
        - {plane: data, id: router}
      required_rollout_groups:
        - {plane: control, id: management}
        - {plane: data, id: router}
  stores:
    access:
      type: postgres
      postgres:
        dsn_env: VLLM_SR_ACCESS_DATABASE_URL
    access_runtime:
      type: redis
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
    management_api:
      tls:
        certificate_file: /run/secrets/management-cert
        private_key_file: /run/secrets/management-key
      auth:
        mode: router
        token_signing_keyring_file: /run/secrets/management-signing-keys
        service_account_hmac_keyring_file: /run/secrets/management-peppers
        invitation_hmac_keyring_file: /run/secrets/invitation-peppers
        control_plane_hmac_keyring_file: /run/secrets/control-plane-hmac-keys
        response_kek_keyring_file: /run/secrets/response-keks
        bootstrap:
          token_file: /run/secrets/bootstrap-token
          disable_after_first_cluster_admin: true
`
