package config

import "testing"

func TestDefaultManagementAPIConfigUsesLoopbackAndDisabledAuth(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	if cfg.Enabled {
		t.Fatal("management API should default to disabled")
	}
	if cfg.BindAddress != "127.0.0.1" {
		t.Fatalf("bind_address = %q, want 127.0.0.1", cfg.BindAddress)
	}
	if cfg.Port != 8080 {
		t.Fatalf("port = %d, want 8080", cfg.Port)
	}
	if cfg.RemoteExposure {
		t.Fatal("remote_exposure should default to false")
	}
	if cfg.Auth.Mode != ManagementAuthModeDisabled {
		t.Fatalf("auth.mode = %q, want %q", cfg.Auth.Mode, ManagementAuthModeDisabled)
	}
}

func TestResolvedManagementAPIPreservesConfigRemoteExposureWhenOverrideNil(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.RemoteExposure = true
	cfg.Auth.Mode = ManagementAuthModeBearer
	cfg.Auth.Tokens = []ManagementAPITokenRef{{Env: "VSR_MGMT_TOKEN", Role: "admin"}}
	t.Setenv("VSR_MGMT_TOKEN", "test-token")

	resolved, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{})
	if err != nil {
		t.Fatalf("ResolvedManagementAPI() error = %v", err)
	}
	if !resolved.RemoteExposure {
		t.Fatal("nil RemoteExposure override must preserve config remote_exposure: true")
	}
}

func TestResolvedManagementAPIRejectsRemoteExposureWithoutBearerTokens(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	remote := true
	_, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		RemoteExposure: &remote,
		AuthMode:       ManagementAuthModeBearer,
	})
	if err == nil {
		t.Fatal("expected remote exposure without tokens to fail")
	}
}

func TestResolvedManagementAPIRejectsWideBindWithoutRemoteExposure(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	_, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		BindAddress: "0.0.0.0",
	})
	if err == nil {
		t.Fatal("expected wide bind without remote_exposure to fail")
	}
}

func TestResolvedManagementAPIAllowsWideBindForInternalListener(t *testing.T) {
	t.Setenv(ManagementInternalListenerEnv, "true")
	cfg := DefaultManagementAPIConfig()
	resolved, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		BindAddress: "0.0.0.0",
	})
	if err != nil {
		t.Fatalf("ResolvedManagementAPI() error = %v", err)
	}
	if resolved.BindAddress != "0.0.0.0" {
		t.Fatalf("bind_address = %q, want 0.0.0.0", resolved.BindAddress)
	}
}

func TestResolvedManagementAPIAppliesCLIOverrides(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	remote := true
	_, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		Port:           9090,
		BindAddress:    "0.0.0.0",
		RemoteExposure: &remote,
		AuthMode:       ManagementAuthModeBearer,
	})
	if err == nil {
		t.Fatal("expected validation error without configured tokens")
	}

	t.Setenv(ManagementInternalListenerEnv, "true")
	resolved, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		Port:        9090,
		BindAddress: "0.0.0.0",
		AuthMode:    ManagementAuthModeDisabled,
	})
	if err != nil {
		t.Fatalf("ResolvedManagementAPI() error = %v", err)
	}
	if resolved.Port != 9090 {
		t.Fatalf("port = %d, want 9090", resolved.Port)
	}
	if resolved.BindAddress != "0.0.0.0" {
		t.Fatalf("bind_address = %q, want 0.0.0.0", resolved.BindAddress)
	}
	if resolved.ListenAddress() != "0.0.0.0:9090" {
		t.Fatalf("listen address = %q", resolved.ListenAddress())
	}
}

func TestResolvedManagementAPIRejectsOutOfRangeConfiguredPort(t *testing.T) {
	for _, port := range []int{-1, 65536} {
		cfg := DefaultManagementAPIConfig()
		cfg.Port = port
		if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{}); err == nil {
			t.Fatalf("port %d should be rejected", port)
		}
	}
}

func TestResolvedManagementAPIRejectsUnsupportedAuthMode(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = "mtls"
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{}); err == nil {
		t.Fatal("unsupported management auth mode should be rejected")
	}
}

func TestResolvedManagementAPIValidatesDurableSecurity(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Enabled = true
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"

	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		DurableRouting: true,
	}); err != nil {
		t.Fatalf("durable ResolvedManagementAPI() error = %v", err)
	}
}

func TestResolvedManagementAPIRejectsAmbiguousDurableTLSSources(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Enabled = true
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.CertificateEnv = "VLLM_SR_MANAGEMENT_TLS_CERT"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"

	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		DurableRouting: true,
	}); err == nil {
		t.Fatal("durable Management TLS should reject ambiguous certificate sources")
	}
}

func TestResolvedManagementAPIRequiresRouterAuthForDurableAuthority(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	configureDurableRoutingSecurity(&cfg)
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		DurableRouting: true,
	}); err == nil {
		t.Fatal("durable Management should reject file-authoritative authentication")
	}
}

func TestResolvedManagementAPIRejectsStaticAuthorizationWithDurableAuthority(t *testing.T) {
	for _, mutate := range []func(*ManagementAPIConfig){
		func(cfg *ManagementAPIConfig) {
			cfg.Auth.Tokens = []ManagementAPITokenRef{{Env: "VSR_MGMT_TOKEN", Role: "admin"}}
		},
		func(cfg *ManagementAPIConfig) { cfg.Auth.Roles = DefaultManagementAPIRoles() },
	} {
		cfg := DefaultManagementAPIConfig()
		cfg.Auth.Mode = ManagementAuthModeRouter
		cfg.Auth.Roles = nil
		configureDurableRoutingSecurity(&cfg)
		mutate(&cfg)
		if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
			DurableRouting: true,
		}); err == nil {
			t.Fatal("durable Management should reject static tokens and roles")
		}
	}
}

func TestResolvedManagementAPIRejectsRouterAuthWithoutDurableAuthority(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{}); err == nil {
		t.Fatal("file-authoritative configuration should reject Router-native Management authentication")
	}
}

func TestResolvedManagementAPIAllowsDurableRemoteExposureWithoutStaticTokens(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.RemoteExposure = true
	cfg.BindAddress = "0.0.0.0"
	configureDurableRoutingSecurity(&cfg)
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		DurableRouting: true,
	}); err != nil {
		t.Fatalf("durable Router authentication should not require static bearer tokens: %v", err)
	}
}

func configureDurableRoutingSecurity(cfg *ManagementAPIConfig) {
	cfg.Enabled = true
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"
}
