package config

import "testing"

func TestDefaultManagementAPIConfigUsesLoopbackAndDisabledAuth(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
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

func TestResolvedManagementAPIValidatesManagedSecurityInManagedMode(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac"
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"

	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		ControlPlaneMode: ControlPlaneModeManaged,
	}); err != nil {
		t.Fatalf("managed ResolvedManagementAPI() error = %v", err)
	}
}

func TestResolvedManagementAPIRejectsAmbiguousManagedTLSSources(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac"
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.CertificateEnv = "VLLM_SR_MANAGEMENT_TLS_CERT"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"

	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		ControlPlaneMode: ControlPlaneModeManaged,
	}); err == nil {
		t.Fatal("managed Management TLS should reject ambiguous certificate sources")
	}
}

func TestResolvedManagementAPIRequiresRouterAuthInManagedMode(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	configureManagedManagementSecurity(&cfg)
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		ControlPlaneMode: ControlPlaneModeManaged,
	}); err == nil {
		t.Fatal("managed mode should reject standalone Management authentication")
	}
}

func TestResolvedManagementAPIRejectsStaticAuthorizationInManagedMode(t *testing.T) {
	for _, mutate := range []func(*ManagementAPIConfig){
		func(cfg *ManagementAPIConfig) {
			cfg.Auth.Tokens = []ManagementAPITokenRef{{Env: "VSR_MGMT_TOKEN", Role: "admin"}}
		},
		func(cfg *ManagementAPIConfig) { cfg.Auth.Roles = DefaultManagementAPIRoles() },
	} {
		cfg := DefaultManagementAPIConfig()
		cfg.Auth.Mode = ManagementAuthModeRouter
		cfg.Auth.Roles = nil
		configureManagedManagementSecurity(&cfg)
		mutate(&cfg)
		if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
			ControlPlaneMode: ControlPlaneModeManaged,
		}); err == nil {
			t.Fatal("managed mode should reject static tokens and roles")
		}
	}
}

func TestResolvedManagementAPIRejectsRouterAuthInStandaloneMode(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{}); err == nil {
		t.Fatal("standalone mode should reject Router-native Management authentication")
	}
}

func TestResolvedManagementAPIAllowsManagedRemoteExposureWithoutStaticTokens(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.Mode = ManagementAuthModeRouter
	cfg.Auth.Roles = nil
	cfg.RemoteExposure = true
	cfg.BindAddress = "0.0.0.0"
	configureManagedManagementSecurity(&cfg)
	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{
		ControlPlaneMode: ControlPlaneModeManaged,
	}); err != nil {
		t.Fatalf("managed Router authentication should not require static bearer tokens: %v", err)
	}
}

func configureManagedManagementSecurity(cfg *ManagementAPIConfig) {
	cfg.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac"
	cfg.Auth.TokenSigningKeyringFile = "/run/secrets/management-signing"
	cfg.Auth.ServiceAccountHMACKeyringFile = "/run/secrets/service-account-hmac"
	cfg.Auth.InvitationHMACKeyringFile = "/run/secrets/invitation-hmac"
	cfg.Auth.ResponseKEKKeyringFile = "/run/secrets/response-kek"
	cfg.TLS.CertificateFile = "/run/secrets/management-tls-cert"
	cfg.TLS.PrivateKeyFile = "/run/secrets/management-tls-key"
}

func TestResolvedManagementAPIRejectsManagedOnlyKeyringInStandaloneMode(t *testing.T) {
	cfg := DefaultManagementAPIConfig()
	cfg.Auth.ControlPlaneHMACKeyringFile = "/run/secrets/control-plane-hmac"

	if _, err := cfg.ResolvedManagementAPI(ManagementAPIRuntimeOptions{}); err == nil {
		t.Fatal("standalone mode should reject the managed-only control-plane keyring")
	}
}
