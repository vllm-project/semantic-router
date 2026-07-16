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
