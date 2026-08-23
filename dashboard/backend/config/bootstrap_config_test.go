package config

import "testing"

func TestRouterBootstrapRequiresCanonicalFileAndPrivateIssuer(t *testing.T) {
	t.Parallel()
	complete := Config{
		RouterBootstrapTokenFile: "/run/secrets/bootstrap/router-token",
		DashboardIssuer:          "https://dashboard:8743", DashboardIssuerID: "10000000-0000-4000-8000-000000000001",
		DashboardSigningKeyFile: "/run/secrets/dashboard-signing.pem", DashboardKeyID: "local-v1",
	}
	if err := validateRouterBootstrapConfig(&complete, true); err != nil {
		t.Fatalf("complete config error = %v", err)
	}
	for _, testCase := range []struct {
		name string
		edit func(*Config)
		tls  bool
	}{
		{name: "relative token path", edit: func(cfg *Config) { cfg.RouterBootstrapTokenFile = "secrets/router-token" }, tls: true},
		{name: "missing issuer", edit: func(cfg *Config) { cfg.DashboardIssuer = "" }, tls: true},
		{name: "missing issuer tls", edit: func(*Config) {}, tls: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			cfg := complete
			testCase.edit(&cfg)
			if err := validateRouterBootstrapConfig(&cfg, testCase.tls); err == nil {
				t.Fatal("validateRouterBootstrapConfig() unexpectedly succeeded")
			}
		})
	}
}
