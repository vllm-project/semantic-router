package config

import "testing"

func TestIdentityConfigHasVerifiedIngress(t *testing.T) {
	tests := []struct {
		name    string
		ingress string
		want    bool
	}{
		{name: "no ingress"},
		{name: "credential provider is not identity trust", ingress: "", want: false},
		{name: "verified header injection", ingress: IdentityIngressHeaderInjection, want: true},
		{name: "invalid spelling fails closed", ingress: "Header-Injection", want: false},
		{name: "unknown ingress fails closed", ingress: "jwt", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			identity := IdentityConfig{Ingress: tt.ingress}
			if got := identity.HasVerifiedIngress(); got != tt.want {
				t.Fatalf("HasVerifiedIngress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAuthzCredentialProviderDoesNotEstablishIdentityTrust(t *testing.T) {
	cfg := AuthzConfig{
		Providers: []AuthzProviderConfig{{Type: "header-injection"}},
	}
	if len(cfg.Providers) != 1 || cfg.Providers[0].Type != "header-injection" {
		t.Fatalf("credential providers = %#v, want one header-injection provider", cfg.Providers)
	}
	if cfg.Identity.HasVerifiedIngress() {
		t.Fatal("credential header-injection provider must not establish identity trust")
	}
}
