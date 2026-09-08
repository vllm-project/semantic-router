package config

import "testing"

func TestAuthzConfigHasExternalAuthProvider(t *testing.T) {
	tests := []struct {
		name      string
		providers []AuthzProviderConfig
		want      bool
	}{
		{name: "no providers"},
		{
			name:      "static config only",
			providers: []AuthzProviderConfig{{Type: "static-config"}},
		},
		{
			name:      "header injection",
			providers: []AuthzProviderConfig{{Type: "header-injection"}},
			want:      true,
		},
		{
			name: "header injection in a chain",
			providers: []AuthzProviderConfig{
				{Type: "static-config"},
				{Type: "header-injection"},
			},
			want: true,
		},
		{
			name:      "invalid provider spelling fails closed",
			providers: []AuthzProviderConfig{{Type: "Header-Injection"}},
		},
		{
			name: "invalid header injection mapping fails closed",
			providers: []AuthzProviderConfig{{
				Type:    "header-injection",
				Headers: map[string]string{"openai": ""},
			}},
		},
		{
			name: "unknown provider in chain fails closed",
			providers: []AuthzProviderConfig{
				{Type: "header-injection"},
				{Type: "unknown"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := AuthzConfig{Providers: tt.providers}
			if got := cfg.HasExternalAuthProvider(); got != tt.want {
				t.Fatalf("HasExternalAuthProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}
