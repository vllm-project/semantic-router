package mcp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestValidateSecurityRejectsSettingsTheClientCannotEnforce(t *testing.T) {
	const secretCanary = "validate-security-secret-canary" //nolint:gosec // Canary verifies rejection messages never echo credentials.
	tests := []struct {
		name      string
		security  *SecurityConfig
		wantField string
	}{
		{name: "absent"},
		{name: "cleared", security: &SecurityConfig{}},
		{
			name:      "oauth",
			security:  &SecurityConfig{OAuth: &OAuthConfig{ClientID: "dashboard", ClientSecret: secretCanary}},
			wantField: "security.oauth",
		},
		{name: "local only", security: &SecurityConfig{LocalOnly: true}, wantField: "security.local_only"},
		{
			name:      "allowed origins",
			security:  &SecurityConfig{AllowedOrigins: []string{"https://dashboard.example.test"}},
			wantField: "security.allowed_origins",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateSecurity(test.security)
			if test.wantField == "" {
				if err != nil {
					t.Fatalf("ValidateSecurity() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, ErrUnsupportedSecurity) {
				t.Fatalf("ValidateSecurity() = %v, want ErrUnsupportedSecurity", err)
			}
			if !strings.Contains(err.Error(), test.wantField) || strings.Contains(err.Error(), secretCanary) {
				t.Fatalf("error must name %s without echoing credentials: %q", test.wantField, err)
			}
		})
	}
}

func TestClientConnectRefusesUnsupportedSecurityBeforeContactingServer(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusTeapot)
	}))
	t.Cleanup(server.Close)

	client, err := NewClient(&ServerConfig{
		Name:       "oauth-server",
		Transport:  TransportStreamableHTTP,
		Connection: ConnectionConfig{URL: server.URL + "/mcp"},
		Security:   &SecurityConfig{OAuth: &OAuthConfig{ClientID: "dashboard", TokenURL: server.URL + "/token"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	err = client.Connect(context.Background())
	if !errors.Is(err, ErrUnsupportedSecurity) {
		t.Fatalf("Connect() = %v, want ErrUnsupportedSecurity", err)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("server received %d requests, want none", got)
	}
	if got := client.GetStatus(); got != StatusError {
		t.Fatalf("status = %q, want %q", got, StatusError)
	}
}
