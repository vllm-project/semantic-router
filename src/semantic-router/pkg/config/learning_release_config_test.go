package config

import (
	"strings"
	"testing"
)

func TestProtectionReleaseConfigDefaults(t *testing.T) {
	var cfg RouterLearningProtectionReleaseConfig
	if !cfg.EffectiveEnabled() {
		t.Fatal("expected release enabled by default")
	}
	if got := cfg.EffectiveHeader(); got != "x-vsr-protection-release" {
		t.Fatalf("expected default header, got %q", got)
	}
	if got := cfg.EffectiveMinUserTurnsBetween(); got != 2 {
		t.Fatalf("expected default min_user_turns_between 2, got %d", got)
	}
	header := " X-My-Release "
	cfg.Header = &header
	if got := cfg.EffectiveHeader(); got != "x-my-release" {
		t.Fatalf("expected trimmed lower-case header, got %q", got)
	}
}

func TestValidateProtectionReleaseConfig(t *testing.T) {
	str := func(value string) *string { return &value }
	num := func(value int) *int { return &value }
	for _, tc := range []struct {
		name    string
		release RouterLearningProtectionReleaseConfig
		agent   *string
		wantErr string
	}{
		{name: "defaults"},
		{name: "blank_header", release: RouterLearningProtectionReleaseConfig{Header: str("  ")}, wantErr: "protection.release.header cannot be empty"},
		{name: "invalid_header", release: RouterLearningProtectionReleaseConfig{Header: str("x release")}, wantErr: "not a valid header name"},
		{name: "colon_header", release: RouterLearningProtectionReleaseConfig{Header: str("x-release:")}, wantErr: "not a valid header name"},
		{name: "session_header", release: RouterLearningProtectionReleaseConfig{Header: str("X-Session-Id")}, wantErr: "must differ from identity"},
		{name: "conversation_header", release: RouterLearningProtectionReleaseConfig{Header: str("x-conversation-id")}, wantErr: "must differ from identity"},
		{name: "agent_header", release: RouterLearningProtectionReleaseConfig{Header: str("x-agent")}, agent: str("x-agent"), wantErr: "must differ from identity"},
		{name: "authorization_header", release: RouterLearningProtectionReleaseConfig{Header: str("Authorization")}, wantErr: "is reserved"},
		{name: "api_key_header", release: RouterLearningProtectionReleaseConfig{Header: str("x-api-key")}, wantErr: "is reserved"},
		{name: "router_control_header", release: RouterLearningProtectionReleaseConfig{Header: str("x-vsr-selected-model")}, wantErr: "is reserved"},
		{name: "default_header_explicit", release: RouterLearningProtectionReleaseConfig{Header: str("x-vsr-protection-release")}},
		{name: "negative_interval", release: RouterLearningProtectionReleaseConfig{MinUserTurnsBetween: num(-1)}, wantErr: "must be >= 0"},
		{name: "zero_interval", release: RouterLearningProtectionReleaseConfig{MinUserTurnsBetween: num(0)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := RouterLearningProtectionConfig{Release: tc.release}
			cfg.Identity.Headers.Agent = tc.agent
			err := validateRouterLearningProtectionConfig(cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

const protectionReleaseYAMLPrefix = `
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        release:
`

func TestParseYAMLBytesAcceptsProtectionRelease(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(protectionReleaseYAMLPrefix + `
          enabled: false
          header: x-my-release
          min_user_turns_between: 3
`))
	if err != nil {
		t.Fatalf("expected release config to load: %v", err)
	}
	release := cfg.RouterLearning.Protection.Release
	if release.EffectiveEnabled() || release.EffectiveHeader() != "x-my-release" || release.EffectiveMinUserTurnsBetween() != 3 {
		t.Fatalf("release config not loaded: %+v", release)
	}
}

func TestParseYAMLBytesRejectsUnknownProtectionReleaseField(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(protectionReleaseYAMLPrefix + `
          typo: 1
`))
	if err == nil || !strings.Contains(err.Error(), "global.router.learning.protection.release.typo") {
		t.Fatalf("expected unknown release field in error, got %v", err)
	}
}
