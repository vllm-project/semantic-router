package config

import (
	"strings"
	"testing"
)

func TestRouterReplayShadowDefaults(t *testing.T) {
	cfg := RouterReplayShadowConfig{Model: " candidate "}.WithDefaults()
	defaults := DefaultRouterReplayShadowConfig()
	if cfg.Model != "candidate" {
		t.Fatalf("model = %q, want trimmed", cfg.Model)
	}
	if cfg.MaxConcurrency != defaults.MaxConcurrency ||
		cfg.MaxQueueDepth != defaults.MaxQueueDepth ||
		cfg.TimeoutSeconds != defaults.TimeoutSeconds ||
		cfg.MaxResponseBytes != defaults.MaxResponseBytes ||
		cfg.MaxCaptureBytes != defaults.MaxCaptureBytes {
		t.Fatalf("defaults not applied: %+v", cfg)
	}
	if cfg.EffectiveSampleRate() != 1 {
		t.Fatalf("effective sample rate = %v, want 1", cfg.EffectiveSampleRate())
	}
	explicit := RouterReplayShadowConfig{MaxQueueDepth: 1, MaxConcurrency: 4}.WithDefaults()
	if explicit.MaxQueueDepth != 1 || explicit.MaxConcurrency != 4 {
		t.Fatalf("explicit bounds overwritten: %+v", explicit)
	}
}

func TestRouterReplayShadowPayloadValidation(t *testing.T) {
	half := 0.5
	tooHigh := 1.5
	shadow := func(fields map[string]interface{}) map[string]interface{} {
		return map[string]interface{}{"enabled": true, "shadow": fields}
	}
	cases := []struct {
		name    string
		payload map[string]interface{}
		wantErr string
	}{
		{name: "valid", payload: shadow(map[string]interface{}{"model": "candidate", "sample_rate": half, "max_retries": 3})},
		{name: "no shadow", payload: map[string]interface{}{"enabled": true}},
		{name: "missing model", payload: shadow(map[string]interface{}{"sample_rate": half}), wantErr: "model or arms is required"},
		{name: "arms only", payload: shadow(map[string]interface{}{"arms": []string{"candidate"}})},
		{name: "sample rate out of range", payload: shadow(map[string]interface{}{"model": "m", "sample_rate": tooHigh}), wantErr: "sample_rate"},
		{name: "negative bound", payload: shadow(map[string]interface{}{"model": "m", "max_concurrency": -1}), wantErr: "max_concurrency cannot be negative"},
		{name: "too many retries", payload: shadow(map[string]interface{}{"model": "m", "max_retries": 4}), wantErr: "max_retries cannot exceed"},
		{name: "unknown shadow field", payload: shadow(map[string]interface{}{"model": "m", "bogus": 1}), wantErr: "bogus"},
		{name: "unknown replay field", payload: map[string]interface{}{"enabled": true, "bogus": 1}, wantErr: "bogus"},
		{name: "forward headers allowlist", payload: shadow(map[string]interface{}{"model": "m", "forward_headers": []string{"x-tenant", "X-Request-Source"}})},
		{name: "forward headers reject credentials", payload: shadow(map[string]interface{}{"model": "m", "forward_headers": []string{"x-tenant", "Authorization"}}), wantErr: "forward_headers cannot include credential header"},
		{name: "forward headers reject user keys", payload: shadow(map[string]interface{}{"model": "m", "forward_headers": []string{"X-User-OpenAI-Key"}}), wantErr: "forward_headers cannot include credential header"},
		{name: "forward headers reject pseudo headers", payload: shadow(map[string]interface{}{"model": "m", "forward_headers": []string{":authority"}}), wantErr: "pseudo-header"},
		{name: "forward headers reject blanks", payload: shadow(map[string]interface{}{"model": "m", "forward_headers": []string{" "}}), wantErr: "cannot be empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DecodeDecisionPluginAt(PluginAt{Decision: "route", Type: DecisionPluginRouterReplay}, DecisionPlugin{
				Type:          DecisionPluginRouterReplay,
				Configuration: MustStructuredPayload(tc.payload),
			})
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestRouterReplayShadowModelMustHaveBackend(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"primary":   {PreferredEndpoints: []string{"ep"}},
				"candidate": {PreferredEndpoints: []string{"ep"}},
			},
			VLLMEndpoints: []VLLMEndpoint{{Name: "ep", Address: "127.0.0.1", Port: 8000}},
		},
	}
	replay := func(model string) []DecisionPlugin {
		return []DecisionPlugin{{
			Type: DecisionPluginRouterReplay,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"shadow":  map[string]interface{}{"model": model},
			}),
		}}
	}
	decision := &Decision{Name: "route", ModelRefs: []ModelRef{{Model: "primary"}}, Plugins: replay("candidate")}
	if err := validateDecisionReplayShadow(cfg, decision); err != nil {
		t.Fatalf("configured shadow model rejected: %v", err)
	}
	if got := cfg.EffectiveRouterReplayConfig(decision); got == nil || got.Shadow == nil || got.Shadow.Model != "candidate" {
		t.Fatalf("EffectiveRouterReplayConfig = %+v", got)
	}

	decision.Plugins = replay("unknown")
	err := validateDecisionReplayShadow(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "no configured backend") {
		t.Fatalf("error = %v, want unknown backend rejection", err)
	}

	decision.Plugins = replay("candidate")
	decision.Algorithm = &AlgorithmConfig{Type: DecisionAlgorithmRatings}
	err = validateDecisionReplayShadow(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "executor algorithm") {
		t.Fatalf("error = %v, want executor rejection", err)
	}
}

func TestRouterReplayShadowBudgetCapsRejectSilentCombinations(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ShadowDispatchBudgetConfig
		wantErr string
	}{
		{name: "token cap without admission reserve", cfg: ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100}, wantErr: "requires reserve_tokens_per_arm"},
		{name: "cost cap without price", cfg: ShadowDispatchBudgetConfig{MaxCostPerRequest: 1.0}, wantErr: "requires reserve_tokens_per_arm and price_per_million_tokens"},
		{name: "cost cap without admission reserve", cfg: ShadowDispatchBudgetConfig{MaxCostPerRequest: 1.0, PricePerMillionTokens: 2.0}, wantErr: "requires reserve_tokens_per_arm and price_per_million_tokens"},
		{name: "bounded pair accepted", cfg: ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100, ReserveTokensPerArm: 50, MaxCostPerRequest: 1.0, PricePerMillionTokens: 2.0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := (&RouterReplayShadowConfig{Model: "m", Budget: tc.cfg}).Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestRouterReplayShadowValidateSharedWithConverter(t *testing.T) {
	cfg := RouterReplayShadowConfig{Model: "m", MaxRetries: 4}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "max_retries cannot exceed") {
		t.Fatalf("Validate = %v, want retries cap", err)
	}
	var nilCfg *RouterReplayShadowConfig
	if err := nilCfg.Validate(); err != nil {
		t.Fatalf("nil Validate = %v", err)
	}
}

func TestRouterReplayShadowForwardsHeaderIsAllowlistOnly(t *testing.T) {
	var nilCfg *RouterReplayShadowConfig
	if nilCfg.ForwardsHeader("x-tenant") {
		t.Fatal("nil config must forward nothing")
	}
	empty := &RouterReplayShadowConfig{}
	if empty.ForwardsHeader("x-tenant") {
		t.Fatal("empty allowlist must forward nothing")
	}
	cfg := &RouterReplayShadowConfig{ForwardHeaders: []string{" X-Tenant "}}
	if !cfg.ForwardsHeader("x-tenant") || !cfg.ForwardsHeader("X-TENANT") {
		t.Fatal("allowlist must match case-insensitively and ignore surrounding whitespace")
	}
	if cfg.ForwardsHeader("x-tenant-id") {
		t.Fatal("allowlist must match whole names only")
	}
	for _, name := range []string{"Authorization", "x-api-key", "API-KEY", "Cookie", "x-user-openai-key", " proxy-authorization "} {
		if !IsShadowCredentialHeader(name) {
			t.Errorf("IsShadowCredentialHeader(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"x-tenant", "traceparent", "x-authorization-hint"} {
		if IsShadowCredentialHeader(name) {
			t.Errorf("IsShadowCredentialHeader(%q) = true, want false", name)
		}
	}
}
