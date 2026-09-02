package config

import (
	"strings"
	"testing"
)

func TestShadowDispatchPluginConfigDefaults(t *testing.T) {
	cfg := ShadowDispatchPluginConfig{Enabled: true, Model: " candidate "}.WithDefaults()
	defaults := DefaultShadowDispatchPluginConfig()
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
	explicit := ShadowDispatchPluginConfig{MaxQueueDepth: 1, MaxConcurrency: 4}.WithDefaults()
	if explicit.MaxQueueDepth != 1 || explicit.MaxConcurrency != 4 {
		t.Fatalf("explicit bounds overwritten: %+v", explicit)
	}
}

func TestShadowDispatchPluginPayloadValidation(t *testing.T) {
	half := 0.5
	tooHigh := 1.5
	cases := []struct {
		name    string
		payload map[string]interface{}
		wantErr string
	}{
		{name: "valid", payload: map[string]interface{}{"enabled": true, "model": "candidate", "sample_rate": half, "max_retries": 3}},
		{name: "declared but disabled without model", payload: map[string]interface{}{"enabled": false}},
		{name: "missing model", payload: map[string]interface{}{"enabled": true}, wantErr: "model is required"},
		{name: "sample rate out of range", payload: map[string]interface{}{"enabled": true, "model": "m", "sample_rate": tooHigh}, wantErr: "sample_rate"},
		{name: "negative bound", payload: map[string]interface{}{"enabled": true, "model": "m", "max_concurrency": -1}, wantErr: "max_concurrency cannot be negative"},
		{name: "too many retries", payload: map[string]interface{}{"enabled": true, "model": "m", "max_retries": 4}, wantErr: "max_retries cannot exceed"},
		{name: "unknown field", payload: map[string]interface{}{"enabled": true, "model": "m", "bogus": 1}, wantErr: "bogus"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDecisionPluginPayload("route", 0, DecisionPlugin{
				Type:          DecisionPluginShadowDispatch,
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

func TestShadowDispatchModelMustHaveBackend(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"primary":   {PreferredEndpoints: []string{"ep"}},
				"candidate": {PreferredEndpoints: []string{"ep"}},
			},
			VLLMEndpoints: []VLLMEndpoint{{Name: "ep", Address: "127.0.0.1", Port: 8000}},
		},
	}
	decision := &Decision{
		Name:      "route",
		ModelRefs: []ModelRef{{Model: "primary"}},
		Plugins: []DecisionPlugin{{
			Type: DecisionPluginShadowDispatch,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"model":   "candidate",
			}),
		}},
	}
	if err := validateDecisionShadowDispatchPlugin(cfg, decision); err != nil {
		t.Fatalf("configured shadow model rejected: %v", err)
	}
	if got := decision.GetShadowDispatchConfig(); got == nil || got.Model != "candidate" {
		t.Fatalf("GetShadowDispatchConfig = %+v", got)
	}

	decision.Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
		"enabled": true,
		"model":   "unknown",
	})
	err := validateDecisionShadowDispatchPlugin(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "no configured backend") {
		t.Fatalf("error = %v, want unknown backend rejection", err)
	}

	if (&Decision{Name: "plain"}).GetShadowDispatchConfig() != nil {
		t.Fatal("decision without plugin must return nil config")
	}

	decision.Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{
		"enabled": true,
		"model":   "candidate",
	})
	decision.Algorithm = &AlgorithmConfig{Type: DecisionAlgorithmRatings}
	err = validateDecisionShadowDispatchPlugin(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "looper-executed") {
		t.Fatalf("error = %v, want looper rejection", err)
	}
}

func TestShadowDispatchValidateSharedWithConverter(t *testing.T) {
	cfg := ShadowDispatchPluginConfig{Enabled: true, Model: "m", MaxRetries: 4}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "max_retries cannot exceed") {
		t.Fatalf("Validate = %v, want retries cap", err)
	}
	var nilCfg *ShadowDispatchPluginConfig
	if err := nilCfg.Validate(); err != nil {
		t.Fatalf("nil Validate = %v", err)
	}
}
