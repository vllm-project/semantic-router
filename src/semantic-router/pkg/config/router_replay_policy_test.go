package config

import (
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestDecisionBalanceReplayActivatesPIIEvidence(t *testing.T) {
	raw, err := os.ReadFile("../../../../config/recipes/decision-balance/config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseYAMLBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	policy := cfg.EffectiveRouterReplayConfig(nil)
	if policy == nil || policy.CapturesPersonalData() || !cfg.ReplayNeedsPIIEvidence() || !cfg.NeedsPIIMappingForRouting() {
		t.Fatal("DecisionBalance must prepare PII to suppress personal replay content")
	}
}

func TestReplayGlobalDefaultsAndExplicitDecisionOverrides(t *testing.T) {
	no, yes, zero := false, true, 0
	cfg := &RouterConfig{RouterReplay: RouterReplayConfig{Enabled: true, CaptureRequestBody: &no, CapturePersonalData: &no, MaxToolTraceSteps: &zero}}
	inherited := cfg.EffectiveRouterReplayConfig(nil)
	if inherited.CaptureRequestBody || inherited.CapturesPersonalData() || inherited.MaxToolTraceSteps != 0 || !inherited.CaptureResponseBody {
		t.Fatalf("incorrect global defaults: %+v", inherited)
	}
	d := &Decision{Plugins: []DecisionPlugin{{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(map[string]interface{}{
		"capture_request_body": true, "capture_personal_data": true, "capture_response_body": false, "max_body_bytes": 0,
	})}}}
	got := cfg.EffectiveRouterReplayConfig(d)
	if !got.CaptureRequestBody || !got.CapturesPersonalData() || got.CaptureResponseBody || got.MaxBodyBytes != 0 || got.MaxToolTraceSteps != 0 {
		t.Fatalf("incorrect overlay: %+v", got)
	}
	if *cfg.RouterReplay.CapturePersonalData || *cfg.RouterReplay.CaptureRequestBody {
		t.Fatal("resolution mutated global defaults")
	}
	for _, global := range []bool{false, true} {
		for _, override := range []*bool{nil, &yes, &no} {
			cfg.RouterReplay.Enabled, d.Plugins = global, nil
			want := global
			if override != nil {
				want = *override
				d.Plugins = []DecisionPlugin{{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(map[string]interface{}{"enabled": *override})}}
			}
			if (cfg.EffectiveRouterReplayConfig(d) != nil) != want {
				t.Fatalf("global %v override %v", global, override)
			}
		}
	}
}

func TestReplayCaptureDefaultsRoundTripWithoutPIIRules(t *testing.T) {
	raw := []byte(`version: v0.3
global:
  services:
    router_replay:
      enabled: true
      capture_request_body: false
      capture_response_body: false
      capture_personal_data: false
      max_records: 77
      max_body_bytes: 0
      max_tool_trace_bytes: 0
      max_tool_trace_steps: 0
`)
	cfg, err := ParseYAMLBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ReplayNeedsPIIEvidence() || cfg.UsesSignalTypeInReachableRouting(SignalTypePII) {
		t.Fatal("no PII rules must not add a model dependency")
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	got := again.EffectiveRouterReplayConfig(nil)
	if got == nil || got.CapturesPersonalData() || got.CaptureRequestBody || got.CaptureResponseBody || got.MaxRecords != 77 || got.MaxBodyBytes != 0 || got.MaxToolTraceBytes != 0 || got.MaxToolTraceSteps != 0 {
		t.Fatalf("roundtrip lost defaults: %+v", got)
	}
}

func TestRemovedDataPolicyIsUnknownInCanonicalAndFragments(t *testing.T) {
	for _, field := range []string{"replay: false", "replay_personal_data: false"} {
		for _, prefix := range []string{"routing:\n  ", "recipes:\n  - name: other\n    routing:\n      "} {
			_, err := ParseYAMLBytes([]byte("version: v0.3\n" + prefix + "data_policy: {" + field + "}\n"))
			if err == nil || !strings.Contains(err.Error(), "unknown field \"data_policy\"") {
				t.Fatalf("want unknown data_policy: %v", err)
			}
		}
		_, err := ParseRoutingYAMLBytes([]byte("routing:\n  data_policy: {" + field + "}\n"))
		if err == nil || !strings.Contains(err.Error(), "unknown field \"data_policy\"") {
			t.Fatalf("fragment accepted data_policy: %v", err)
		}
	}
}

func TestReplayPIIDemandTracksPossibleEffectivePolicies(t *testing.T) {
	no := false
	cfg := &RouterConfig{RouterReplay: RouterReplayConfig{Enabled: true, CapturePersonalData: &no}, IntelligentRouting: IntelligentRouting{Signals: Signals{PIIRules: []PIIRule{{Name: "personal"}}}}}
	if !cfg.ReplayNeedsPIIEvidence() || !cfg.UsesSignalTypeInRouting(SignalTypePII) {
		t.Fatal("global policy must consume existing PII rules")
	}
	cfg.RouterReplay.Enabled = false
	if cfg.ReplayNeedsPIIEvidence() {
		t.Fatal("disabled replay must not consume PII")
	}
	cfg.Decisions = []Decision{{Plugins: []DecisionPlugin{{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(map[string]interface{}{"enabled": true})}}}}
	if !cfg.ReplayNeedsPIIEvidence() {
		t.Fatal("decision opt-in must consume PII")
	}
	cfg.Decisions[0].Plugins[0].Configuration = MustStructuredPayload(map[string]interface{}{"enabled": true, "capture_personal_data": true})
	if cfg.ReplayNeedsPIIEvidence() {
		t.Fatal("explicit capture must remove unused PII demand")
	}
}
