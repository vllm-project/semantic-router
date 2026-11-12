package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestReplayWithoutPersonalDataActivatesPIIEvidence(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults:
    model: local
  models:
    - name: local
      backend_refs:
        - endpoint: local:8000
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
          input: {history: false}
          ignore_labels: [DATE_TIME]
    decisions:
      - name: fast
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: local
global:
  services:
    router_replay: {}
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.8B
    system:
      decision_model:
        deployment: primary
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        capture_personal_data: false
`))
	if err != nil {
		t.Fatal(err)
	}
	policy := cfg.EffectiveRouterReplayConfig(nil)
	if policy == nil || policy.CapturesPersonalData() || !cfg.ReplayNeedsPIIEvidence() {
		t.Fatal("a PII rule must be prepared to suppress personal replay content")
	}
}

func replayPlugin(configuration map[string]interface{}) DecisionPlugin {
	return DecisionPlugin{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(configuration)}
}

func TestReplayGlobalDefaultsAndExplicitDecisionOverrides(t *testing.T) {
	yes, no := true, false
	global := []DecisionPlugin{replayPlugin(map[string]interface{}{
		"enabled": true, "capture_request_body": false, "capture_personal_data": false, "max_tool_trace_steps": 0,
	})}
	inherited := layeredRouting(global, Decision{Name: "plain"}).EffectiveRouterReplayConfig(nil)
	if inherited.CaptureRequestBody || inherited.CapturesPersonalData() || inherited.MaxToolTraceSteps != 0 || !inherited.CaptureResponseBody {
		t.Fatalf("incorrect global defaults: %+v", inherited)
	}
	cfg := layeredRouting(global, Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{
		"capture_request_body": true, "capture_personal_data": true, "capture_response_body": false, "max_body_bytes": 0,
	})}})
	got := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("d"))
	if !got.CaptureRequestBody || !got.CapturesPersonalData() || got.CaptureResponseBody || got.MaxBodyBytes != 0 || got.MaxToolTraceSteps != 0 {
		t.Fatalf("incorrect overlay: %+v", got)
	}
	for _, enabled := range []bool{false, true} {
		for _, override := range []*bool{nil, &yes, &no} {
			var layer []DecisionPlugin
			if enabled {
				layer = []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true})}
			}
			want, decision := enabled, Decision{Name: "d"}
			if override != nil {
				want = *override
				decision.Plugins = []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": *override})}
			}
			cfg := layeredRouting(layer, decision)
			if (cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("d")) != nil) != want {
				t.Fatalf("global %v override %v", enabled, override)
			}
		}
	}
}

func TestReplayCaptureDefaultsRoundTripWithoutPIIRules(t *testing.T) {
	raw := []byte(`version: v0.5
global:
  services:
    router_replay: {}
  plugins:
    - type: router_replay
      configuration:
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

// A recipe's data_policy is a removed form, in a configuration and in a
// routing fragment: the line names the router_replay plugin migrate writes.
func TestRemovedDataPolicyIsARemovedFormInCanonicalAndFragments(t *testing.T) {
	const line = "data_policy: removed_field: a recipe's data_policy is removed"
	for _, field := range []string{"replay: false", "replay_personal_data: false"} {
		for _, prefix := range []string{"recipes:\n  - name: default\n    ", "recipes:\n  - name: default\n  - name: other\n    "} {
			_, err := ParseYAMLBytes([]byte("version: v0.5\n" + prefix + "data_policy: {" + field + "}\n"))
			if err == nil || !strings.Contains(err.Error(), line) {
				t.Fatalf("want the removed data_policy: %v", err)
			}
		}
		_, err := ParseRoutingYAMLBytes([]byte("recipes:\n  - name: default\n    data_policy: {" + field + "}\n"))
		if err == nil || !strings.Contains(err.Error(), line) {
			t.Fatalf("fragment accepted data_policy: %v", err)
		}
	}
}

func TestReplayPIIDemandTracksPossibleEffectivePolicies(t *testing.T) {
	withheld := []DecisionPlugin{replayPlugin(map[string]interface{}{"capture_personal_data": false})}
	routing := func(global []DecisionPlugin, decision Decision) *RouterConfig {
		cfg := layeredRouting(global, decision)
		cfg.Recipes[0].Profile.Signals.PIIRules = []PIIRule{{Name: "personal"}}
		cfg.mirrorDefaultRecipe()
		return cfg
	}
	cfg := routing([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "capture_personal_data": false})}, Decision{Name: "plain"})
	if !cfg.ReplayNeedsPIIEvidence() || !cfg.UsesSignalTypeInRouting(SignalTypePII) {
		t.Fatal("global policy must consume existing PII rules")
	}
	if routing(withheld, Decision{Name: "plain"}).ReplayNeedsPIIEvidence() {
		t.Fatal("disabled replay must not consume PII")
	}
	optedIn := Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true})}}
	if !routing(withheld, optedIn).ReplayNeedsPIIEvidence() {
		t.Fatal("decision opt-in must consume PII")
	}
	captured := Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "capture_personal_data": true})}}
	if routing(withheld, captured).ReplayNeedsPIIEvidence() {
		t.Fatal("explicit capture must remove unused PII demand")
	}
}

func TestReplayToolResultCaptureDefaultsAndDecisionOverrides(t *testing.T) {
	no := false
	cfg := layeredRouting([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true})}, Decision{Name: "plain"})
	if cfg.EffectiveRouterReplayConfig(nil).CapturesToolResults() {
		t.Fatal("tool results must stay redacted by default")
	}
	global := []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "capture_tool_results": true})}
	cfg = layeredRouting(global, Decision{Name: "off", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"capture_tool_results": false})}})
	if !cfg.EffectiveRouterReplayConfig(nil).CapturesToolResults() {
		t.Fatal("global capture_tool_results was not applied")
	}
	if cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("off")).CapturesToolResults() {
		t.Fatal("a decision could not turn tool-result capture off")
	}
	withheld := &RouterReplayPluginConfig{Enabled: true, CaptureToolResults: true, CapturePersonalData: &no}
	if withheld.CapturesToolResults() {
		t.Fatal("tool results were captured while personal data is withheld")
	}
}

func TestReplayToolResultCaptureRequiresPersonalDataAtLoad(t *testing.T) {
	const decisions = `
recipes:
  - name: default
    decisions:
      - name: agent_loop
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: local
        plugins:
          - type: router_replay
            configuration:
`
	document := func(global, plugin string) []byte {
		return []byte(`version: v0.5
providers:
  defaults:
    model: local
  models:
    - name: local
      backend_refs:
        - endpoint: local:8000
` + decisions + "              " + plugin + `
global:
  services:
    router_replay: {}
  plugins:
    - type: router_replay
      configuration:
        enabled: true
` + global)
	}
	for _, test := range []struct {
		name, global, plugin, refused string
	}{
		{"allowed", "        capture_tool_results: true\n", "enabled: true", ""},
		{"global conflict", "        capture_tool_results: true\n        capture_personal_data: false\n", "capture_tool_results: false", "global.plugins[router_replay]"},
		{"decision inherits withheld personal data", "        capture_personal_data: false\n", "capture_tool_results: true", `decision "agent_loop"`},
		{"decision withholds inherited capture", "        capture_tool_results: true\n", "capture_personal_data: false", `decision "agent_loop"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(document(test.global, test.plugin))
			if test.refused == "" {
				if err != nil {
					t.Fatalf("valid tool-result capture refused: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.refused) || !strings.Contains(err.Error(), "capture_tool_results") {
				t.Fatalf("want refusal naming %s, got %v", test.refused, err)
			}
		})
	}
}
