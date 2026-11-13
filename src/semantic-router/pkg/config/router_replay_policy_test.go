package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// replayModeDocument is a recipe with a pii rule that no decision reads and a
// global router_replay plugin of the given configuration lines.
func replayModeDocument(configuration string) []byte {
	return []byte(`version: v0.5
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
` + configuration)
}

// Only verified capture reads a prompt for personal data, so only it makes
// the recipe's pii rules a dependency of every request. Unset is none.
func TestReplayOnlyVerifiedCaptureActivatesPIIEvidence(t *testing.T) {
	for _, test := range []struct {
		name, configuration, mode string
		warned                    bool
	}{
		{"unset", "", PromptCaptureNone, false},
		{"none", "        prompt_capture: none\n", PromptCaptureNone, false},
		{"raw", "        prompt_capture: raw\n", PromptCaptureRaw, false},
		{"verified", "        prompt_capture: verified\n", PromptCaptureVerified, false},
		{"deprecated false", "        capture_personal_data: false\n", PromptCaptureVerified, true},
		{"deprecated true", "        capture_personal_data: true\n", PromptCaptureRaw, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes(replayModeDocument(test.configuration))
			if err != nil {
				t.Fatal(err)
			}
			policy := cfg.EffectiveRouterReplayConfig(nil)
			if policy == nil || policy.PromptCaptureMode() != test.mode {
				t.Fatalf("mode = %q, want %q", policy.PromptCaptureMode(), test.mode)
			}
			verified := test.mode == PromptCaptureVerified
			if cfg.ReplayNeedsPIIEvidence() != verified || cfg.UsesSignalTypeInReachableRouting(SignalTypePII) != verified {
				t.Fatalf("pii evidence needed = %v, pii read = %v, want %v",
					cfg.ReplayNeedsPIIEvidence(), cfg.UsesSignalTypeInReachableRouting(SignalTypePII), verified)
			}
			if warned := len(cfg.FormWarnings) > 0; warned != test.warned {
				t.Fatalf("form warnings %+v, want warned=%v", cfg.FormWarnings, test.warned)
			}
		})
	}
}

// A deprecated capture_personal_data loads as the prompt_capture it meant, in
// its own layer, with a warning that names the rewrite; it is never carried
// past the load.
func TestCapturePersonalDataLoadsAsPromptCaptureWithAWarning(t *testing.T) {
	for _, value := range []bool{false, true} {
		mode := personalDataPromptCapture(value)
		flag := map[bool]string{false: "false", true: "true"}[value]
		cfg, err := ParseYAMLBytes(replayModeDocument("        capture_personal_data: " + flag + "\n"))
		if err != nil {
			t.Fatal(err)
		}
		want := ConfigWarning{
			Code:    CodeRenamedField,
			Field:   "global.plugins[0].configuration.capture_personal_data",
			Message: "capture_personal_data: " + flag + " is deprecated; run `vllm-sr config migrate` (prompt_capture: " + mode + ")",
		}
		if len(cfg.FormWarnings) != 1 || cfg.FormWarnings[0] != want {
			t.Fatalf("warnings %+v, want %+v", cfg.FormWarnings, want)
		}
		policy := cfg.EffectiveRouterReplayConfig(nil)
		if policy.PromptCapture != mode || policy.CapturePersonalData != nil {
			t.Fatalf("loaded as prompt_capture %q, capture_personal_data %v; want %q alone", policy.PromptCapture, policy.CapturePersonalData, mode)
		}
		encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(encoded), "capture_personal_data") || !strings.Contains(string(encoded), "prompt_capture: "+mode) {
			t.Fatalf("the canonical configuration keeps the deprecated form:\n%s", encoded)
		}
	}
}

func TestCapturePersonalDataBesidePromptCaptureMustAgree(t *testing.T) {
	cfg, err := ParseYAMLBytes(replayModeDocument("        prompt_capture: verified\n        capture_personal_data: false\n"))
	if err != nil {
		t.Fatalf("agreeing forms refused: %v", err)
	}
	if policy := cfg.EffectiveRouterReplayConfig(nil); !policy.VerifiesPrompt() || policy.CapturePersonalData != nil || len(cfg.FormWarnings) != 1 {
		t.Fatalf("agreeing forms: %+v, warnings %+v", policy, cfg.FormWarnings)
	}
	_, err = ParseYAMLBytes(replayModeDocument("        prompt_capture: raw\n        capture_personal_data: false\n"))
	if err == nil || !strings.Contains(err.Error(), "capture_personal_data means prompt_capture: verified, but prompt_capture is raw") {
		t.Fatalf("want the disagreement refused, got %v", err)
	}
}

// Each layer's deprecated form is rewritten before the layers merge, so the
// most specific layer's mode wins whichever spelling each layer used.
func TestDeprecatedCaptureIsRewrittenPerLayer(t *testing.T) {
	document := func(global, decision string) []byte {
		return []byte(`version: v0.5
providers:
  defaults:
    model: local
  models:
    - name: local
      backend_refs:
        - endpoint: local:8000
recipes:
  - name: default
    decisions:
      - name: d
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: local
        plugins:
          - type: router_replay
            configuration:
              ` + decision + `
global:
  services:
    router_replay: {}
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        ` + global + "\n")
	}
	for _, test := range []struct{ global, decision, inherited, own string }{
		{"capture_personal_data: false", "prompt_capture: raw", PromptCaptureVerified, PromptCaptureRaw},
		{"prompt_capture: verified", "capture_personal_data: true", PromptCaptureVerified, PromptCaptureRaw},
		{"capture_personal_data: true", "prompt_capture: none", PromptCaptureRaw, PromptCaptureNone},
	} {
		cfg, err := ParseYAMLBytes(document(test.global, test.decision))
		if err != nil {
			t.Fatal(err)
		}
		if got := cfg.EffectiveRouterReplayConfig(nil).PromptCaptureMode(); got != test.inherited {
			t.Fatalf("%s: default mode %q, want %q", test.global, got, test.inherited)
		}
		if got := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("d")).PromptCaptureMode(); got != test.own {
			t.Fatalf("%s under %s: decision mode %q, want %q", test.decision, test.global, got, test.own)
		}
	}
}

func TestPromptCaptureRefusesAnUnknownMode(t *testing.T) {
	_, err := ParseYAMLBytes(replayModeDocument("        prompt_capture: full\n"))
	if err == nil || !strings.Contains(err.Error(), `prompt_capture must be none, raw or verified, not "full"`) {
		t.Fatalf("want the unknown mode refused, got %v", err)
	}
}

func TestPromptCaptureModeResolution(t *testing.T) {
	yes, no := true, false
	for _, test := range []struct {
		name   string
		policy *RouterReplayPluginConfig
		mode   string
	}{
		{"no policy", nil, PromptCaptureNone},
		{"unset", &RouterReplayPluginConfig{}, PromptCaptureNone},
		{"deprecated false", &RouterReplayPluginConfig{CapturePersonalData: &no}, PromptCaptureVerified},
		{"deprecated true", &RouterReplayPluginConfig{CapturePersonalData: &yes}, PromptCaptureRaw},
		{"prompt_capture decides", &RouterReplayPluginConfig{PromptCapture: PromptCaptureNone, CapturePersonalData: &yes}, PromptCaptureNone},
		{"an unknown mode keeps nothing", &RouterReplayPluginConfig{PromptCapture: "full", CapturePersonalData: &yes}, PromptCaptureNone},
	} {
		if got := test.policy.PromptCaptureMode(); got != test.mode {
			t.Errorf("%s: mode %q, want %q", test.name, got, test.mode)
		}
	}
	for _, test := range []struct {
		mode                            string
		requestText, responseText, tool bool
	}{
		{PromptCaptureNone, false, false, false},
		{PromptCaptureRaw, true, true, true},
		{PromptCaptureVerified, true, true, false},
	} {
		policy := &RouterReplayPluginConfig{PromptCapture: test.mode, CaptureRequestBody: true, CaptureResponseBody: true, CaptureToolResults: true}
		if policy.CapturesRequestText() != test.requestText || policy.CapturesResponseText() != test.responseText || policy.CapturesToolResults() != test.tool {
			t.Errorf("%s: request text %v, response text %v, tool results %v", test.mode,
				policy.CapturesRequestText(), policy.CapturesResponseText(), policy.CapturesToolResults())
		}
		policy.CaptureRequestBody, policy.CaptureResponseBody = false, false
		if policy.CapturesRequestText() || policy.CapturesResponseText() {
			t.Errorf("%s: text captured with both capture switches off", test.mode)
		}
	}
}

func replayPlugin(configuration map[string]interface{}) DecisionPlugin {
	return DecisionPlugin{Type: DecisionPluginRouterReplay, Configuration: MustStructuredPayload(configuration)}
}

func TestReplayGlobalDefaultsAndExplicitDecisionOverrides(t *testing.T) {
	yes, no := true, false
	global := []DecisionPlugin{replayPlugin(map[string]interface{}{
		"enabled": true, "capture_request_body": false, "prompt_capture": PromptCaptureVerified, "max_tool_trace_steps": 0,
	})}
	inherited := layeredRouting(global, Decision{Name: "plain"}).EffectiveRouterReplayConfig(nil)
	if inherited.CaptureRequestBody || !inherited.VerifiesPrompt() || inherited.MaxToolTraceSteps != 0 || !inherited.CaptureResponseBody {
		t.Fatalf("incorrect global defaults: %+v", inherited)
	}
	cfg := layeredRouting(global, Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{
		"capture_request_body": true, "prompt_capture": PromptCaptureRaw, "capture_response_body": false, "max_body_bytes": 0,
	})}})
	got := cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("d"))
	if !got.CaptureRequestBody || !got.CapturesRawContent() || got.CaptureResponseBody || got.MaxBodyBytes != 0 || got.MaxToolTraceSteps != 0 {
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
        prompt_capture: verified
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
	if got == nil || !got.VerifiesPrompt() || got.CaptureRequestBody || got.CaptureResponseBody || got.MaxRecords != 77 || got.MaxBodyBytes != 0 ||
		got.MaxToolTraceBytes == nil || *got.MaxToolTraceBytes != 0 || got.MaxToolTraceSteps != 0 {
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
	verified := []DecisionPlugin{replayPlugin(map[string]interface{}{"prompt_capture": PromptCaptureVerified})}
	routing := func(global []DecisionPlugin, decision Decision) *RouterConfig {
		cfg := layeredRouting(global, decision)
		cfg.Recipes[0].Profile.Signals.PIIRules = []PIIRule{{Name: "personal"}}
		cfg.mirrorDefaultRecipe()
		return cfg
	}
	cfg := routing([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "prompt_capture": PromptCaptureVerified})}, Decision{Name: "plain"})
	if !cfg.ReplayNeedsPIIEvidence() || !cfg.UsesSignalTypeInRouting(SignalTypePII) {
		t.Fatal("global verified capture must consume existing PII rules")
	}
	deprecated := routing([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "capture_personal_data": false})}, Decision{Name: "plain"})
	if !deprecated.ReplayNeedsPIIEvidence() {
		t.Fatal("capture_personal_data: false, assembled in code, must still verify")
	}
	if routing(verified, Decision{Name: "plain"}).ReplayNeedsPIIEvidence() {
		t.Fatal("disabled replay must not consume PII")
	}
	optedIn := Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true})}}
	if !routing(verified, optedIn).ReplayNeedsPIIEvidence() {
		t.Fatal("decision opt-in must consume PII")
	}
	for _, mode := range []string{PromptCaptureRaw, PromptCaptureNone} {
		unverified := Decision{Name: "d", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "prompt_capture": mode})}}
		cfg := routing(verified, unverified)
		if cfg.ReplayNeedsPIIEvidence() || cfg.UsesSignalTypeInRouting(SignalTypePII) {
			t.Fatalf("prompt_capture: %s must remove unused PII demand", mode)
		}
	}
	if routing([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true})}, Decision{Name: "plain"}).ReplayNeedsPIIEvidence() {
		t.Fatal("the default capture, none, must not consume PII")
	}
}

func TestReplayToolResultCaptureDefaultsAndDecisionOverrides(t *testing.T) {
	no := false
	cfg := layeredRouting([]DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "prompt_capture": PromptCaptureRaw})}, Decision{Name: "plain"})
	if cfg.EffectiveRouterReplayConfig(nil).CapturesToolResults() {
		t.Fatal("tool results must stay redacted by default")
	}
	global := []DecisionPlugin{replayPlugin(map[string]interface{}{"enabled": true, "prompt_capture": PromptCaptureRaw, "capture_tool_results": true})}
	cfg = layeredRouting(global, Decision{Name: "off", Plugins: []DecisionPlugin{replayPlugin(map[string]interface{}{"capture_tool_results": false})}})
	if !cfg.EffectiveRouterReplayConfig(nil).CapturesToolResults() {
		t.Fatal("global capture_tool_results was not applied")
	}
	if cfg.EffectiveRouterReplayConfig(cfg.GetDecisionByName("off")).CapturesToolResults() {
		t.Fatal("a decision could not turn tool-result capture off")
	}
	for _, test := range []struct {
		name   string
		policy RouterReplayPluginConfig
	}{
		{"none", RouterReplayPluginConfig{Enabled: true, CaptureToolResults: true}},
		{"verified", RouterReplayPluginConfig{Enabled: true, CaptureToolResults: true, PromptCapture: PromptCaptureVerified}},
		{"capture_personal_data: false", RouterReplayPluginConfig{Enabled: true, CaptureToolResults: true, CapturePersonalData: &no}},
	} {
		if test.policy.CapturesToolResults() {
			t.Fatalf("tool results were captured under %s", test.name)
		}
	}
}

func TestReplayToolResultCaptureRequiresRawCaptureAtLoad(t *testing.T) {
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
		{"allowed", "        prompt_capture: raw\n        capture_tool_results: true\n", "enabled: true", ""},
		{"stored nowhere under none", "        capture_tool_results: true\n", "enabled: true", ""},
		{"global conflict", "        capture_tool_results: true\n        prompt_capture: verified\n", "capture_tool_results: false", "global.plugins[router_replay]"},
		{"deprecated global conflict", "        capture_tool_results: true\n        capture_personal_data: false\n", "capture_tool_results: false", "global.plugins[router_replay]"},
		{"decision inherits verified capture", "        prompt_capture: verified\n", "capture_tool_results: true", `decision "agent_loop"`},
		{"decision verifies inherited capture", "        prompt_capture: raw\n        capture_tool_results: true\n", "prompt_capture: verified", `decision "agent_loop"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(document(test.global, test.plugin))
			if test.refused == "" {
				if err != nil {
					t.Fatalf("valid tool-result capture refused: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.refused) || !strings.Contains(err.Error(), "capture_tool_results: true requires prompt_capture: raw") {
				t.Fatalf("want refusal naming %s, got %v", test.refused, err)
			}
		})
	}
}
