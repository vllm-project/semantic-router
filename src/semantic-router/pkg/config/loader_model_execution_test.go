package config

import (
	"strings"
	"testing"
)

func TestRejectRemovedModelExecutionFieldsNamesEveryRetiredField(t *testing.T) {
	cases := []struct {
		name string
		yaml string
		want []string
	}{
		{
			name: "removed providers and numerics",
			yaml: `
global:
  model_catalog:
    deployments:
      guard:
        provider: candle
      domain:
        provider: model_runtime
        precision: fp16
        custom_ops_profile: ck
`,
			want: []string{
				"global.model_catalog.deployments.guard.provider: candle",
				"global.model_catalog.deployments.domain.precision",
				"global.model_catalog.deployments.domain.custom_ops_profile",
			},
		},
		{
			name: "module backend selectors and the NLI explainer",
			yaml: `
global:
  model_catalog:
    system:
      hallucination_explainer: models/nli
    modules:
      prompt_guard:
        variant: mmbert32k
      classifier:
        pii:
          use_mmbert_32k: true
      hallucination_mitigation:
        nli_model: {}
        detector:
          backend: candle
          enable_nli_filtering: true
`,
			want: []string{
				"global.model_catalog.modules.prompt_guard.variant",
				"global.model_catalog.modules.classifier.pii.use_mmbert_32k",
				"global.model_catalog.modules.hallucination_mitigation.nli_model",
				"global.model_catalog.modules.hallucination_mitigation.detector.enable_nli_filtering",
				"global.model_catalog.modules.hallucination_mitigation.detector.backend: candle",
				"global.model_catalog.system.hallucination_explainer",
			},
		},
		{
			name: "NLI routing fields in routing and recipes",
			yaml: `
routing:
  signals:
    hallucination:
      - name: grounded
        use_nli: true
  decisions:
    - name: fused
      algorithm:
        type: fusion
        fusion:
          grounding:
            nli_contradiction_penalty: 1.0
recipes:
  - name: private
    routing:
      decisions:
        - name: checked
          plugins:
            - type: hallucination
              configuration:
                use_nli: true
`,
			want: []string{
				"routing.signals.hallucination[0].use_nli",
				"routing.decisions[0].algorithm.fusion.grounding.nli_contradiction_penalty",
				"recipes[0].routing.decisions[0].plugins[0].configuration.use_nli",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := parseRawConfigMap([]byte(tc.yaml))
			if err != nil {
				t.Fatal(err)
			}
			err = rejectRemovedModelExecutionFields(raw)
			if err == nil {
				t.Fatal("expected the retired fields to be rejected")
			}
			for _, field := range append(tc.want, "vllm-sr config migrate") {
				if !strings.Contains(err.Error(), field) {
					t.Errorf("error does not name %q: %v", field, err)
				}
			}
		})
	}
}

func TestRejectRemovedModelExecutionFieldsAcceptsCanonicalFields(t *testing.T) {
	raw, err := parseRawConfigMap([]byte(`
routing:
  decisions:
    - name: fused
      algorithm:
        type: fusion
        fusion:
          grounding:
            contradiction_penalty: 1.0
global:
  model_catalog:
    deployments:
      domain:
        provider: model_runtime
        profile: max_speed
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := rejectRemovedModelExecutionFields(raw); err != nil {
		t.Fatalf("canonical fields rejected: %v", err)
	}
}
