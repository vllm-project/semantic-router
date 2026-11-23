package config

import (
	"strings"
	"testing"
)

const outputContractAlgorithmConfig = `version: v0.5
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      provider_model_id: model-a
      backend_refs: [{name: fake-a, endpoint: 127.0.0.1:8000, protocol: http, provider: vllm}]
    - name: model-b
      provider_model_id: model-b
      backend_refs: [{name: fake-b, endpoint: 127.0.0.1:8000, protocol: http, provider: vllm}]
recipes:
  - name: default
    decisions:
      - name: contract
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: model-a}, {model: model-b}]
        output_contract_spec: SPEC
ALGORITHM`

var outputContractSpecs = map[string]string{
	"choice":              `{type: choice, choice_set: {values: [A, B, C, D]}}`,
	"reference_selection": `{type: reference_selection, postprocess: [{type: dereference_selected_reference}]}`,
	"structured_json":     `{type: structured_json, json_schema: {schema_ref: terminal_action_v1}, extract: {mode: json_object}}`,
}

// TestOutputContractSpecLoadsOnlyWhereAnAlgorithmAppliesIt holds the config
// load to the algorithms that shape their final answer by
// output_contract_spec: Fusion, ReMoM and Flow apply every type but Flow's
// reference_selection, whose final lists worker outputs without numbers, and
// every other decision serves a model's answer as it is.
func TestOutputContractSpecLoadsOnlyWhereAnAlgorithmAppliesIt(t *testing.T) {
	algorithms := map[string]string{
		"none":   "",
		"remom":  "        algorithm: {type: remom, remom: {breadth_schedule: [2]}}",
		"fusion": "        algorithm: {type: fusion, fusion: {model: model-a, analysis_models: [model-a, model-b]}}",
		"workflows": "        algorithm:\n          type: workflows\n          workflows:\n            mode: static\n" +
			"            roles: [{name: worker, models: [model-a]}]\n            final: {model: model-b}",
		"escalate": "        algorithm: {type: escalate, escalate: {order: size, accept: {confidence: {method: avg_logprob, gte: 0.6}}}}",
	}
	cases := []struct {
		algorithm string
		spec      string
		wantErr   string
	}{
		{algorithm: "remom", spec: "choice"},
		{algorithm: "remom", spec: "reference_selection"},
		{algorithm: "remom", spec: "structured_json"},
		{algorithm: "fusion", spec: "choice"},
		{algorithm: "fusion", spec: "reference_selection"},
		{algorithm: "fusion", spec: "structured_json"},
		{algorithm: "workflows", spec: "choice"},
		{algorithm: "workflows", spec: "structured_json"},
		{
			algorithm: "workflows", spec: "reference_selection",
			wantErr: "decision 'contract': output_contract_spec type reference_selection is not supported by algorithm workflows",
		},
		{
			algorithm: "escalate", spec: "choice",
			wantErr: "decision 'contract': output_contract_spec requires algorithm fusion, remom or workflows, which apply it to the final answer",
		},
		{
			algorithm: "none", spec: "choice",
			wantErr: "decision 'contract': output_contract_spec requires algorithm fusion, remom or workflows, which apply it to the final answer",
		},
	}
	for _, tc := range cases {
		t.Run(tc.algorithm+"/"+tc.spec, func(t *testing.T) {
			yaml := strings.NewReplacer("SPEC", outputContractSpecs[tc.spec], "ALGORITHM", algorithms[tc.algorithm]).Replace(outputContractAlgorithmConfig)
			_, err := ParseYAMLBytes([]byte(yaml))
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("load failed: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("load error = %v, want one containing %q", err, tc.wantErr)
			}
		})
	}
}
