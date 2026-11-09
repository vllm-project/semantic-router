package config

import (
	"strings"
	"testing"
)

func TestDefaultDecisionModelRejectsATaskInput(t *testing.T) {
	documents := []string{
		withDecisionQuestion(selectedDecisionYAML("vllm-sr/d3", "")),
		withOtherModel(withDecisionSelector(selectedDecisionYAML("vllm-sr/d3", ""), "")),
	}
	for _, raw := range documents {
		if _, err := ParseYAMLBytes([]byte(raw)); err != nil {
			t.Fatal(err)
		}
		for _, input := range []string{
			"input: {max_tokens: 32768, overflow: reject}",
			"input: {max_tokens: 512, overflow: truncate}",
		} {
			broken := strings.Replace(raw, "device: cpu}", "device: cpu, "+input+"}", 1)
			_, err := ParseYAMLBytes([]byte(broken))
			if err == nil || !strings.Contains(err.Error(), "never truncate") {
				t.Fatalf("%s: %v", input, err)
			}
		}
		window := strings.Replace(raw, "device: cpu}", "device: cpu, input: {max_tokens: 32768, overflow: window}}", 1)
		if _, err := ParseYAMLBytes([]byte(window)); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSystemOnePublicationRejectsATaskInput(t *testing.T) {
	for _, input := range []string{
		"input: {max_tokens: 32768, overflow: reject}",
		"input: {max_tokens: 512, overflow: truncate}",
	} {
		for _, raw := range []string{
			systemOnePublicationYAML("vllm-sr/d3", "", input),
			systemOnePublicationYAML("judge", systemOneAliasModel, input),
		} {
			_, err := ParseYAMLBytes([]byte(raw))
			if err == nil || !strings.Contains(err.Error(), "never truncate") {
				t.Fatalf("%s: %v\n%s", input, err, raw)
			}
		}
	}
	for _, raw := range []string{
		systemOnePublicationYAML("vllm-sr/d3", "", "input: {max_tokens: 32768, overflow: window}"),
		systemOnePublicationYAML("judge", systemOneAliasModel, ""),
	} {
		if _, err := ParseYAMLBytes([]byte(raw)); err != nil {
			t.Fatal(err)
		}
	}
}

const systemOneAliasModel = `
    - name: judge
      api_format: systemone
      deployment: d3`

func systemOnePublicationYAML(models, extraModels, input string) string {
	body := "provider: model_runtime\n        artifact: vllm-sr/d3\n        device: cpu"
	if input != "" {
		body += "\n        " + input
	}
	return `version: v0.3
listeners:
  - name: public
    port: 8899
    systemone:
      models: [` + models + `]
providers:
  defaults: {model: m}
  models:
    - name: m
      backend_refs: [{endpoint: "127.0.0.1:8000", protocol: http}]
` + extraModels + `
routing:
  modelCards: [{name: m}]
  decisions:
    - name: d
      priority: 1
      rules: {operator: AND, conditions: [{type: keyword, name: k}]}
      modelRefs: [{model: m}]
  signals:
    keywords:
      - name: k
        operator: OR
        keywords: [hello]
global:
  model_catalog:
    deployments:
      d3:
        ` + body + "\n"
}

func TestTaskBindingKeepsATaskInputUntilPublishedOnSystemOne(t *testing.T) {
	for _, input := range []string{
		"input: {max_tokens: 512, overflow: truncate}",
		"input: {max_tokens: 32768, overflow: reject}",
	} {
		if _, err := ParseYAMLBytes([]byte(taskBindingYAML(input, false))); err != nil {
			t.Fatalf("task binding %s: %v", input, err)
		}
		_, err := ParseYAMLBytes([]byte(taskBindingYAML(input, true)))
		if err == nil || !strings.Contains(err.Error(), "never truncate") {
			t.Fatalf("system one %s: %v", input, err)
		}
	}
}

func taskBindingYAML(input string, publish bool) string {
	listener := ""
	if publish {
		listener = "\n    systemone: {models: [encoder-public]}"
	}
	return `version: v0.3
listeners:
  - name: public
    port: 8899` + listener + `
providers:
  defaults: {model: m}
  models:
    - name: m
      backend_refs: [{endpoint: "127.0.0.1:8000", protocol: http}]
routing:
  modelCards: [{name: m}]
  signals:
    domains: [{name: math, description: Math requests., mmlu_categories: [math]}]
  decisions:
    - name: d
      priority: 1
      rules: {operator: AND, conditions: [{type: domain, name: math}]}
      modelRefs: [{model: m}]
global:
  model_catalog:
    deployments:
      encoder:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        public_name: encoder-public
        ` + input + `
    bindings:
      domain_classifier: {deployment: encoder, contract: label_distribution.v1}
`
}
