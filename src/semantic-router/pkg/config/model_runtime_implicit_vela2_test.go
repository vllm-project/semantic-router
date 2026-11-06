package config

import (
	"sort"
	"testing"
)

// builtInSignalsYAML reads every built-in signal Vela 2.0 answers without
// spans, each from a decision; catalog is the body of global.model_catalog.
func builtInSignalsYAML(catalog string) string {
	doc := `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  modelCards: [{name: general, modality: omni}]
  signals:
    domains: [{name: math, description: Math requests., labels: [math]}]
    jailbreak: [{name: attack, predicate: {gte: 0.75}}]
    fact_check: [{name: needs_fact_check, description: Needs checking.}]
    user_feedbacks: [{name: wrong_answer, labels: [WRONG_ANSWER], description: Wrong.}]
    modality: [{name: DIFFUSION, labels: [image], description: Images., predicate: {gte: 0.51}}]
    safety: [{name: harmful, predicate: {gte: 0.46}}]
  decisions:
    - name: reads-all
      priority: 100
      rules:
        operator: OR
        conditions:
          - {type: domain, name: math}
          - {type: jailbreak, name: attack}
          - {type: fact_check, name: needs_fact_check}
          - {type: user_feedback, name: wrong_answer}
          - {type: modality, name: DIFFUSION}
          - {type: safety, name: harmful}
      modelRefs: [{model: general}]
`
	if catalog == "" {
		return doc
	}
	return doc + "global:\n  model_catalog:\n" + catalog
}

func deploymentsByArtifact(t *testing.T, raw string) map[string][]string {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	byArtifact := map[string][]string{}
	for name, deployment := range ModelRuntimeDeploymentsInUse(cfg) {
		if deployment.HasEngineArgs() {
			t.Fatalf("implicit deployment %s must keep the runtime's default engine arguments: %+v", name, deployment)
		}
		byArtifact[deployment.Artifact+" "+deployment.Device] = append(byArtifact[deployment.Artifact+" "+deployment.Device], name)
	}
	for _, names := range byArtifact {
		sort.Strings(names)
	}
	return byArtifact
}

func TestBuiltInSignalsShareOneDefaultDeploymentOnCPU(t *testing.T) {
	got := deploymentsByArtifact(t, builtInSignalsYAML(""))
	if len(got) != 1 || len(got["vllm-sr/vela3-fast cpu"]) != 1 || got["vllm-sr/vela3-fast cpu"][0] != DefaultDecisionDeployment {
		t.Fatalf("every built-in signal must run on one vela3-fast CPU deployment, got %v", got)
	}
}

func TestBuiltInSignalsOnAnAcceleratorShareOneDeployment(t *testing.T) {
	got := deploymentsByArtifact(t, builtInSignalsYAML(`    deployments:
      primary: {provider: model_runtime, artifact: vllm-sr/Vela-2.0-0.3B, device: auto}
`))
	if len(got) != 1 || len(got["vllm-sr/Vela-2.0-0.3B auto"]) != 1 || got["vllm-sr/Vela-2.0-0.3B auto"][0] != DefaultDecisionDeployment {
		t.Fatalf("an accelerator default runs one Vela 2.0 deployment, got %v", got)
	}
}
