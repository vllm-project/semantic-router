package config

import (
	"sort"
	"strings"
	"testing"
)

// builtInSignalsYAML reads every built-in signal Vela 2.0 answers, each from a decision.
func builtInSignalsYAML(global string) string {
	return `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  modelCards: [{name: general, modality: omni}]
  signals:
    domains: [{name: math, description: Math requests., mmlu_categories: [math]}]
    jailbreak: [{name: attack, threshold: 0.75}]
    pii: [{name: personal, threshold: 0.01}]
    fact_check: [{name: needs_fact_check, description: Needs checking.}]
    user_feedbacks: [{name: wrong_answer, description: Wrong.}]
    modality: [{name: DIFFUSION, description: Images.}]
    safety: [{name: harmful, labels: [safe, unsafe], unsafe_labels: [unsafe], threshold: 0.46}]
  decisions:
    - name: reads-all
      priority: 100
      rules:
        operator: OR
        conditions:
          - {type: domain, name: math}
          - {type: jailbreak, name: attack}
          - {type: pii, name: personal}
          - {type: fact_check, name: needs_fact_check}
          - {type: user_feedback, name: wrong_answer}
          - {type: modality, name: DIFFUSION}
          - {type: safety, name: harmful}
      modelRefs: [{model: general}]
global:
  model_catalog:
    modules:
      modality_detector: {enabled: true, method: classifier, confidence_threshold: 0.51}
` + global
}

func deploymentsByArtifact(t *testing.T, raw string) map[string][]string {
	t.Helper()
	cfg, err := ParseYAMLBytes([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	byArtifact := map[string][]string{}
	for name, deployment := range ModelRuntimeDeploymentsInUse(cfg) {
		byArtifact[deployment.Artifact+" "+deployment.Device+" "+deployment.Profile] = append(byArtifact[deployment.Artifact+" "+deployment.Device+" "+deployment.Profile], name)
	}
	for _, names := range byArtifact {
		sort.Strings(names)
	}
	return byArtifact
}

func TestBuiltInSignalsShareOneVela2DeploymentOnCPU(t *testing.T) {
	got := deploymentsByArtifact(t, builtInSignalsYAML(""))
	if len(got) != 1 || len(got["vllm-sr/Vela-2.0-0.3B cpu max_speed"]) != 1 || got["vllm-sr/Vela-2.0-0.3B cpu max_speed"][0] != "@Vela-2.0-0.3B" {
		t.Fatalf("every built-in signal must run on one Vela 2.0 0.3B CPU deployment under max_speed, got %v", got)
	}
}

func TestBuiltInSignalsOnAnAcceleratorShareOneExactDeployment(t *testing.T) {
	global := `      prompt_guard: {use_cpu: false}
      classifier:
        domain: {use_cpu: false}
        pii: {use_cpu: false}
      hallucination_mitigation:
        fact_check: {use_cpu: false}
      feedback_detector: {use_cpu: false}
      safety:
        safety: {use_cpu: false}
`
	raw := strings.Replace(builtInSignalsYAML(global), "      modality_detector: {enabled: true, method: classifier, confidence_threshold: 0.51}\n",
		"      modality_detector: {enabled: true, method: classifier, confidence_threshold: 0.51, classifier: {use_cpu: false}}\n", 1)
	got := deploymentsByArtifact(t, raw)
	if len(got) != 1 || len(got["vllm-sr/Vela-2.0-0.3B auto exact"]) != 1 || got["vllm-sr/Vela-2.0-0.3B auto exact"][0] != "@Vela-2.0-0.3B/auto" {
		t.Fatalf("an accelerator default runs one exact Vela 2.0 deployment, got %v", got)
	}
}

func TestVela1SystemModelsRestoreOneDeploymentPerSpecialist(t *testing.T) {
	vela1 := Vela1SystemModels()
	got := deploymentsByArtifact(t, strings.Replace(builtInSignalsYAML(`    system:
      safety: `+vela1.Safety+`
      prompt_guard: `+vela1.PromptGuard+`
      domain_classifier: `+vela1.DomainClassifier+`
      pii_classifier: `+vela1.PIIClassifier+`
      fact_check_classifier: `+vela1.FactCheckClassifier+`
      feedback_detector: `+vela1.FeedbackDetector+`
`), "method: classifier, confidence_threshold: 0.51}", "method: classifier, confidence_threshold: 0.51, classifier: {model_path: models/Vela-1.0-Encoder-307M-Modality, use_cpu: true}}", 1))
	for artifact := range got {
		if !strings.HasPrefix(artifact, "vllm-sr/Vela-1.0-") || !strings.HasSuffix(artifact, " cpu exact") {
			t.Fatalf("the restored defaults run only Vela 1.0 specialists on exact CPU deployments, got %v", got)
		}
	}
	if len(got) != 7 {
		t.Fatalf("seven specialists (Domain, Guard, PII, FactCheck, Feedback, Modality, Safety), got %v", got)
	}
}
