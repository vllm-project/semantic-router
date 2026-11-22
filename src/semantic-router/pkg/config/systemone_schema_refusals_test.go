package config

import (
	"strings"
	"testing"
)

// A systemone deployment names a schema the Router speaks, deviations that
// apply to it and a max_concurrency of at least 1, and declares only what its
// schema carries; each refusal names what is wrong and what to do instead.
func TestASystemOneDeploymentsSchemaRefusals(t *testing.T) {
	zero := 0
	for _, tc := range []struct {
		name   string
		change func(*ModelDeployment)
		want   string
	}{
		{"an unknown schema", func(d *ModelDeployment) { d.Schema = "systemone/9.9" }, `schema "systemone/9.9" is not supported; the schemas are `},
		{"an unknown deviation", func(d *ModelDeployment) { d.Deviations = []string{"no_such_deviation"} }, `deviation "no_such_deviation" is not supported; the deviations are state_text`},
		{"a deviation twice", func(d *ModelDeployment) { d.Deviations = []string{"state_text", "state_text"} }, `deviations lists "state_text" twice`},
		{"a deviation of another schema", func(d *ModelDeployment) { d.Schema, d.Deviations = "vllm-sr/5.0", []string{"state_text"} }, `deviation "state_text" does not apply to schema vllm-sr/5.0`},
		{"no concurrency", func(d *ModelDeployment) { d.MaxConcurrency = &zero }, "max_concurrency must be at least 1"},
		{"set questions", func(d *ModelDeployment) { d.Capabilities.QuestionTypes = []string{"noul", "set"} }, "capabilities.question_types lists set, which schema systemone/0.2 has no questions of; the Router asks a set question as one noul per label"},
		{"span questions", func(d *ModelDeployment) { d.Capabilities.QuestionTypes = []string{"span"} }, "capabilities.question_types lists span, which schema systemone/0.2 has no questions of; the Router asks a set question as one noul per label, and span rules need a vllm-sr/5.0 deployment"},
		{"track questions", func(d *ModelDeployment) { d.Capabilities.QuestionTypes = []string{"track"} }, "capabilities.question_types lists track, which schema systemone/0.2 has no questions of; a track question reads a video"},
		{"presets", func(d *ModelDeployment) { d.Capabilities.Presets = []string{"pii"} }, "capabilities.presets: schema systemone/0.2 carries no presets; the Router asks its presets with their instructions and criteria"},
		{"video", func(d *ModelDeployment) { d.Capabilities.Modalities = []string{"text", "video"} }, "capabilities.modalities lists video, which schema systemone/0.2 does not carry"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deployment := ModelDeployment{
				Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra",
				Auth: &DeploymentAuth{APIKeyEnv: "KEY"}, Capabilities: &DeploymentCapabilities{QuestionTypes: []string{"choice", "noul"}},
			}
			if err := deployment.validateSystemOne(); err != nil {
				t.Fatalf("the standard deployment: %v", err)
			}
			tc.change(&deployment)
			if err := deployment.validateSystemOne(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
	runtime := ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Schema: "systemone/0.2"}
	if err := runtime.validate(); err == nil || !strings.Contains(err.Error(), "schema, deviations and max_concurrency describe a remote System One API; they apply to systemone deployments") {
		t.Fatalf("a schema on a model_runtime deployment: %v", err)
	}
}

// A rule or an answer check that needs what a systemone/0.2 API does not
// carry fails at load, naming the deployment, its schema and what to do.
func TestWhatTheStandardDoesNotCarryFailsAtLoad(t *testing.T) {
	standard := func(yaml string) string {
		return strings.Replace(yaml, "schema: vllm-sr/5.0", "schema: systemone/0.2", 1)
	}
	noul := capabilityQuestions[SystemOneQuestionNoul].question
	for _, tc := range []struct{ name, yaml, want string }{
		{
			"a rule that reads its whole input", standard(capabilityConfig("remote", noul, "coverage: full", "", "noul")),
			`reads its whole input (coverage: full) and resolves to deployment "remote", whose schema systemone/0.2 has no input_coverage to prove it; set coverage: truncate, or bind it to a deployment that reports coverage (schema vllm-sr/5.0)`,
		},
		{
			"a rule that reads a video", standard(capabilityConfig("remote", noul, "input: {video: true}", "", "noul")),
			`input.video resolves to deployment "remote", whose schema systemone/0.2 carries no video`,
		},
		{
			"an answer check", answerCheckConfig("systemone/0.2"),
			`decision 'answer', algorithm.escalate.accept.systemone: deployment "remote" speaks systemone/0.2, which has no input_coverage to prove an answer check read the whole answer; bind the check to a deployment whose schema reports coverage (vllm-sr/5.0)`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseYAMLBytes([]byte(tc.yaml)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
	if _, err := ParseYAMLBytes([]byte(answerCheckConfig("vllm-sr/5.0"))); err != nil {
		t.Fatalf("an answer check on vllm-sr/5.0: %v", err)
	}
}

// answerCheckConfig is a decision whose answers a systemone deployment of
// schema checks.
func answerCheckConfig(schema string) string {
	return `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: strong
      backend_refs: [{name: strong, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
recipes:
  - name: default
    decisions:
      - name: answer
        priority: 100
        rules: {}
        modelRefs: [{model: general}, {model: strong}]
        algorithm:
          type: escalate
          escalate:
            accept: {systemone: {deployment: remote, gte: 0.5}}
global:
  model_catalog:
    deployments:
      remote:
        provider: systemone
        schema: ` + schema + `
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: CAPABILITY_TEST_KEY}
        capabilities: {question_types: [noul]}
`
}
