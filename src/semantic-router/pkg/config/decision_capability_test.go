package config

import (
	"fmt"
	"strings"
	"testing"
)

// capabilityConfig is a configuration whose one systemone rule, probe, asks
// question of deployment ("" asks the decision model, primary: vela3-fast).
// remote is a systemone deployment that declares remoteTypes; kai serves
// Decision 2.0 Kai, which answers choice, noul and score.
func capabilityConfig(deployment, question, extra, label, remoteTypes string) string {
	if deployment != "" {
		deployment = "deployment: " + deployment
	}
	return fmt.Sprintf(`version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      systemone:
        - name: probe
          %s
          question: %s
          %s
    decisions:
      - name: probe
        priority: 100
        rules:
          operator: AND
          on_unknown: no_match
          conditions: [{type: systemone, name: probe%s}]
        modelRefs: [{model: general}]
global:
  model_catalog:
    deployments:
      remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: CAPABILITY_TEST_KEY}
        capabilities: {question_types: [%s]}
      kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
`, deployment, question, extra, label, remoteTypes)
}

// The questions of each type a rule asks, with what the rule and its
// condition need besides.
var capabilityQuestions = map[string]struct{ question, extra, label string }{
	SystemOneQuestionChoice: {`{type: choice, instructions: "Which team should handle the request?", choices: [{key: billing}, {key: engineering}]}`, "", ", label: billing"},
	SystemOneQuestionNoul:   {`{type: noul, instructions: "Does the request need a tool?"}`, "", ""},
	SystemOneQuestionScore:  {`{type: score, instructions: "How difficult is the request?", levels: [Trivial, Moderate, Hard]}`, "predicate: {gte: 1.5}", ""},
	SystemOneQuestionSet:    {`{type: set, instructions: "Which topics does the request raise?", labels: [{key: billing}, {key: outage}]}`, "", ", label: billing"},
	SystemOneQuestionSpan:   {`{type: span, instructions: "Find account numbers.", labels: [{key: account}]}`, "", ", label: account"},
}

func capabilityRule(questionType, deployment, remoteTypes string) string {
	q := capabilityQuestions[questionType]
	return capabilityConfig(deployment, q.question, q.extra, q.label, remoteTypes)
}

// Every question type a decision-model deployment cannot answer fails at
// load, naming the question, the deployment and the type: the built-in
// model's catalog entry tells for a model_runtime deployment, and the
// declared capabilities for a systemone one.
func TestQuestionTheDeploymentCannotAnswerFailsAtLoad(t *testing.T) {
	const path = "recipes[default].signals.systemone[probe]: "
	for _, test := range []struct {
		name, questionType, deployment, remoteTypes, want string
	}{
		{
			"score on vela3-fast, the default decision model", SystemOneQuestionScore, "", "choice, noul, score",
			path + `question_type_unsupported: resolves to deployment "primary", which answers choice, noul, span questions and cannot answer score questions; bind it to a deployment that does (rule deployment)`,
		},
		{
			"span on Decision 2.0", SystemOneQuestionSpan, "kai", "choice, noul, score",
			path + `span_required: resolves to deployment "kai", which cannot answer span questions`,
		},
		{
			"set on a deployment that answers neither set nor noul", SystemOneQuestionSet, "remote", "choice, score",
			path + `question_type_unsupported: resolves to deployment "remote", which answers choice, score questions and cannot answer set questions`,
		},
		{
			"choice on a deployment that answers noul only", SystemOneQuestionChoice, "remote", "noul",
			path + `question_type_unsupported: resolves to deployment "remote", which answers noul questions and cannot answer choice questions`,
		},
		{
			"noul on a deployment that answers choice only", SystemOneQuestionNoul, "remote", "choice",
			path + `question_type_unsupported: resolves to deployment "remote", which answers choice questions and cannot answer noul questions`,
		},
		{
			"score on a deployment that declares no score", SystemOneQuestionScore, "remote", "choice, noul",
			path + `question_type_unsupported: resolves to deployment "remote", which answers choice, noul questions and cannot answer score questions`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(capabilityRule(test.questionType, test.deployment, test.remoteTypes)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// A question its deployment answers loads: natively, or a set question as
// one noul per label on a model that answers no set question.
func TestQuestionTheDeploymentAnswersLoads(t *testing.T) {
	for _, test := range []struct{ name, questionType, deployment, remoteTypes string }{
		{"score on Decision 2.0", SystemOneQuestionScore, "kai", "choice, noul"},
		{"span on vela3-fast", SystemOneQuestionSpan, "", "choice, noul"},
		{"set on vela3-fast, asked as nouls", SystemOneQuestionSet, "", "choice, noul"},
		{"set on a remote deployment that answers noul", SystemOneQuestionSet, "remote", "choice, noul, score"},
		{"choice on a remote deployment that declares it", SystemOneQuestionChoice, "remote", "choice"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := ParseYAMLBytes([]byte(capabilityRule(test.questionType, test.deployment, test.remoteTypes))); err != nil {
				t.Fatalf("a question its deployment answers must load: %v", err)
			}
		})
	}
}

// A judge, the systemone selector and an answer check are held to the same
// rule as a rule: their question's type must be one the deployment answers.
func TestJudgeTheDeploymentCannotAnswerFailsAtLoad(t *testing.T) {
	score := &SystemOneQuestion{Type: SystemOneQuestionScore, Instructions: "How relevant is the candidate?", Levels: []string{"no", "partly", "yes"}}
	err := validateMatch(matchConfig(), "m", &MatchConfig{Judge: &MatchJudge{Question: score}}, MatchCompress)
	want := `m.judge: deployment "primary" serves a model that answers choice, noul, span questions, so it does not answer a score question`
	if err == nil || err.Error() != want {
		t.Fatalf("error = %v, want %q", err, want)
	}
	if err := validateMatch(matchConfig(), "m", &MatchConfig{Judge: &MatchJudge{Question: score, Deployment: "lite"}}, MatchCompress); err != nil {
		t.Fatalf("vela3-lite answers score questions: %v", err)
	}
}

// spanHeadConfig asks a span question with head of deployment: "" asks the
// decision model, primary: vela3-fast (router head); v2small is Vela 2.0
// 0.3B (router head), v2 Vela 2.0 0.8B (router and broad heads), lite
// vela3-lite (router head), and remote a systemone deployment that declares
// span answers but no heads.
func spanHeadConfig(deployment, head string) string {
	if deployment != "" {
		deployment = "deployment: " + deployment
	}
	return fmt.Sprintf(`version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      systemone:
        - name: probe
          %s
          question: {type: span, instructions: "Find account numbers.", labels: [{key: account}], head: %s}
    decisions:
      - name: probe
        priority: 100
        rules:
          operator: AND
          on_unknown: no_match
          conditions: [{type: systemone, name: probe, label: account}]
        modelRefs: [{model: general}]
global:
  model_catalog:
    deployments:
      v2small: {provider: model_runtime, artifact: vllm-sr/Vela-2.0-0.3B, device: cpu}
      v2: {provider: model_runtime, artifact: vllm-sr/Vela-2.0-0.8B, device: cpu}
      lite: {provider: model_runtime, artifact: vllm-sr/vela3-lite}
      remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: CAPABILITY_TEST_KEY}
        capabilities: {question_types: [choice, noul, span]}
`, deployment, head)
}

// A span question whose head its deployment's model lacks fails at load: the
// catalog records each built-in model's span heads.
func TestSpanHeadTheModelLacksFailsAtLoad(t *testing.T) {
	const path = "recipes[default].signals.systemone[probe]: span_head_unsupported: "
	for _, test := range []struct{ name, deployment, want string }{
		{"broad on vela3-fast, the default decision model", "", path + `resolves to deployment "primary", whose model has span heads router and no broad head; drop head, or bind it to a deployment whose model has one (rule deployment)`},
		{"broad on vela3-lite", "lite", path + `resolves to deployment "lite", whose model has span heads router and no broad head`},
		{"broad on Vela 2.0 0.3B, which ships no broad head", "v2small", path + `resolves to deployment "v2small", whose model has span heads router and no broad head`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(spanHeadConfig(test.deployment, SystemOneSpanHeadBroad)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// A head the model has loads, and so does any head of a remote deployment,
// whose heads are unknown at load: its API refuses one it lacks.
func TestSpanHeadTheModelHasLoads(t *testing.T) {
	for _, test := range []struct{ name, deployment, head string }{
		{"broad on Vela 2.0 0.8B", "v2", SystemOneSpanHeadBroad},
		{"router on vela3-fast", "", SystemOneSpanHeadRouter},
		{"broad on a remote deployment that declares no heads", "remote", SystemOneSpanHeadBroad},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := ParseYAMLBytes([]byte(spanHeadConfig(test.deployment, test.head))); err != nil {
				t.Fatalf("a head the model has must load: %v", err)
			}
		})
	}
}
