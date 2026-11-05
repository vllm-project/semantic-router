package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// systemOneDecisionYAML declares two systemone deployments: hosted-d3 behind
// Bearer auth and gateway-d3 behind a gateway's own header and path prefix.
// One question names gateway-d3; the rest, and a selector, ask the decision
// model, hosted-d3.
const systemOneDecisionYAML = `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: other-model
      backend_refs: [{name: other, endpoint: "127.0.0.1:8001", protocol: http}]
routing:
  signals:
    systemone:
      - name: support_team
        deployment: gateway-d3
        question: {type: choice, instructions: "Which team should handle this request?", choices: [{key: billing}, {key: engineering}]}
      - name: needs_tools
        question: {type: noul, instructions: "Does the request need a tool?"}
  decisions:
    - name: engineering
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions: [{type: systemone, name: support_team, label: engineering}]
      modelRefs: [{model: general}]
    - name: choose
      priority: 100
      rules:
        operator: AND
        on_unknown: no_match
        conditions: [{type: systemone, name: needs_tools}]
      modelRefs: [{model: general}, {model: other-model}]
      algorithm:
        type: systemone
        systemone: {instructions: "Which model should answer?"}
global:
  model_catalog:
    deployments:
      hosted-d3:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: HOSTED_D3_TEST_KEY}
        timeout_ms: 3000
      gateway-d3:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        question_types: [choice, noul, score]
        auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_file: /run/secrets/gateway-key}
    system:
      decision_model: {deployment: hosted-d3}
`

func TestSystemOneDeploymentsAnswerDecisionQuestions(t *testing.T) {
	cfg := mustParseDecisionModel(t, systemOneDecisionYAML)
	if !cfg.IsSystemOneDeployment("hosted-d3") || !cfg.IsSystemOneDeployment("gateway-d3") || !cfg.RemoteDecisionModel() {
		t.Fatal("both deployments are systemone and hosted-d3 is the decision model")
	}
	want := map[string]string{"support_team": "gateway-d3", "needs_tools": "hosted-d3"}
	for _, rule := range cfg.SystemOneRules {
		if got := cfg.SystemOneRuleDeployment(rule); got != want[rule.Name] {
			t.Fatalf("rule %s asks %q, want %q", rule.Name, got, want[rule.Name])
		}
	}
	if got := cfg.SystemOneSelectorDeployment(*cfg.Decisions[1].Algorithm.SystemOne); got != "hosted-d3" {
		t.Fatalf("selector asks %q", got)
	}
	gateway := cfg.ModelDeployments["gateway-d3"]
	if gateway.Auth.HeaderName() != "Ocp-Apim-Subscription-Key" || gateway.Auth.HeaderValue("k") != "k" ||
		cfg.ModelDeployments["hosted-d3"].Auth.HeaderValue("k") != "Bearer k" || cfg.ModelDeployments["hosted-d3"].Auth.HeaderName() != "Authorization" {
		t.Fatalf("auth: gateway %+v", gateway.Auth)
	}
	if cfg.ModelDeployments["hosted-d3"].CallTimeout().Milliseconds() != 3000 {
		t.Fatal("timeout_ms bounds the deployment's calls")
	}
	for name := range ModelRuntimeDeploymentsInUse(cfg) {
		t.Fatalf("a systemone deployment starts no model runtime, got %q", name)
	}
	for _, model := range []string{"hosted-d3", "gateway-d3"} {
		if _, routable := cfg.ModelConfig[model]; routable {
			t.Fatalf("%s joined the request backend pool", model)
		}
	}
	local := mustParseDecisionModel(t, withDecisionQuestion(builtInSignalsYAML("")))
	if got, want := cfg.DecisionModelSpec().Model, local.DecisionModelSpec().Model; got != want || got == "" {
		t.Fatalf("modules keep their built-in model under a systemone decision model: %q, want %q", got, want)
	}

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	canonical := gateway
	canonical.Capabilities, canonical.QuestionTypes = &DeploymentCapabilities{QuestionTypes: gateway.QuestionTypes}, nil
	if roundTrip.DecisionModel != "hosted-d3" || !reflect.DeepEqual(roundTrip.ModelDeployments["gateway-d3"], canonical) {
		t.Fatalf("round trip changed the deployment:\n%+v\n%+v", roundTrip.ModelDeployments["gateway-d3"], canonical)
	}
}

func TestSystemOneDeploymentsTakeOnlyTheirOwnFields(t *testing.T) {
	for name, tc := range map[string]struct{ old, replacement, want string }{
		"an engine argument": {
			"        timeout_ms: 3000\n", "        timeout_ms: 3000\n        max_num_seqs: 4\n",
			"a systemone deployment takes a connection block",
		},
		"a device": {
			"        timeout_ms: 3000\n", "        timeout_ms: 3000\n        device: cpu\n",
			"a systemone deployment takes a connection block",
		},
		"replicas": {
			"        timeout_ms: 3000\n", "        timeout_ms: 3000\n        replicas: [{device: cpu}]\n",
			"a systemone deployment takes a connection block",
		},
		"an input budget": {
			"        timeout_ms: 3000\n", "        timeout_ms: 3000\n        input: {max_tokens: 512, overflow: window}\n",
			"takes no input budget",
		},
		"an endpoint without a scheme": {
			"https://systemone.example.com/v1/systemone", "systemone.example.com/v1/systemone",
			"endpoint must be the http:// or https:// URL",
		},
		"credentials in the endpoint": {
			"https://systemone.example.com/v1/systemone", "https://user:secret@systemone.example.com/v1/systemone",
			"endpoint must not carry credentials",
		},
		"no model": {
			"        model: vllm-sr/d3-ultra\n", "",
			"model must name the model the API answers with",
		},
		"an inline key": {
			"auth: {api_key_env: HOSTED_D3_TEST_KEY}", "auth: {header: Authorization}",
			"auth takes exactly one of api_key_env and api_key_file",
		},
		"two key sources": {
			"auth: {api_key_env: HOSTED_D3_TEST_KEY}", "auth: {api_key_env: HOSTED_D3_TEST_KEY, api_key_file: /run/secrets/key}",
			"auth takes exactly one of api_key_env and api_key_file",
		},
		"a relative key file": {
			"api_key_file: /run/secrets/gateway-key", "api_key_file: secrets/gateway-key",
			"auth.api_key_file must be an absolute path",
		},
		"a prefix with spaces": {
			`prefix: ""`, `prefix: "Bearer token"`,
			"auth.prefix must be one word",
		},
		"an unknown question type": {
			"question_types: [choice, noul, score]", "question_types: [choice, box]",
			`question_types lists "box"`,
		},
		"a timeout past the decision limit": {
			"timeout_ms: 3000", "timeout_ms: 90000",
			"timeout_ms must be within",
		},
		"systemone fields on a model_runtime deployment": {
			"    system:\n", "      local-d3: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu, timeout_ms: 100}\n    system:\n",
			"apply to an attached endpoint, not to a worker the Router starts",
		},
		"an embedding binding": {
			"    system:\n", "    bindings:\n      embedding: hosted-d3\n    system:\n",
			"embeddings and reranking need a model_runtime, openai_embeddings or rerank deployment",
		},
	} {
		t.Run(name, func(t *testing.T) {
			raw := strings.Replace(systemOneDecisionYAML, tc.old, tc.replacement, 1)
			if raw == systemOneDecisionYAML {
				t.Fatal("the case changed nothing")
			}
			_, err := ParseYAMLBytes([]byte(raw))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestSystemOneDeploymentKeyIsReadWhenACallNeedsIt(t *testing.T) {
	auth := &DeploymentAuth{APIKeyEnv: "SYSTEMONE_KEY_TEST"}
	t.Setenv("SYSTEMONE_KEY_TEST", "")
	if _, err := auth.Key(); err == nil || !strings.Contains(err.Error(), "SYSTEMONE_KEY_TEST is not set") {
		t.Fatalf("an unset variable: %v", err)
	}
	t.Setenv("SYSTEMONE_KEY_TEST", " rotated-key\n")
	if key, err := auth.Key(); err != nil || key != "rotated-key" {
		t.Fatalf("key %q, %v", key, err)
	}
	var none *DeploymentAuth
	if key, err := none.Key(); err != nil || key != "" || none.HeaderName() != "Authorization" {
		t.Fatalf("no auth sends no key: %q %v", key, err)
	}
}
