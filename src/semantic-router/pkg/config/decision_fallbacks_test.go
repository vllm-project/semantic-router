package config

import (
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func withDecisionFallbacks(list, deployments string) string {
	raw := strings.Replace(systemOneDecisionYAML, "        timeout_ms: 3000\n", "        timeout_ms: 3000\n        fallbacks: "+list+"\n", 1)
	raw = strings.Replace(raw, "api_key_file: /run/secrets/gateway-key", "api_key_env: TEST_FALLBACK_GATEWAY_KEY", 1)
	return strings.Replace(raw, "    system:\n", deployments+"    system:\n", 1)
}

func TestDecisionDeploymentFallbacksResolveInOrder(t *testing.T) {
	raw := withDecisionFallbacks("[warm, gateway-d3]", `      warm:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: cpu
        fallbacks: [gateway-d3, cold]
      cold: {provider: model_runtime, endpoint: "http://127.0.0.1:8003", model: cpu-model}
      unused: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
`)
	cfg := mustParseDecisionModel(t, raw)
	want := []string{"hosted-d3", "warm", "gateway-d3", "cold"}
	if got := cfg.DecisionDeploymentChain("hosted-d3"); !slices.Equal(got, want) {
		t.Fatalf("fallback chain = %v, want %v", got, want)
	}
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 2 || used["warm"].Device != "cpu" || used["cold"].Model != "cpu-model" {
		t.Fatalf("only the two local fallback deployments should start or attach: %+v", used)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	cfg.ModelDeployments["hosted-d3"].Fallbacks[0] = "unused"
	encoded, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got := roundTrip.DecisionDeploymentChain("hosted-d3"); !slices.Equal(got, want) {
		t.Fatalf("round-tripped fallback chain = %v, want %v", got, want)
	}
}

func TestDecisionDeploymentFallbacksRejectInvalidChains(t *testing.T) {
	for _, test := range []struct {
		name, list, deployments, want string
	}{
		{name: "unknown", list: "[missing]", want: `fallbacks[0]: deployment "missing" is not declared`},
		{name: "duplicate", list: "[gateway-d3, gateway-d3]", want: `fallbacks[1]: deployment "gateway-d3" is listed twice`},
		{name: "self cycle", list: "[hosted-d3]", want: `cyclic decision fallback to deployment "hosted-d3"`},
		{name: "indirect cycle", list: "[loop]", deployments: `      loop: {provider: model_runtime, endpoint: "http://127.0.0.1:8003", fallbacks: [hosted-d3]}
`, want: `deployments.loop.fallbacks[0]: cyclic decision fallback to deployment "hosted-d3"`},
		{name: "not a decision provider", list: "[chat]", deployments: `      chat: {provider: openai_chat, endpoint: "http://127.0.0.1:8003/v1", model: chat-model}
`, want: `fallbacks[0]: deployment "chat" must use provider model_runtime or systemone`},
		{name: "incompatible question", list: "[choice-only]", deployments: `      choice-only:
        provider: systemone
        endpoint: https://backup.example.com/v1/systemone
        model: choice-only
        capabilities: {question_types: [choice]}
`, want: `deployment "choice-only", which answers choice questions and cannot answer noul questions`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(withDecisionFallbacks(test.list, test.deployments)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}
