package config

import "testing"

func taskBindingConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"vela-domain": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
		"vela-pii": {
			Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-PII", Device: "cpu",
			Input: ModelInputBudget{MaxTokens: 32768, Overflow: "window"},
		},
		"vela-factcheck": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-FactCheck"},
	}
	cfg.ModelBindings = map[string]ModelBinding{
		"domain_classifier":     {Deployment: "vela-domain", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
		"pii_classifier":        {Deployment: "vela-pii", Contract: RemoteClassifierContractTokenSpans, Adapter: "modernbert"},
		"fact_check_classifier": {Deployment: "vela-factcheck", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
	}
	cfg.Decisions = []Decision{{Name: "math", Rules: RuleNode{Type: SignalTypeDomain, Name: "math"}}}
	return cfg
}

func TestModelRuntimeServesTaskBindings(t *testing.T) {
	plan, err := CompileModelBindings(taskBindingConfig())
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := plan.Lookup(DefaultRecipeName, "pii_classifier")
	if !ok || !spec.Deployment.IsModelRuntime() || spec.Deployment.Input.Overflow != "window" {
		t.Fatalf("pii binding = %+v", spec)
	}
	cfg := taskBindingConfig()
	cfg.ModelBindings["hallucination_explainer"] = ModelBinding{Deployment: "vela-domain", Contract: "text_pair_distribution.v1", Adapter: "modernbert"}
	if _, err := CompileModelBindings(cfg); err == nil {
		t.Fatal("the retired NLI explainer must not bind a model_runtime deployment")
	}
}

func TestModelRuntimeDeploymentsInUseFollowActiveConsumers(t *testing.T) {
	cfg := taskBindingConfig()
	used := ModelRuntimeDeploymentsInUse(cfg)
	if _, ok := used["vela-domain"]; !ok || len(used) != 1 {
		t.Fatalf("only the domain consumer is active, in use = %v", used)
	}
	if used["vela-domain"].Profile != "exact" || used["vela-domain"].Device == "" {
		t.Fatalf("defaults are applied: %+v", used["vela-domain"])
	}
	cfg.Decisions = append(cfg.Decisions, Decision{Name: "private", Rules: RuleNode{Type: SignalTypePII, Name: "any"}})
	if _, ok := ModelRuntimeDeploymentsInUse(cfg)["vela-pii"]; !ok {
		t.Fatal("a decision that reads the PII signal activates its deployment")
	}

	moveTestRoutingToUnmappedRecipe(cfg)
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("an unreachable routing profile starts nothing, in use = %v", used)
	}
}
