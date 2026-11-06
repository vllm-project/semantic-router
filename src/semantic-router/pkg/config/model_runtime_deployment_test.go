package config

import "testing"

func taskBindingConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"kai":        {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "cpu"},
		"vela-guard": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B", Device: "cpu"},
		"vela-pii": {
			Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast", Device: "cpu",
			Input: ModelInputBudget{MaxTokens: 32768, Overflow: "window"},
		},
	}
	cfg.ModelBindings = map[string]ModelBinding{
		"domain":         {Deployment: "kai"},
		"prompt_guard":   {Deployment: "vela-guard"},
		"pii_classifier": {Deployment: "vela-pii"},
	}
	cfg.Categories = []Category{{Name: "math", Labels: []string{"math"}}}
	cfg.Decisions = []Decision{{Name: "math", Rules: RuleNode{Type: SignalTypeDomain, Name: "math"}}}
	return cfg
}

func TestModelRuntimeServesTaskBindings(t *testing.T) {
	plan, err := CompileModelBindings(taskBindingConfig())
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := plan.Lookup(DefaultRecipeName, "pii_classifier")
	if !ok || !spec.Deployment.IsModelRuntime() || spec.Deployment.Input.Overflow != "window" || spec.Binding.Contract != DecisionTaskContract {
		t.Fatalf("pii binding = %+v", spec)
	}
	for _, rule := range taskBindingConfig().ApplicationRules() {
		if rule.Rule.Deployment != "kai" {
			t.Fatalf("the domain binding answers %s:%s, got %q", rule.Type, rule.Rule.Name, rule.Rule.Deployment)
		}
	}
	cfg := taskBindingConfig()
	cfg.ModelBindings["hallucination_explainer"] = ModelBinding{Deployment: "vela-guard", Contract: "text_pair_distribution.v1", Adapter: "modernbert"}
	if _, err := CompileModelBindings(cfg); err == nil {
		t.Fatal("the retired NLI explainer must not bind a model_runtime deployment")
	}
}

func TestModelRuntimeDeploymentsInUseFollowActiveConsumers(t *testing.T) {
	cfg := taskBindingConfig()
	used := ModelRuntimeDeploymentsInUse(cfg)
	if _, ok := used["kai"]; !ok || len(used) != 1 {
		t.Fatalf("only the deployment bound to the domain rules is active, in use = %v", used)
	}
	if used["kai"].HasEngineArgs() || used["kai"].Device == "" {
		t.Fatalf("defaults are applied: %+v", used["kai"])
	}

	moveTestRoutingToUnmappedRecipe(cfg)
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("an unreachable routing profile starts nothing, in use = %v", used)
	}
}

func TestModelRuntimeDeploymentsInUseStartTheModelGroundingAsks(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		DefaultDecisionDeployment: {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast", Device: "cpu"},
		"halu":                    {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B", Device: "cpu"},
	}
	grounding := &FusionGroundingConfig{Enabled: true, Reference: FusionGroundingReferencePanel}
	cfg.Decisions = []Decision{{Name: "fuse", Algorithm: &AlgorithmConfig{
		Type: DecisionAlgorithmFusion, Fusion: &FusionAlgorithmConfig{Grounding: grounding},
	}}}
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 1 || used[DefaultDecisionDeployment].Artifact == "" {
		t.Fatalf("grounding asks the decision model, in use = %v", used)
	}
	cfg.GlobalModelBindings = map[string]ModelBinding{SignalTypeHallucination: {Deployment: "halu"}}
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 1 || used["halu"].Artifact == "" {
		t.Fatalf("grounding asks the deployment the hallucination binding names, in use = %v", used)
	}
	grounding.Enabled = false
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("a fusion that does not ground asks no model, in use = %v", used)
	}
}
