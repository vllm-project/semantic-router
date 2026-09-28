package config

import "testing"

func TestSafetyBindingsKeepIndependentHeadContractAndRecipeScope(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.SafetyRules = []SafetyRule{{Name: "risk", Threshold: .5, Hazard: &SafetyHazardRule{Labels: []string{"a", "b"}, Categories: []string{"b"}, Threshold: .8}}}
	cfg.ModelDeployments = map[string]ModelDeployment{"encoder": {Artifact: "model", Provider: "candle", Device: "cpu", Precision: "native", Input: ModelInputBudget{MaxTokens: 32768}}}
	cfg.ModelBindings = map[string]ModelBinding{
		"safety.risk":        {Deployment: "encoder", Adapter: "modernbert", Contract: RemoteClassifierContractLabelDistribution},
		"safety.risk.hazard": {Deployment: "encoder", Adapter: "modernbert", Contract: RemoteClassifierContractLabelScores, Head: "hazard-head"},
	}
	if _, err := CompileModelBindings(cfg); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"safety.risk.hazard", "safety.missing"} {
		original := cloneModelMap(cfg.ModelBindings)
		cfg.ModelBindings[name] = ModelBinding{Deployment: "encoder", Adapter: "modernbert", Contract: RemoteClassifierContractLabelDistribution}
		if _, err := CompileModelBindings(cfg); err == nil {
			t.Fatalf("accepted incompatible/missing safety consumer %q", name)
		}
		cfg.ModelBindings = original
	}
}

func TestExplicitClassificationBudgetIsResolvedByLoadedArtifact(t *testing.T) {
	cfg := testDeploymentConfig()
	for _, limit := range []int{0, 512, 32768} {
		d := cfg.ModelDeployments["shared-encoder"]
		d.Input.MaxTokens = limit
		cfg.ModelDeployments["shared-encoder"] = d
		plan, err := CompileModelBindings(cfg)
		if err != nil {
			t.Fatal(err)
		}
		spec, ok := plan.Lookup("support", "domain_classifier")
		if !ok || spec.Deployment.Input.MaxTokens != limit {
			t.Fatalf("budget was silently rewritten: %+v", spec)
		}
	}
}

func TestSafetyBindingValidationUsesEffectiveDeploymentWithoutMutatingDefaults(t *testing.T) {
	cfg := safetyTestConfig()
	cfg.SafetyModels.Safety.ModelID = ""
	cfg.SafetyModels.Safety.MaxSequenceLength = 512
	cfg.SafetyModels.Safety.Window = &SequenceHeadWindowConfig{Size: 1024, Overlap: 128}
	cfg.ModelDeployments = map[string]ModelDeployment{"head": {Provider: "candle", Artifact: "mounted", Input: ModelInputBudget{MaxTokens: 2048}}}
	cfg.ModelBindings = map[string]ModelBinding{"safety.unsafe": {Deployment: "head", Adapter: "modernbert", Contract: RemoteClassifierContractLabelDistribution}}
	if err := validateSafetySignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.SafetyModels.Safety.ModelID != "" || cfg.SafetyModels.Safety.MaxSequenceLength != 512 {
		t.Fatal("mutated canonical module")
	}
	deployment := cfg.ModelDeployments["head"]
	deployment.Input.MaxTokens = 512
	cfg.ModelDeployments["head"] = deployment
	if err := validateSafetySignalContracts(cfg); err == nil {
		t.Fatal("ignored actual bound budget")
	}
}

func TestSafetyHTTPBindingUsesBoundEndpointAndCannotScanLocalWindows(t *testing.T) {
	cfg := safetyTestConfig()
	cfg.SafetyModels.Safety.ModelID = ""
	cfg.ExternalModels = []ExternalModelConfig{{Name: "remote", ModelRole: ModelRoleClassification, ModelName: "safe", ModelEndpoint: ClassifierVLLMEndpoint{Address: "localhost", Port: 8000}}}
	cfg.ModelDeployments = map[string]ModelDeployment{"endpoint": {Provider: "http", ExternalModel: "remote"}}
	cfg.ModelBindings = map[string]ModelBinding{"safety.unsafe": {Deployment: "endpoint", Adapter: RemoteClassifierProtocolHTTPClassify, Contract: RemoteClassifierContractLabelDistribution}}
	if err := validateSafetySignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.ExternalModels[0].ModelRole = "embedding"
	if err := validateSafetySignalContracts(cfg); err == nil {
		t.Fatal("accepted non-classification endpoint")
	}
	cfg.ExternalModels[0].ModelRole = ModelRoleClassification
	cfg.SafetyModels.Safety.Window = &SequenceHeadWindowConfig{Size: 128}
	if err := validateSafetySignalContracts(cfg); err == nil {
		t.Fatal("accepted local tokenizer policy on HTTP")
	}
}

func TestSafetyBindingRejectsAmbiguousConsumerAndUnusedMapping(t *testing.T) {
	rules := []SafetyRule{{Name: "risk", Hazard: &SafetyHazardRule{}}, {Name: "risk.hazard"}}
	decl := ModelBinding{Adapter: "modernbert", Contract: RemoteClassifierContractLabelScores}
	if err := validateSafetyModelBinding(rules, "safety.risk.hazard", decl, ModelDeployment{Provider: "candle"}); err == nil {
		t.Fatal("ambiguous consumer accepted")
	}
	decl.Contract = RemoteClassifierContractLabelDistribution
	decl.MappingPath = "ignored.json"
	if err := validateSafetyModelBinding(rules, "safety.risk", decl, ModelDeployment{Provider: "candle"}); err == nil {
		t.Fatal("ignored label mapping accepted")
	}
}

func TestModelDeploymentROCmCustomOpsIsExplicitAndScoped(t *testing.T) {
	d := ModelDeployment{Provider: "ort", Artifact: "model", Device: "rocm:0", Precision: "native", CustomOpsProfile: "ck_flash_attention"}
	if err := d.WithDefaults().validate(&RouterConfig{}); err != nil {
		t.Fatal(err)
	}
	for _, device := range []string{"cpu", "migraphx:0", "rocm:-1"} {
		bad := d
		bad.Device = device
		if err := bad.WithDefaults().validate(&RouterConfig{}); err == nil {
			t.Fatalf("accepted CK on %q", device)
		}
	}
	d.CustomOpsProfile = "none"
	if d.WithDefaults().CustomOpsProfile != "" {
		t.Fatal("none did not normalize to no custom ops")
	}
}
