package config

import (
	"math"
	"testing"
)

func TestValidateClassifierSignalContracts(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      "phishing",
			Type:      "local",
			ModelPath: "models/phishing",
			Labels:    []string{"BENIGN", "PHISHING"},
			UseCPU:    true,
		}}},
	}}
	if err := validateClassifierSignalContracts(cfg); err != nil {
		t.Fatalf("validateClassifierSignalContracts() error = %v", err)
	}
}

func TestValidateClassifierSignalContractsRejectsUnknownLLM(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:         "risk",
			Type:         "llm",
			Model:        "missing",
			Labels:       []string{"SAFE", "RISKY"},
			Instructions: "Choose.",
		}}},
	}}
	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected missing external model error")
	}
}

func TestValidateClassifierSignalContractsRejectsWhitespaceName(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      " risk ",
			Type:      "local",
			ModelPath: "models/risk",
			Labels:    []string{"SAFE", "RISKY"},
		}}},
	}}
	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected surrounding whitespace validation error")
	}
}

func TestValidateClassifierSignalContractsRejectsAmbiguousLabels(t *testing.T) {
	for _, labels := range [][]string{
		{" SAFE ", "RISKY"},
		{"SAFE", "RISK:Y"},
	} {
		cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
				Name:      "risk",
				Type:      "local",
				ModelPath: "models/risk",
				Labels:    labels,
			}}},
		}}
		if err := validateClassifierSignalContracts(cfg); err == nil {
			t.Fatalf("expected classifier label validation error for %v", labels)
		}
	}
}

func TestValidateClassifierSignalContractsRejectsInvalidEndpoint(t *testing.T) {
	cfg := &RouterConfig{
		ExternalModels: []ExternalModelConfig{{
			Name:      "judge",
			ModelRole: ModelRoleClassification,
		}},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
				Name:         "risk",
				Type:         "llm",
				Model:        "judge",
				Labels:       []string{"SAFE", "RISKY"},
				Instructions: "Classify.",
			}}},
		},
	}
	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected invalid external classifier endpoint error")
	}
}

func TestValidateClassifierSignalContractsRejectsCaseCollidingNames(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{
			{
				Name:      "Risk",
				Type:      "local",
				ModelPath: "models/risk",
				Labels:    []string{"SAFE", "RISKY"},
			},
			{
				Name:      "risk",
				Type:      "local",
				ModelPath: "models/risk",
				Labels:    []string{"SAFE", "RISKY"},
			},
		}},
	}}
	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected case-colliding classifier name error")
	}
}

func TestLocalClassifierDecisionPredicateRequiresWinningConfidence(t *testing.T) {
	upperBound := 0.4
	err := validateClassifierDecisionLeaf(
		&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
				Name:   "risk",
				Type:   "local",
				Labels: []string{"SAFE", "RISKY"},
			}}},
		}},
		"risk-route",
		&RuleNode{
			Type:      SignalTypeClassifier,
			Name:      "risk",
			Label:     "RISKY",
			Predicate: &NumericPredicate{LTE: &upperBound},
		},
	)
	if err == nil {
		t.Fatal("expected local classifier lower-bound predicate error")
	}
}

func TestDecisionLeafRejectsOnErrorOutsideClassifier(t *testing.T) {
	err := validateDecisionLeafNode(
		&RouterConfig{},
		"metadata-route",
		&RuleNode{
			Type:      SignalTypeMetadata,
			Name:      "cohort",
			Predicate: &NumericPredicate{},
			OnError:   "match",
		},
	)
	if err == nil {
		t.Fatal("expected on_error ownership validation error")
	}
}

func TestPromptAlgorithmRejectsDuplicateModelNames(t *testing.T) {
	useReasoning := false
	err := validatePromptAlgorithmConfig(
		"prompt-route",
		[]ModelRef{
			{Model: "model-a", ModelReasoningControl: ModelReasoningControl{UseReasoning: &useReasoning}},
			{Model: "model-a", LoRAName: "adapter", ModelReasoningControl: ModelReasoningControl{UseReasoning: &useReasoning}},
		},
		&AlgorithmConfig{
			Prompt: &PromptSelectionConfig{Model: "router", Instructions: "Choose."},
		},
	)
	if err == nil {
		t.Fatal("expected duplicate prompt model validation error")
	}
}

func TestPromptAlgorithmRejectsEffectiveLoRAIdentityCollision(t *testing.T) {
	useReasoning := false
	err := validatePromptAlgorithmConfig(
		"prompt-route",
		[]ModelRef{
			{
				Model:    "base",
				LoRAName: "specialist",
				ModelReasoningControl: ModelReasoningControl{
					UseReasoning: &useReasoning,
				},
			},
			{
				Model: "specialist",
				ModelReasoningControl: ModelReasoningControl{
					UseReasoning: &useReasoning,
				},
			},
		},
		&AlgorithmConfig{Prompt: &PromptSelectionConfig{
			Model: "router", Instructions: "Choose.",
		}},
	)
	if err == nil {
		t.Fatal("expected effective prompt candidate identity collision")
	}
}

func TestValidateLocalClassifierReloadRequiresRestartForChanges(t *testing.T) {
	current := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      "risk",
			Type:      "local",
			ModelPath: "models/risk-v1",
			Labels:    []string{"SAFE", "RISKY"},
		}}},
	}}
	same := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      "renamed-risk",
			Type:      "local",
			ModelPath: "models/risk-v1",
			Labels:    []string{"SAFE", "RISKY"},
		}}},
	}}
	changed := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      "risk",
			Type:      "local",
			ModelPath: "models/risk-v2",
			Labels:    []string{"SAFE", "RISKY"},
		}}},
	}}

	if err := ValidateLocalClassifierReload(current, same); err != nil {
		t.Fatalf("same runtime contract rejected: %v", err)
	}
	if err := ValidateLocalClassifierReload(current, changed); err == nil {
		t.Fatal("expected restart-required error for local classifier change")
	}
}

func TestDecisionPredicateRejectsNonFiniteValues(t *testing.T) {
	nan := math.NaN()
	err := validateDecisionLeafPredicate(
		"nan-route",
		&RuleNode{
			Type:      SignalTypeStructure,
			Name:      "bytes",
			Predicate: &NumericPredicate{GTE: &nan},
		},
	)
	if err == nil {
		t.Fatal("expected non-finite predicate validation error")
	}
	if err := ValidateStructureRuleContract(StructureRule{
		Name: "bytes",
		Feature: StructureFeature{
			Type:   "count",
			Source: StructureSource{Type: "text_bytes"},
		},
		Predicate: &NumericPredicate{GTE: &nan},
	}); err == nil {
		t.Fatal("expected non-finite structure predicate validation error")
	}
	if err := ValidateConversationRuleContract(ConversationRule{
		Name: "images",
		Feature: ConversationFeature{
			Type:   "count",
			Source: ConversationSource{Type: "image_content"},
		},
		Predicate: &NumericPredicate{GTE: &nan},
	}); err == nil {
		t.Fatal("expected non-finite conversation predicate validation error")
	}
}

func TestNumericPredicateRejectsEmptyRanges(t *testing.T) {
	five := 5.0
	three := 3.0
	for _, predicate := range []*NumericPredicate{
		{GT: &five, LT: &three},
		{GTE: &five, LTE: &three},
		{GT: &five, LTE: &five},
		{GTE: &five, LT: &five},
	} {
		if err := validateNumericPredicateContract(predicate); err == nil {
			t.Fatalf("predicate %#v should define an empty range", predicate)
		}
	}
	if err := validateNumericPredicateContract(
		&NumericPredicate{GTE: &five, LTE: &five},
	); err != nil {
		t.Fatalf("inclusive point range rejected: %v", err)
	}
}

func TestDecisionAlgorithmRejectsUnknownType(t *testing.T) {
	err := validateDecisionAlgorithmConfig(
		"typo-route",
		nil,
		&AlgorithmConfig{Type: "typo"},
	)
	if err == nil {
		t.Fatal("expected unknown algorithm type error")
	}
}

func TestDecisionAlgorithmRejectsMixedCaseType(t *testing.T) {
	err := validateDecisionAlgorithmConfig(
		"prompt-route",
		nil,
		&AlgorithmConfig{Type: "Prompt"},
	)
	if err == nil {
		t.Fatal("expected mixed-case algorithm type error")
	}
}

func TestDecisionPluginRejectsUnknownTypeAndFields(t *testing.T) {
	payload := MustStructuredPayload(map[string]interface{}{"enabled": true})
	if err := validateDecisionPluginPayload(
		"route",
		0,
		DecisionPlugin{Type: "unknown", Configuration: payload},
	); err == nil {
		t.Fatal("expected unknown plugin type error")
	}
	payload = MustStructuredPayload(map[string]interface{}{
		"enabled": true,
		"typo":    true,
	})
	if err := validateDecisionPluginPayload(
		"route",
		0,
		DecisionPlugin{
			Type:          DecisionPluginSemanticCache,
			Configuration: payload,
		},
	); err == nil {
		t.Fatal("expected unknown semantic-cache field error")
	}
}

func TestDecisionPluginRequiresFastResponseMessage(t *testing.T) {
	err := validateDecisionPluginPayload(
		"route",
		0,
		DecisionPlugin{
			Type: DecisionPluginFastResponse,
			Configuration: MustStructuredPayload(
				FastResponsePluginConfig{},
			),
		},
	)
	if err == nil {
		t.Fatal("expected fast_response message validation error")
	}
}

func TestDecisionPluginAliasResolvesCanonicalAccessor(t *testing.T) {
	decision := &Decision{Plugins: []DecisionPlugin{{
		Type: "semantic_cache",
		Configuration: MustStructuredPayload(
			SemanticCachePluginConfig{Enabled: true},
		),
	}}}
	cache := decision.GetSemanticCacheConfig()
	if cache == nil || !cache.Enabled {
		t.Fatalf("semantic_cache alias did not resolve: %#v", cache)
	}
}

func TestPromptModelRejectsAnthropicAPIFormat(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
			"anthropic-helper": {APIFormat: ClientProtocolAnthropic},
		}},
	}
	err := validateDecisionPromptModel(cfg, Decision{
		Name: "prompt-route",
		Algorithm: &AlgorithmConfig{Prompt: &PromptSelectionConfig{
			Model:        "anthropic-helper",
			Instructions: "Choose.",
		}},
	})
	if err == nil {
		t.Fatal("expected Anthropic prompt helper validation error")
	}
}

func TestPromptModelRejectsNonTextModality(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
			"image-helper": {Modality: "image"},
		}},
	}
	err := validateDecisionPromptModel(cfg, Decision{
		Name: "prompt-route",
		Algorithm: &AlgorithmConfig{Prompt: &PromptSelectionConfig{
			Model:        "image-helper",
			Instructions: "Choose.",
		}},
	})
	if err == nil {
		t.Fatal("expected non-text prompt helper validation error")
	}
}

func TestPromptModelRequiresProviderBackend(t *testing.T) {
	cfg := &RouterConfig{
		Looper: LooperConfig{Endpoint: "http://router:8899"},
		BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
			"helper": {Modality: "text"},
		}},
	}
	err := validateDecisionPromptModel(cfg, Decision{
		Name: "prompt-route",
		Algorithm: &AlgorithmConfig{Prompt: &PromptSelectionConfig{
			Model: "helper", Instructions: "Choose.",
		}},
	})
	if err == nil {
		t.Fatal("expected providerless prompt helper validation error")
	}
}
