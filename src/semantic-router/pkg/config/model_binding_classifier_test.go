package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func genericBindingConfig(provider, ruleType string) *RouterConfig {
	cfg := &RouterConfig{ExternalModels: []ExternalModelConfig{{Name: "new-endpoint", ModelRole: ModelRoleClassification, ModelName: "served-model", ModelEndpoint: ClassifierVLLMEndpoint{Address: "localhost", Port: 8080}}}}
	cfg.ClassifierRules = []ClassifierSignalRule{{Name: "risk.tenant", Type: ruleType, ModelPath: "models/obsolete", Labels: []string{"safe", "unsafe"}}}
	deployment := ModelDeployment{Provider: provider, Artifact: "models/selected"}
	adapter := "modernbert"
	if provider == "http" {
		deployment.Artifact, deployment.ExternalModel = "", "new-endpoint"
		adapter = RemoteClassifierProtocolHTTPClassify
	}
	if ruleType != ClassifierSignalTypeLocal {
		cfg.ClassifierRules[0].ModelPath = ""
		cfg.ClassifierRules[0].Model = "removed-endpoint"
	}
	if ruleType == ClassifierSignalTypeLLM {
		cfg.ClassifierRules[0].Instructions = "Score risk."
		adapter = RemoteClassifierProtocolHTTPChat
	}
	cfg.ModelDeployments = map[string]ModelDeployment{"selected": deployment}
	cfg.ModelBindings = map[string]ModelBinding{"classifier.risk.tenant": {Deployment: "selected", Adapter: adapter, Contract: RemoteClassifierContractLabelDistribution}}
	return cfg
}

func TestGenericBindingUsesResolvedProviderAndKeepsCanonicalSelectors(t *testing.T) {
	for _, provider := range []string{"candle", "ort", "http"} {
		for _, ruleType := range []string{ClassifierSignalTypeLocal, ClassifierSignalTypeSequenceClassifier} {
			t.Run(provider+"/"+ruleType, func(t *testing.T) {
				cfg := genericBindingConfig(provider, ruleType)
				original := cfg.ClassifierRules[0]
				if err := validateClassifierSignalContracts(cfg); err != nil {
					t.Fatal(err)
				}
				plan, err := CompileModelBindings(cfg)
				if err != nil {
					t.Fatal(err)
				}
				projected, err := ProjectRecipeModelBindings(cfg, plan, DefaultRecipeName)
				if err != nil {
					t.Fatal(err)
				}
				rule := projected.ClassifierRules[0]
				if provider == "http" {
					if rule.Type != ClassifierSignalTypeSequenceClassifier || rule.Model != "new-endpoint" || rule.ModelPath != "" {
						t.Fatalf("remote rule: %+v", rule)
					}
				} else if rule.Type != ClassifierSignalTypeLocal || rule.ModelPath != "models/selected" || rule.Model != "" {
					t.Fatalf("local rule: %+v", rule)
				}
				if !reflect.DeepEqual(cfg.ClassifierRules[0], original) {
					t.Fatal("canonical rule mutated")
				}
			})
		}
	}
}

func TestGenericBindingRequiresExactPrivateRuleAndSupportedExtraction(t *testing.T) {
	for _, scenario := range []string{"foreign rule", "mapping file", "chat sequence", "local llm", "decision contract", "wrong role", "bad endpoint"} {
		t.Run(scenario, func(t *testing.T) {
			cfg := genericBindingConfig("http", ClassifierSignalTypeSequenceClassifier)
			decl := cfg.ModelBindings["classifier.risk.tenant"]
			switch scenario {
			case "foreign rule":
				cfg.Recipes = []RoutingRecipe{{Name: "private", Profile: RoutingProfile{ModelBindings: cfg.ModelBindings}}}
			case "mapping file":
				decl.MappingPath = "ignored.json"
			case "chat sequence":
				decl.Adapter = RemoteClassifierProtocolHTTPChat
			case "local llm":
				cfg = genericBindingConfig("candle", ClassifierSignalTypeLLM)
				decl = cfg.ModelBindings["classifier.risk.tenant"]
			case "decision contract":
				decl.Contract = RemoteClassifierContractLabelDecision
			case "wrong role":
				cfg.ExternalModels[0].ModelRole = ModelRoleGuardrail
			case "bad endpoint":
				cfg.ExternalModels[0].ModelEndpoint.Port = 0
			}
			cfg.ModelBindings["classifier.risk.tenant"] = decl
			if _, err := CompileModelBindings(cfg); err == nil {
				t.Fatal("invalid binding accepted")
			}
		})
	}
	cfg := genericBindingConfig("http", ClassifierSignalTypeLLM)
	if _, err := CompileModelBindings(cfg); err != nil {
		t.Fatal(err)
	}
	if err := validateClassifierSignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestMultipleLocalClassifierRulesRoundTripIndependently(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ClassifierRules = []ClassifierSignalRule{
		{Name: "risk", Type: ClassifierSignalTypeLocal, ModelPath: "models/risk", Labels: []string{"safe", "unsafe"}},
		{Name: "topic", Type: ClassifierSignalTypeLocal, ModelPath: "models/topic", Labels: []string{"billing", "support", "other"}},
	}
	if err := validateClassifierSignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
	data, err := yaml.Marshal(cfg.Signals)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip Signals
	if decodeErr := yaml.UnmarshalStrict(data, &roundTrip); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if !reflect.DeepEqual(cfg.ClassifierRules, roundTrip.ClassifierRules) {
		t.Fatal("independent model selectors lost")
	}
	cfg.ClassifierRules[1].Name = "RISK"
	if err := validateClassifierSignalContracts(cfg); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate identity error=%v", err)
	}
}

func TestIndependentBindingAndDecisionContract(t *testing.T) {
	cfg := genericBindingConfig("candle", ClassifierSignalTypeLocal)
	decl := cfg.ModelBindings["classifier.risk.tenant"]
	decl.Contract = RemoteClassifierContractLabelScores
	decl.OperatingPoint = &OperatingPointReference{Path: "point.json", SHA256: strings.Repeat("a", 64)}
	cfg.ModelBindings["classifier.risk.tenant"] = decl
	deployment := cfg.ModelDeployments["selected"]
	deployment.Input = ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}
	cfg.ModelDeployments["selected"] = deployment
	if _, err := CompileModelBindings(cfg); err != nil {
		t.Fatal(err)
	}
	node := RuleNode{Type: SignalTypeClassifier, Name: "risk.tenant", Label: "unsafe"}
	if err := validateClassifierDecisionLeaf(cfg, "route", &node); err != nil {
		t.Fatal(err)
	}
	raw, err := yaml.Marshal(decl)
	if err != nil {
		t.Fatal(err)
	}
	var restored ModelBinding
	if err := yaml.UnmarshalStrict(raw, &restored); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decl, restored) {
		t.Fatal("policy reference lost in canonical roundtrip")
	}
	for _, scenario := range []string{"no policy", "categorical policy", "HTTP", "separate head", "bad sha", "escape", "missing budget", "truncate", "half precision"} {
		t.Run(scenario, func(t *testing.T) {
			bad := decl
			dep := deployment
			ref := *decl.OperatingPoint
			bad.OperatingPoint = &ref
			switch scenario {
			case "no policy":
				bad.OperatingPoint = nil
			case "categorical policy":
				bad.Contract = RemoteClassifierContractLabelDistribution
			case "HTTP":
				dep.Provider = "http"
			case "separate head":
				bad.Head = "head"
			case "bad sha":
				ref.SHA256 = "abc"
			case "escape":
				ref.Path = "../policy.json"
			case "missing budget":
				dep.Input.MaxTokens = 0
			case "truncate":
				dep.Input.Overflow = "truncate"
			case "half precision":
				dep.Precision = "fp16"
			}
			cfg.ModelBindings["classifier.risk.tenant"] = bad
			cfg.ModelDeployments["selected"] = dep
			if _, err := CompileModelBindings(cfg); err == nil {
				t.Fatal("invalid independent binding accepted")
			}
		})
	}
	cfg.ModelBindings = nil
	if validateClassifierDecisionLeaf(cfg, "route", &node) == nil {
		t.Fatal("unbound classifier omitted predicate")
	}
}

func TestIndependentORTBindingDefersGraphIdentityToPreparation(t *testing.T) {
	cfg := genericBindingConfig("ort", ClassifierSignalTypeLocal)
	decl := cfg.ModelBindings["classifier.risk.tenant"]
	decl.Contract = RemoteClassifierContractLabelScores
	decl.OperatingPoint = &OperatingPointReference{Path: "point.json", SHA256: strings.Repeat("a", 64)}
	decl.Head = "onnx/model.onnx"
	cfg.ModelBindings["classifier.risk.tenant"] = decl
	dep := cfg.ModelDeployments["selected"]
	dep.Input = ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}
	cfg.ModelDeployments["selected"] = dep
	if _, err := CompileModelBindings(cfg); err != nil {
		t.Fatal(err)
	}
	dep.Precision = "fp16"
	cfg.ModelDeployments["selected"] = dep
	if _, err := CompileModelBindings(cfg); err == nil {
		t.Fatal("unqualified conversion accepted")
	}
}
