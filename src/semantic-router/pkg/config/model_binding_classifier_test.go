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
