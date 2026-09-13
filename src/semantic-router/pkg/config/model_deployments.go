package config

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ModelDeployment declares one local model resource or externally owned
// inference service. Artifact aliases and module policy remain in the catalog;
// recipe bindings select the adapter and head separately from execution.
type ModelDeployment struct {
	Artifact      string           `yaml:"artifact,omitempty" json:"artifact,omitempty"`
	Revision      string           `yaml:"revision,omitempty" json:"revision,omitempty"`
	ExternalModel string           `yaml:"external_model,omitempty" json:"external_model,omitempty"`
	Provider      string           `yaml:"provider" json:"provider"`
	Device        string           `yaml:"device,omitempty" json:"device,omitempty"`
	Precision     string           `yaml:"precision,omitempty" json:"precision,omitempty"`
	Input         ModelInputBudget `yaml:"input,omitempty" json:"input,omitempty"`
}

// ModelInputBudget is a deployment restriction, not an advertised model
// capability. The provider additionally enforces its actual task/tokenizer
// limit. No setting here enables long-context classification.
type ModelInputBudget struct {
	MaxTokens int    `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
	Overflow  string `yaml:"overflow,omitempty" json:"overflow,omitempty"`
}

// ModelBinding is a recipe-local use of a deployment. Head and MappingPath
// describe task interpretation and never imply physical resource compatibility.
type ModelBinding struct {
	Deployment  string `yaml:"deployment" json:"deployment"`
	Contract    string `yaml:"contract" json:"contract"`
	Adapter     string `yaml:"adapter" json:"adapter"`
	Head        string `yaml:"head,omitempty" json:"head,omitempty"`
	MappingPath string `yaml:"mapping_path,omitempty" json:"mapping_path,omitempty"`
}

// ResolvedModelBinding is immutable preparation input, containing no engine
// handles or secrets. The runtime attaches the corresponding typed task handle
// before publishing its generation.
type ResolvedModelBinding struct {
	Recipe     RecipeName
	Name       string
	Binding    ModelBinding
	Deployment ModelDeployment
	Admission  AdmissionConfig
}

// ModelBindingPlan contains already-resolved recipe references. Lookup is
// exact: absence never falls back to a binding from a different recipe.
type ModelBindingPlan struct {
	recipes map[RecipeName]map[string]ResolvedModelBinding
}

func (p *ModelBindingPlan) Lookup(recipe RecipeName, name string) (ResolvedModelBinding, bool) {
	if p == nil {
		return ResolvedModelBinding{}, false
	}
	binding, ok := p.recipes[recipe][name]
	return binding, ok
}

func (d ModelDeployment) WithDefaults() ModelDeployment {
	if d.Provider != "http" {
		if d.Device == "" {
			d.Device = "cpu"
		}
		if d.Precision == "" {
			d.Precision = "native"
		}
	}
	if d.Input.Overflow == "" {
		d.Input.Overflow = "reject"
	}
	return d
}

func (d ModelDeployment) validate(cfg *RouterConfig) error {
	switch d.Provider {
	case "candle", "ort":
		if strings.TrimSpace(d.Artifact) == "" || d.ExternalModel != "" {
			return fmt.Errorf("local deployment requires artifact and cannot set external_model")
		}
		if d.Device != "cpu" {
			parts := strings.Split(d.Device, ":")
			if len(parts) != 2 {
				return fmt.Errorf("device must name cpu or an explicit provider:index")
			}
			index, err := strconv.Atoi(parts[1])
			if err != nil || index < 0 {
				return fmt.Errorf("device index must be a non-negative integer")
			}
			if (d.Provider == "ort" && parts[0] != "migraphx") || (d.Provider == "candle" && parts[0] != "cuda" && parts[0] != "metal") {
				return fmt.Errorf("device %q is incompatible with provider %q", d.Device, d.Provider)
			}
		}
		if d.Precision != "native" && d.Precision != "fp32" && d.Precision != "fp16" {
			return fmt.Errorf("precision must be native, fp32 or fp16")
		}
	case "http":
		if d.Artifact != "" || strings.TrimSpace(d.ExternalModel) == "" {
			return fmt.Errorf("http deployment requires external_model and cannot set artifact")
		}
		if d.Device != "" || d.Precision != "" {
			return fmt.Errorf("external service device and precision are not controlled by the router")
		}
		if _, err := findNamedExternalModel(cfg, d.ExternalModel); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported provider %q", d.Provider)
	}
	if d.Input.MaxTokens < 0 {
		return fmt.Errorf("input.max_tokens must not be negative")
	}
	switch d.Input.Overflow {
	case "reject", "truncate", "window":
	default:
		return fmt.Errorf("unsupported input.overflow %q", d.Input.Overflow)
	}
	return nil
}

// CompileModelBindings resolves all declarations before loading resources.
// Existing module defaults remain canonical catalog bindings; explicit recipe
// declarations override those defaults only for their own consumers.
func CompileModelBindings(cfg *RouterConfig) (*ModelBindingPlan, error) {
	if cfg == nil {
		return nil, fmt.Errorf("model bindings require router configuration")
	}
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		if strings.TrimSpace(name) == "" || strings.TrimSpace(name) != name {
			return nil, fmt.Errorf("model deployment name must be non-empty and trimmed")
		}
		if err := cfg.ModelDeployments[name].WithDefaults().validate(cfg); err != nil {
			return nil, fmt.Errorf("global.model_catalog.deployments.%s: %w", name, err)
		}
	}
	plan := &ModelBindingPlan{recipes: make(map[RecipeName]map[string]ResolvedModelBinding)}
	profiles := cfg.Recipes
	if len(profiles) == 0 {
		recipe := cfg.RoutingScope
		if recipe == "" {
			recipe = DefaultRecipeName
		}
		profiles = []RoutingRecipe{{Name: recipe, Profile: RoutingProfile{ModelBindings: cfg.ModelBindings, Signals: cfg.Signals}}}
	}
	for _, recipe := range profiles {
		bindings := make(map[string]ResolvedModelBinding, len(recipe.Profile.ModelBindings))
		for _, name := range sortedModelKeys(recipe.Profile.ModelBindings) {
			decl := recipe.Profile.ModelBindings[name]
			deployment, exists := cfg.ModelDeployments[decl.Deployment]
			if !exists {
				return nil, fmt.Errorf("recipes[%s].routing.model_bindings.%s: unknown deployment %q", recipe.Name, name, decl.Deployment)
			}
			deployment = deployment.WithDefaults()
			if err := validateTaskModelBinding(name, decl, deployment); err != nil {
				return nil, fmt.Errorf("recipes[%s].routing.model_bindings.%s: %w", recipe.Name, name, err)
			}
			if strings.HasPrefix(name, "classifier.") {
				rule := classifierSignalRuleByName(recipe.Profile.Signals.ClassifierRules, strings.TrimPrefix(name, "classifier."))
				if err := validateGenericModelBinding(cfg, rule, decl, deployment); err != nil {
					return nil, fmt.Errorf("recipes[%s].routing.model_bindings.%s: %w", recipe.Name, name, err)
				}
			}
			bindings[name] = ResolvedModelBinding{Recipe: recipe.Name, Name: name, Binding: decl, Deployment: deployment, Admission: cfg.ModelAdmission[decl.Deployment]}
		}
		plan.recipes[recipe.Name] = bindings
	}
	return plan, nil
}

func validateTaskModelBinding(name string, decl ModelBinding, deployment ModelDeployment) error {
	want := ""
	switch name {
	case "prompt_guard":
		want = RemoteClassifierContractLabelDistribution
		if deployment.Provider == "http" && decl.Adapter == RemoteClassifierProtocolHTTPChat {
			want = "label_decision.v1"
		}
	case "domain_classifier", "fact_check_classifier", "feedback_detector", "modality_detector":
		want = RemoteClassifierContractLabelDistribution
	case "pii_classifier":
		want = RemoteClassifierContractTokenSpans
	case "hallucination_detector":
		want = RemoteClassifierContractTokenSpans
	case "hallucination_explainer":
		want = "text_pair_distribution.v1"
	case "embedding":
		want = "embedding.v1"
	case "complexity":
		if decl.Contract == RemoteClassifierContractLabelDistribution {
			want = RemoteClassifierContractLabelDistribution
			break
		}
		want = RemoteClassifierContractScore
	default:
		if !strings.HasPrefix(name, "classifier.") {
			return fmt.Errorf("unknown task consumer %q", name)
		}
		want = RemoteClassifierContractLabelDistribution
	}
	if decl.Contract != want {
		return fmt.Errorf("contract must be %q for %s", want, name)
	}
	if strings.TrimSpace(decl.Adapter) == "" {
		return fmt.Errorf("adapter is required")
	}
	if name == "complexity" && deployment.Provider != "http" {
		return fmt.Errorf("complexity requires an HTTP score or distribution adapter")
	}
	if deployment.Provider == "http" && decl.Head != "" {
		return fmt.Errorf("remote task cannot bind a local head")
	}
	if deployment.Provider == "http" {
		if name == "fact_check_classifier" || name == "feedback_detector" || name == "modality_detector" || name == "hallucination_explainer" {
			return fmt.Errorf("%s has no HTTP task adapter", name)
		}
		if name != "embedding" && (deployment.Input.MaxTokens != 0 || deployment.Input.Overflow != "reject") {
			return fmt.Errorf("HTTP classifier adapters cannot enforce local tokenizer input budgets")
		}
		if name == "hallucination_detector" && decl.Adapter != RemoteClassifierProtocolHTTPChat {
			return fmt.Errorf("hallucination detector requires http_chat adapter")
		}
	}
	if deployment.Provider == "ort" && (name == "hallucination_detector" || name == "hallucination_explainer") {
		return fmt.Errorf("%s has no ORT task adapter", name)
	}
	if name != "embedding" && deployment.Provider != "http" && deployment.Input.MaxTokens > 512 {
		return fmt.Errorf("classification task supports at most 512 tokens; input.max_tokens is a deployment budget")
	}
	return nil
}

func sortedModelKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func cloneModelMap[T any](values map[string]T) map[string]T {
	if values == nil {
		return nil
	}
	cloned := make(map[string]T, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func validateModelDeploymentContracts(cfg *RouterConfig) error {
	_, err := CompileModelBindings(cfg)
	return err
}
