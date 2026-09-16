package config

import (
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ModelDeployment declares one local model resource or externally owned
// inference service. Artifact aliases and module policy remain in the catalog;
// recipe bindings select the adapter and head separately from execution.
type ModelDeployment struct {
	Artifact            string           `yaml:"artifact,omitempty" json:"artifact,omitempty"`
	Revision            string           `yaml:"revision,omitempty" json:"revision,omitempty"`
	ExternalModel       string           `yaml:"external_model,omitempty" json:"external_model,omitempty"`
	Provider            string           `yaml:"provider" json:"provider"`
	Device              string           `yaml:"device,omitempty" json:"device,omitempty"`
	Precision           string           `yaml:"precision,omitempty" json:"precision,omitempty"`
	CustomOpsProfile    string           `yaml:"custom_ops_profile,omitempty" json:"custom_ops_profile,omitempty"`
	CompilationCacheDir string           `yaml:"compilation_cache_dir,omitempty" json:"compilation_cache_dir,omitempty"`
	Input               ModelInputBudget `yaml:"input,omitempty" json:"input,omitempty"`
}

// ModelInputBudget is a deployment restriction, not an advertised model
// capability. The provider additionally enforces its actual task/tokenizer
// limit. An explicit larger budget requires a checkpoint with that capacity.
type ModelInputBudget struct {
	MaxTokens int    `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
	Overflow  string `yaml:"overflow,omitempty" json:"overflow,omitempty"`
}

// ModelBinding declares a shared task default or a recipe-local deployment use.
// Head and MappingPath
// describe task interpretation and never imply physical resource compatibility.
type ModelBinding struct {
	Deployment     string                   `yaml:"deployment" json:"deployment"`
	Contract       string                   `yaml:"contract" json:"contract"`
	Adapter        string                   `yaml:"adapter" json:"adapter"`
	Head           string                   `yaml:"head,omitempty" json:"head,omitempty"`
	MappingPath    string                   `yaml:"mapping_path,omitempty" json:"mapping_path,omitempty"`
	PairScorer     *PairScorerSelection     `yaml:"pair_scorer,omitempty" json:"pair_scorer,omitempty"`
	OperatingPoint *OperatingPointReference `yaml:"operating_point,omitempty" json:"operating_point,omitempty"`
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
	global  map[string]ResolvedModelBinding
}

func (p *ModelBindingPlan) Lookup(recipe RecipeName, name string) (ResolvedModelBinding, bool) {
	if p == nil {
		return ResolvedModelBinding{}, false
	}
	binding, ok := p.recipes[recipe][name]
	return binding, ok
}

func (d ModelDeployment) WithDefaults() ModelDeployment {
	if d.CustomOpsProfile == "none" {
		d.CustomOpsProfile = ""
	}
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
			if (d.Provider == "ort" && parts[0] != "migraphx" && parts[0] != "rocm") || (d.Provider == "candle" && parts[0] != "cuda" && parts[0] != "metal") {
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
	if d.CustomOpsProfile != "" && (d.CustomOpsProfile != "ck_flash_attention" || d.Provider != "ort" || !strings.HasPrefix(d.Device, "rocm:")) {
		return fmt.Errorf("custom_ops_profile requires ck_flash_attention on an ORT rocm:index deployment")
	}
	if err := d.ValidateCompilationCache(); err != nil {
		return err
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

// ValidateCompilationCache checks an explicitly selected provider cache. An
// empty directory disables caching; runtime preparation checks artifact paths.
func (d ModelDeployment) ValidateCompilationCache() error {
	if d.CompilationCacheDir == "" {
		return nil
	}
	if d.Provider != "ort" || !strings.HasPrefix(d.Device, "migraphx:") {
		return fmt.Errorf("compilation_cache_dir requires an ORT migraphx:index deployment")
	}
	if strings.TrimSpace(d.CompilationCacheDir) != d.CompilationCacheDir || strings.ContainsRune(d.CompilationCacheDir, '\x00') || !filepath.IsAbs(d.CompilationCacheDir) {
		return fmt.Errorf("compilation_cache_dir must be an absolute, trimmed path without null bytes")
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
	if err := validateModelDeploymentContracts(cfg); err != nil {
		return nil, err
	}
	return compileModelBindings(cfg)
}

// compileModelBindings assumes global deployment contracts were already
// validated by the caller, then resolves global defaults and recipe overrides.
func compileModelBindings(cfg *RouterConfig) (*ModelBindingPlan, error) {
	global, err := resolveGlobalModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	plan := &ModelBindingPlan{recipes: make(map[RecipeName]map[string]ResolvedModelBinding), global: global}
	profiles := cfg.Recipes
	if len(profiles) == 0 {
		recipe := cfg.RoutingScope
		if recipe == "" {
			recipe = DefaultRecipeName
		}
		profiles = []RoutingRecipe{{Name: recipe, Profile: RoutingProfile{ModelBindings: cfg.ModelBindings, Signals: cfg.Signals}}}
	}
	for _, recipe := range profiles {
		declarations := cfg.EffectiveModelBindings(recipe.Profile.Signals, recipe.Profile.ModelBindings)
		bindings := make(map[string]ResolvedModelBinding, len(declarations))
		for _, name := range sortedModelKeys(declarations) {
			decl := declarations[name]
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
			if strings.HasPrefix(name, "safety.") {
				if err := validateSafetyModelBinding(recipe.Profile.Signals.SafetyRules, name, decl, deployment); err != nil {
					return nil, fmt.Errorf("recipes[%s].routing.model_bindings.%s: %w", recipe.Name, name, err)
				}
			}
			bindings[name] = ResolvedModelBinding{Recipe: recipe.Name, Name: name, Binding: decl, Deployment: deployment, Admission: cfg.ModelAdmission[decl.Deployment]}
		}
		// The legacy `backend: endpoint` scalar is shorthand for a binding the
		// recipe did not write out. Desugar it here so the runtime has one
		// selection mechanism; an explicit binding for the consumer wins.
		if _, declared := bindings["hallucination_detector"]; !declared {
			decl, deployment, legacy, err := LegacyHallucinationBinding(&cfg.HallucinationMitigation.HallucinationModel)
			if err != nil {
				return nil, fmt.Errorf("global.model_catalog.modules.hallucination_mitigation.detector: %w", err)
			}
			if legacy {
				bindings["hallucination_detector"] = ResolvedModelBinding{Recipe: recipe.Name, Name: "hallucination_detector", Binding: decl, Deployment: deployment.WithDefaults(), Admission: cfg.ModelAdmission[decl.Deployment]}
			}
		}
		plan.recipes[recipe.Name] = bindings
	}
	return plan, nil
}

func validateTaskModelBinding(name string, decl ModelBinding, deployment ModelDeployment) error {
	want := ""
	if decl.OperatingPoint != nil {
		if !strings.HasPrefix(name, "classifier.") {
			return fmt.Errorf("operating_point is only supported by generic classifier bindings")
		}
		if err := decl.OperatingPoint.Validate(); err != nil {
			return err
		}
	}
	if decl.PairScorer != nil && name != RAGRerankerConsumer {
		return fmt.Errorf("pair_scorer selection is only supported by rag.reranker")
	}
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
	case RAGRerankerConsumer:
		want = RelevanceScoresContract
		if err := validateRerankerBinding(decl, deployment); err != nil {
			return err
		}
	case "complexity":
		if decl.Contract == RemoteClassifierContractLabelDistribution {
			want = RemoteClassifierContractLabelDistribution
			break
		}
		want = RemoteClassifierContractScore
	default:
		if strings.HasPrefix(name, "safety.") {
			// The matching rule disambiguates names containing ".hazard".
			want = decl.Contract
			if want != RemoteClassifierContractLabelDistribution && want != RemoteClassifierContractLabelScores {
				return fmt.Errorf("safety binding requires a categorical or independent label contract")
			}
			break
		}
		if !strings.HasPrefix(name, "classifier.") {
			return fmt.Errorf("unknown task consumer %q", name)
		}
		want = RemoteClassifierContractLabelDistribution
		if decl.Contract == RemoteClassifierContractLabelScores {
			want = RemoteClassifierContractLabelScores
		}
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
		if name == "hallucination_detector" && decl.Adapter != RemoteClassifierProtocolHTTPChat && decl.Adapter != RemoteClassifierProtocolHTTPClassify {
			return fmt.Errorf("hallucination detector requires http_chat or http_classify adapter")
		}
	}
	if deployment.Provider == "ort" && (name == "hallucination_detector" || name == "hallucination_explainer") {
		return fmt.Errorf("%s has no ORT task adapter", name)
	}
	// Artifact-specific capacity is checked by the loaded provider. Config
	// cannot infer a checkpoint limit from its adapter name or a fixed 512 cap.
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
	if cfg == nil {
		return fmt.Errorf("model bindings require router configuration")
	}
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		if strings.TrimSpace(name) == "" || strings.TrimSpace(name) != name {
			return fmt.Errorf("model deployment name must be non-empty and trimmed")
		}
		if err := cfg.ModelDeployments[name].WithDefaults().validate(cfg); err != nil {
			return fmt.Errorf("global.model_catalog.deployments.%s: %w", name, err)
		}
	}
	return nil
}

func validateModelBindingContracts(cfg *RouterConfig) error {
	_, err := compileModelBindings(cfg)
	return err
}
