package config

import (
	"fmt"
	"sort"
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
	Input         ModelInputBudget `yaml:"input,omitempty" json:"input,omitempty"`
	// Profile selects a model_runtime numerics profile: exact (the default,
	// byte-identical to the released packages) or an opt-in faster profile.
	Profile string `yaml:"profile,omitempty" json:"profile,omitempty"`
	// Endpoint attaches a model_runtime deployment to an engine the Router does
	// not manage (unix:///path, http://host:port or https://host:port).
	Endpoint string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	// Process groups managed model_runtime deployments into one runtime
	// process; without it the Router runs one process per device.
	Process string `yaml:"process,omitempty" json:"process,omitempty"`
	// ServedName selects the model on an attached runtime that serves several
	// (default: the deployment name).
	ServedName string `yaml:"served_name,omitempty" json:"served_name,omitempty"`
	// PublicName is the inference API identity. It never identifies a socket,
	// local package path, or internal deployment key.
	PublicName string `yaml:"public_name,omitempty" json:"public_name,omitempty"`
}

// PublicModelName returns a safe public identity, independently of the name
// used by an attached upstream runtime. Local artifacts require public_name.
func (d ModelDeployment) PublicModelName() string {
	if name := strings.TrimSpace(d.PublicName); name != "" {
		return name
	}
	if !strings.HasPrefix(d.Artifact, "models/") && !strings.HasPrefix(d.Artifact, "./") && !strings.HasPrefix(d.Artifact, "../") && hubRepositoryID.MatchString(d.Artifact) {
		return d.Artifact
	}
	return ""
}

// ModelInputBudget is a deployment restriction, not an advertised model
// capability. With window overflow, MaxTokens admits the complete document;
// the consumer's window must fit the provider's actual task/tokenizer limit.
// Other overflow policies require the budget to fit that single-forward limit.
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

// DecisionTaskContract binds a semantic judgment without requiring token
// positions or a classifier head. Its concrete typed question comes from the
// consumer's task definition.
const DecisionTaskContract = "decision.v1"

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
	if d.Provider == ModelRuntimeProvider {
		if d.Device == "" {
			d.Device = "auto"
		}
		if d.Profile == "" {
			d.Profile = "exact"
		}
	}
	if d.Input.Overflow == "" {
		d.Input.Overflow = "reject"
	}
	return d
}

// ScanBudget is the scan budget a decision deployment declares with input
// {overflow: window, max_tokens}: the most tokens of one state part a model
// that reads a long part in windows (Vela 2.0) reads. Zero keeps the model's
// own budget.
func (d ModelDeployment) ScanBudget() int {
	if d.Input.Overflow == "window" {
		return d.Input.MaxTokens
	}
	return 0
}

// ValidateDecisionInput checks the input of a deployment that answers
// decisions: none, or a scan budget. Decision models never truncate.
func (d ModelDeployment) ValidateDecisionInput(name string) error {
	input := d.WithDefaults().Input
	if (input.MaxTokens == 0 && input.Overflow == "reject") || (input.Overflow == "window" && input.MaxTokens > 0) {
		return nil
	}
	return fmt.Errorf("deployment %q: decision models never truncate; input takes overflow: window with max_tokens, the scan budget of a model that reads a long part in windows, or nothing", name)
}

func (d ModelDeployment) validate(cfg *RouterConfig) error {
	if d.PublicName != "" && (strings.TrimSpace(d.PublicName) != d.PublicName || strings.ContainsAny(d.PublicName, "\x00\r\n\t ") || strings.HasPrefix(d.PublicName, "/") || strings.Contains(d.PublicName, "://")) {
		return fmt.Errorf("public_name must be a trimmed public model ID, not a path or endpoint")
	}
	switch d.Provider {
	case "http":
		if d.Artifact != "" || strings.TrimSpace(d.ExternalModel) == "" {
			return fmt.Errorf("http deployment requires external_model and cannot set artifact")
		}
		if d.Device != "" {
			return fmt.Errorf("external service devices are not controlled by the router")
		}
		if _, err := findNamedExternalModel(cfg, d.ExternalModel); err != nil {
			return err
		}
	case ModelRuntimeProvider:
		return d.validateModelRuntime()
	default:
		return fmt.Errorf("unsupported provider %q", d.Provider)
	}
	if d.Profile != "" || d.Endpoint != "" || d.Process != "" || d.ServedName != "" {
		return fmt.Errorf("profile, endpoint, process and served_name apply only to model_runtime deployments")
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
		plan.recipes[recipe.Name] = bindings
	}
	return plan, nil
}

func validateTaskModelBinding(name string, decl ModelBinding, deployment ModelDeployment) error {
	if decl.Contract == DecisionTaskContract {
		allowed := name == "pii_classifier" || name == "hallucination_detector" || name == "preference" || name == "reask" || name == "complexity" || strings.HasPrefix(name, "classifier.") || strings.HasPrefix(name, "safety.")
		if !allowed || !deployment.IsModelRuntime() || decl.Head != "" || decl.MappingPath != "" || decl.OperatingPoint != nil || decl.PairScorer != nil {
			return fmt.Errorf("decision.v1 requires a supported judgment consumer and model_runtime deployment, without head, mapping_path or a classifier operating point")
		}
		return nil
	}
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
	if strings.TrimSpace(decl.Adapter) == "" && !deployment.IsModelRuntime() {
		return fmt.Errorf("adapter is required")
	}
	if name == "complexity" && deployment.Provider != "http" {
		return fmt.Errorf("complexity requires an HTTP score or distribution adapter")
	}
	if deployment.Provider == "http" && decl.Head != "" {
		return fmt.Errorf("remote task cannot bind a local head")
	}
	if deployment.Provider == "http" {
		if name == "fact_check_classifier" || name == "feedback_detector" || name == "modality_detector" {
			return fmt.Errorf("%s has no HTTP task adapter", name)
		}
		if name != "embedding" && (deployment.Input.MaxTokens != 0 || deployment.Input.Overflow != "reject") {
			return fmt.Errorf("HTTP classifier adapters cannot enforce local tokenizer input budgets")
		}
		if name == "hallucination_detector" && decl.Adapter != RemoteClassifierProtocolHTTPChat && decl.Adapter != RemoteClassifierProtocolHTTPClassify {
			return fmt.Errorf("hallucination detector requires http_chat or http_classify adapter")
		}
	}
	if decl.Adapter == "vela_halu" {
		if name != "hallucination_detector" || !deployment.IsModelRuntime() {
			return fmt.Errorf("vela_halu requires a local hallucination_detector binding")
		}
		if deployment.Input.MaxTokens > 8192 {
			return fmt.Errorf("vela_halu task input budget cannot exceed 8192 tokens")
		}
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
		if strings.HasPrefix(name, ImplicitDeploymentPrefix) {
			return fmt.Errorf("global.model_catalog.deployments.%s: names starting with %q are reserved for module defaults", name, ImplicitDeploymentPrefix)
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
