package config

import (
	"fmt"
	"slices"
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
	// MaxNumSeqs bounds the rows (one per question or input) one forward of a
	// model_runtime deployment holds; 1 runs every row alone. Unset leaves it
	// to the runtime, which picks it by the device's memory class.
	MaxNumSeqs *int `yaml:"max_num_seqs,omitempty" json:"max_num_seqs,omitempty"`
	// MaxNumBatchedTokens bounds one forward in tokens: the sum of its rows
	// for a model that packs merged rows, else its rows times its longest row.
	// Unset leaves it to the runtime: the device class's budget, adjusted by a
	// short startup calibration on GPUs.
	MaxNumBatchedTokens *int `yaml:"max_num_batched_tokens,omitempty" json:"max_num_batched_tokens,omitempty"`
	// BatchWindowMS is how long the runtime waits after the first request to
	// collect more (runtime default 2 ms).
	BatchWindowMS *float64 `yaml:"batch_window_ms,omitempty" json:"batch_window_ms,omitempty"`
	// Dtype is the precision the backbone computes in: auto (the default, the
	// precision the model was published in), float32, bfloat16 or float16.
	Dtype string `yaml:"dtype,omitempty" json:"dtype,omitempty"`
	// EnableSharedContext computes the shared state of a multi-question
	// request once; unset leaves it to the model family (on for Decision 2.0).
	EnableSharedContext *bool `yaml:"enable_shared_context,omitempty" json:"enable_shared_context,omitempty"`
	// MaxModelLen is the most input tokens one question of a local Decision
	// deployment may have: it lowers the package's limit and never raises it.
	// Unset keeps the package's limit; attached endpoints and replicas refuse it.
	MaxModelLen *int `yaml:"max_model_len,omitempty" json:"max_model_len,omitempty"`
	// Endpoint attaches a model_runtime deployment to an engine the Router does
	// not manage (unix:///path, http://host:port or https://host:port). For a
	// systemone deployment it is the URL its System One API answers POST
	// requests on, path prefix included (https://host/v1/systemone); for the
	// other remote providers it is the base URL of their API
	// (https://host/v1), which their operation path follows.
	Endpoint string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	// Model is the model a remote API answers with; on an attached
	// model_runtime endpoint it selects the served model (served_name).
	Model string `yaml:"model,omitempty" json:"model,omitempty"`
	// Auth is the credential an attached or remote deployment takes.
	Auth *DeploymentAuth `yaml:"auth,omitempty" json:"auth,omitempty"`
	// Headers are further non-secret headers every call carries.
	Headers map[string]string `yaml:"headers,omitempty" json:"headers,omitempty"`
	// TimeoutMs bounds one attempt of a call to an attached or remote
	// deployment; a decision question or selector's own timeout_ms wins.
	TimeoutMs int `yaml:"timeout_ms,omitempty" json:"timeout_ms,omitempty"`
	// MaxRetries is the number of transport retries after a first attempt,
	// within the caller's deadline.
	MaxRetries *int `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`
	// TLS verifies an https endpoint and presents a client certificate.
	TLS *DeploymentTLS `yaml:"tls,omitempty" json:"tls,omitempty"`
	// Limits bounds the request and response bodies of one exchange.
	Limits *DeploymentLimits `yaml:"limits,omitempty" json:"limits,omitempty"`
	// Capabilities declares what a remote model answers when its API lists
	// nothing for it.
	Capabilities *DeploymentCapabilities `yaml:"capabilities,omitempty" json:"capabilities,omitempty"`
	// OperatingPoint pins the packaged operating point of a managed scores
	// head: a classifier rule the deployment answers reads independent label
	// scores at the thresholds of the policy file with this digest.
	OperatingPoint *OperatingPointReference `yaml:"operating_point,omitempty" json:"operating_point,omitempty"`
	// Generation sets the completion length and sampling temperature of an
	// openai_chat deployment's calls.
	Generation *DeploymentGeneration `yaml:"generation,omitempty" json:"generation,omitempty"`
	// QuestionTypes is the deprecated spelling of capabilities.question_types.
	QuestionTypes []string `yaml:"question_types,omitempty" json:"question_types,omitempty"`
	// Replicas places independent workers for this logical resource. Omission
	// uses the single device/endpoint placement above.
	Replicas []ModelReplica `yaml:"replicas,omitempty" json:"replicas,omitempty"`
	// ServedName is the deprecated spelling of Model on an attached runtime.
	ServedName string `yaml:"served_name,omitempty" json:"served_name,omitempty"`
	// PublicName is the inference API identity. It never identifies a socket,
	// local package path, or internal deployment key.
	PublicName string `yaml:"public_name,omitempty" json:"public_name,omitempty"`

	// translated marks a deployment a deprecated model path declared.
	translated bool
	// legacyPath is the model path a translated model_runtime deployment
	// came from.
	legacyPath string
	// legacyExternal is the external[] model a translated deployment came
	// from.
	legacyExternal *ExternalModelConfig
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

// ModelBinding names the deployment that answers one signal type, globally
// or in one recipe (see model_binding_forms.go). Head selects a classify head
// of the deployment; the other fields are deprecated.
type ModelBinding struct {
	Deployment     string                   `yaml:"deployment" json:"deployment"`
	Contract       string                   `yaml:"contract,omitempty" json:"contract,omitempty"`
	Adapter        string                   `yaml:"adapter,omitempty" json:"adapter,omitempty"`
	Head           string                   `yaml:"head,omitempty" json:"head,omitempty"`
	MappingPath    string                   `yaml:"mapping_path,omitempty" json:"mapping_path,omitempty"`
	PairScorer     *PairScorerSelection     `yaml:"pair_scorer,omitempty" json:"pair_scorer,omitempty"`
	OperatingPoint *OperatingPointReference `yaml:"operating_point,omitempty" json:"operating_point,omitempty"`

	// contractImplied marks a contract the signal implied rather than the
	// configuration wrote.
	contractImplied bool
}

// ContractImplied reports whether the binding's contract is the one its
// signal implies, not one the configuration wrote.
func (b ModelBinding) ContractImplied() bool { return b.contractImplied }

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
	d.Replicas = append([]ModelReplica(nil), d.Replicas...)
	if d.Provider == ModelRuntimeProvider {
		if d.Device == "" && len(d.Replicas) == 0 && d.Endpoint == "" {
			d.Device = "auto"
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
	case SystemOneProvider:
		return d.validateSystemOne()
	case OpenAIEmbeddingsProvider, ClassifyProvider, OpenAIChatProvider, RerankProvider:
		return d.validateRemote()
	case LegacyHTTPProvider:
		if d.hasConnectionBlock() || d.Capabilities != nil {
			return fmt.Errorf("an http deployment names an external_model; declare provider classify or openai_chat with a connection block instead")
		}
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
	if d.HasEngineArgs() || d.Endpoint != "" || len(d.Replicas) != 0 || d.ServedName != "" {
		return fmt.Errorf("engine arguments, endpoint, replicas and served_name apply only to model_runtime deployments")
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
// validated by the caller. Each recipe's plan holds the consumers whose
// deployment the configuration names, by rule or binding (model_resolution.go),
// with the contract and adapter their signal and provider imply.
func compileModelBindings(cfg *RouterConfig) (*ModelBindingPlan, error) {
	cfg = cfg.withModelResources()
	global, err := resolveGlobalModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	plan := &ModelBindingPlan{recipes: make(map[RecipeName]map[string]ResolvedModelBinding), global: global}
	scopes := map[RecipeName]*RouterConfig{}
	if len(cfg.Recipes) == 0 {
		recipe := cfg.RoutingScope
		if recipe == "" {
			recipe = DefaultRecipeName
		}
		scopes[recipe] = cfg
	}
	for i := range cfg.Recipes {
		scopes[cfg.Recipes[i].Name] = cfg.ConfigForRecipe(&cfg.Recipes[i])
	}
	for _, recipe := range sortedRecipeNames(scopes) {
		scoped := scopes[recipe]
		if err := validateDeclaredBindingKeys(recipe, scoped); err != nil {
			return nil, err
		}
		bindings := make(map[string]ResolvedModelBinding)
		for _, consumer := range scoped.scopeConsumers() {
			resolved, ok, err := scoped.resolveTask(consumer)
			if err != nil && scoped.translationError(err) {
				continue
			}
			if err != nil {
				return nil, fmt.Errorf("recipes[%s]: %w", recipe, err)
			}
			if !ok || !resolved.Explicit() {
				continue
			}
			decl := resolved.Binding
			decl.Deployment = resolved.Name
			deployment := resolved.Deployment
			decl = withImpliedContract(consumer, decl, deployment)
			path := fmt.Sprintf("recipes[%s].routing.model_bindings.%s", recipe, BindingKey(consumer))
			if err := validateTaskModelBinding(consumer, decl, deployment); err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			bindings[consumer] = ResolvedModelBinding{Recipe: recipe, Name: consumer, Binding: decl, Deployment: deployment, Admission: cfg.ModelAdmission[decl.Deployment]}
		}
		plan.recipes[recipe] = bindings
	}
	return plan, nil
}

func sortedRecipeNames(scopes map[RecipeName]*RouterConfig) []RecipeName {
	names := make([]RecipeName, 0, len(scopes))
	for name := range scopes {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return names[i] < names[j] })
	return names
}

// validateDeclaredBindingKeys refuses a recipe binding for a rule the recipe
// does not declare.
func validateDeclaredBindingKeys(recipe RecipeName, scoped *RouterConfig) error {
	for _, key := range sortedModelKeys(scoped.recipeBindings()) {
		binding := scoped.recipeBindings()[key]
		if !knownBindingKey(key) {
			return fmt.Errorf("recipes[%s].routing.model_bindings.%s: unknown task consumer %q", recipe, BindingKey(key), key)
		}
		deployment, exists := scoped.ModelDeployments[binding.Deployment]
		if !exists {
			return fmt.Errorf("recipes[%s].routing.model_bindings.%s: unknown deployment %q; %s", recipe, BindingKey(key), binding.Deployment, declaredDeployments(scoped.ModelDeployments))
		}
		if isApplicationConsumer(key) {
			deployment = deployment.WithDefaults()
			if err := validateTaskModelBinding(key, withImpliedContract(key, binding, deployment), deployment); err != nil {
				return fmt.Errorf("recipes[%s].routing.model_bindings.%s: %w", recipe, BindingKey(key), err)
			}
		}
		if rule, isRuleKey := strings.CutPrefix(key, "safety."); isRuleKey && !slices.ContainsFunc(scoped.SafetyRules, func(safety SafetyRule) bool { return safety.Name == rule }) {
			return fmt.Errorf("recipes[%s].routing.model_bindings.%s: names no safety rule of the recipe", recipe, key)
		}
		if rule, isRuleKey := strings.CutPrefix(key, "complexity."); isRuleKey && !slices.ContainsFunc(scoped.ComplexityRules, func(complexity ComplexityRule) bool { return complexity.Name == rule }) {
			return fmt.Errorf("recipes[%s].routing.model_bindings.%s: names no complexity rule of the recipe", recipe, key)
		}
	}
	return nil
}

// withImpliedContract fills the contract and adapter a binding leaves out:
// the signal decides the contract, the provider the adapter.
// ImpliedBinding is a binding with the contract its signal implies and the
// adapter its deployment's provider implies, where it writes neither.
func ImpliedBinding(consumer string, decl ModelBinding, deployment ModelDeployment) ModelBinding {
	return withImpliedContract(consumer, decl, deployment)
}

func withImpliedContract(consumer string, decl ModelBinding, deployment ModelDeployment) ModelBinding {
	if decl.Contract == "" {
		decl.Contract, decl.contractImplied = impliedContract(consumer), true
	}
	if decl.Adapter == "" {
		switch deployment.Provider {
		case ClassifyProvider:
			decl.Adapter = RemoteClassifierProtocolHTTPClassify
		case OpenAIChatProvider:
			decl.Adapter = RemoteClassifierProtocolHTTPChat
		case OpenAIEmbeddingsProvider:
			decl.Adapter = EmbeddingBackendOpenAICompatible
		case ModelRuntimeProvider:
			// A built-in model's card names the adapter its published task
			// reads with, as an explicit adapter on the binding did.
			if model := RegisteredModel(deployment.Artifact); model != nil {
				decl.Adapter = model.DefaultAdapter
			}
		}
	}
	return decl
}

func impliedContract(consumer string) string {
	consumer = ruleConsumer(consumer)
	switch {
	case consumer == "embedding":
		return "embedding.v1"
	case consumer == RAGRerankerConsumer:
		return RelevanceScoresContract
	case isApplicationConsumer(BindingKey(consumer)) || strings.HasPrefix(consumer, "systemone."):
		return DecisionTaskContract
	}
	return RemoteClassifierContractLabelDistribution
}

// remoteTaskProvider reports whether a deployment answers a task over a
// remote classify, chat or embeddings API (or the deprecated http provider).
func remoteTaskProvider(deployment ModelDeployment) bool {
	switch deployment.Provider {
	case LegacyHTTPProvider, ClassifyProvider, OpenAIChatProvider, OpenAIEmbeddingsProvider:
		return true
	}
	return false
}

func validateTaskModelBinding(name string, decl ModelBinding, deployment ModelDeployment) error {
	if err := providerAnswers(name, deployment, decl); err != nil {
		return err
	}
	name = ruleConsumer(name)
	if deployment.IsSystemOne() {
		return nil
	}
	if deployment.Provider == RerankProvider {
		if decl.MappingPath != "" || decl.Head != "" {
			return fmt.Errorf("a rerank deployment returns relevance scores; it takes no head or mapping_path")
		}
		return nil
	}
	if decl.Contract == DecisionTaskContract {
		allowed := isApplicationConsumer(BindingKey(name)) || strings.HasPrefix(name, "systemone.")
		if !allowed || !deployment.IsModelRuntime() || decl.Head != "" || decl.MappingPath != "" || decl.OperatingPoint != nil || decl.PairScorer != nil {
			return fmt.Errorf("decision.v1 requires a supported judgment consumer and model_runtime deployment, without head, mapping_path or a classifier operating point")
		}
		return nil
	}
	want := ""
	if decl.OperatingPoint != nil {
		return fmt.Errorf("operating_point calibrated a generic classifier binding; classifier rules are systemone questions now")
	}
	if decl.PairScorer != nil && name != RAGRerankerConsumer {
		return fmt.Errorf("pair_scorer selection is only supported by rag.reranker")
	}
	switch {
	case name == "embedding":
		want = "embedding.v1"
	case name == RAGRerankerConsumer:
		want = RelevanceScoresContract
		if err := validateRerankerBinding(decl, deployment); err != nil {
			return err
		}
	case isApplicationConsumer(BindingKey(name)):
		want = DecisionTaskContract
	default:
		return fmt.Errorf("unknown task consumer %q", name)
	}
	if decl.Contract != want {
		return fmt.Errorf("contract must be %q for %s", want, name)
	}
	if strings.TrimSpace(decl.Adapter) == "" && !deployment.IsModelRuntime() {
		return fmt.Errorf("adapter is required")
	}
	if remoteTaskProvider(deployment) && decl.Head != "" {
		return fmt.Errorf("remote task cannot bind a local head")
	}
	if remoteTaskProvider(deployment) && name != "embedding" && (deployment.Input.MaxTokens != 0 || deployment.Input.Overflow != "reject") {
		return fmt.Errorf("HTTP classifier adapters cannot enforce local tokenizer input budgets")
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
		if deployment, ok := any(value).(ModelDeployment); ok {
			deployment.Replicas = append([]ModelReplica(nil), deployment.Replicas...)
			cloned[key] = any(deployment).(T)
		} else {
			cloned[key] = value
		}
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
