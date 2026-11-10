package config

import (
	"cmp"
	"fmt"
	"sort"
	"strings"
)

// ModelDeployment declares one local model resource or externally owned
// inference service. Artifact aliases and module policy remain in the catalog;
// recipe bindings select the adapter and head separately from execution.
type ModelDeployment struct {
	Artifact string           `yaml:"artifact,omitempty" json:"artifact,omitempty"`
	Revision string           `yaml:"revision,omitempty" json:"revision,omitempty"`
	Provider string           `yaml:"provider" json:"provider"`
	Device   string           `yaml:"device,omitempty" json:"device,omitempty"`
	Input    ModelInputBudget `yaml:"input,omitempty" json:"input,omitempty"`
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
	// model_runtime endpoint it selects the served model.
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
	// Generation sets the completion length and sampling temperature of an
	// openai_chat deployment's calls.
	Generation *DeploymentGeneration `yaml:"generation,omitempty" json:"generation,omitempty"`
	// Replicas places independent workers for this logical resource. Omission
	// uses the single device/endpoint placement above.
	Replicas []ModelReplica `yaml:"replicas,omitempty" json:"replicas,omitempty"`
	// OutlierDetection ejects a model_runtime replica whose exchanges keep
	// running out of their deadline; unset keeps its defaults.
	OutlierDetection *DeploymentOutlierDetection `yaml:"outlier_detection,omitempty" json:"outlier_detection,omitempty"`
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

// ModelBinding names the deployment that answers one application signal
// type, globally or in one recipe (see model_binding_forms.go).
type ModelBinding struct {
	Deployment string `yaml:"deployment" json:"deployment"`
}

// ResolvedModelBinding is immutable preparation input, containing no engine
// handles or secrets: the deployment one consumer asks and the contract it
// reads the answers by. The runtime attaches the corresponding typed task
// handle before publishing its generation.
type ResolvedModelBinding struct {
	Recipe     RecipeName
	Name       string
	Contract   string
	Binding    ModelBinding
	Deployment ModelDeployment
	Admission  AdmissionConfig
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

func (d ModelDeployment) validate() error {
	if d.PublicName != "" && (strings.TrimSpace(d.PublicName) != d.PublicName || strings.ContainsAny(d.PublicName, "\x00\r\n\t ") || strings.HasPrefix(d.PublicName, "/") || strings.Contains(d.PublicName, "://")) {
		return fmt.Errorf("public_name must be a trimmed public model ID, not a path or endpoint")
	}
	switch d.Provider {
	case SystemOneProvider:
		return d.validateSystemOne()
	case OpenAIEmbeddingsProvider, OpenAIChatProvider:
		return d.validateRemote()
	case ModelRuntimeProvider:
		return d.validateModelRuntime()
	default:
		return fmt.Errorf("unsupported provider %q", d.Provider)
	}
}

// validateRecipeBindings checks each binding a recipe writes: it names a
// declared deployment that answers its application signal.
func validateRecipeBindings(recipe RecipeName, scoped *RouterConfig) error {
	bindings := scoped.recipeBindings()
	for _, key := range sortedModelKeys(bindings) {
		if err := validateModelBinding(scoped, key, bindings[key]); err != nil {
			return fmt.Errorf("recipes[%s].model_bindings.%s: %w", recipe, key, err)
		}
	}
	return nil
}

// validateModelBinding checks one binding: an application signal type bound
// to a declared deployment whose provider answers it.
func validateModelBinding(cfg *RouterConfig, key string, binding ModelBinding) error {
	if !knownBindingKey(key) {
		return fmt.Errorf("unknown signal type %q", key)
	}
	deployment, exists := cfg.ModelDeployments[binding.Deployment]
	if !exists {
		return fmt.Errorf("unknown deployment %q; %s", binding.Deployment, declaredDeployments(cfg.ModelDeployments))
	}
	return providerAnswers(key, deployment.WithDefaults())
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
		if err := cfg.ModelDeployments[name].WithDefaults().validate(); err != nil {
			return fmt.Errorf("global.model_catalog.deployments.%s: %w", name, err)
		}
	}
	return nil
}

// ValidateModelBindings checks the deployments of a configuration built in
// code and every binding it writes, globally and in each recipe, before any
// model is prepared; a loaded configuration was checked as it loaded.
func ValidateModelBindings(cfg *RouterConfig) error {
	if err := validateModelDeploymentContracts(cfg); err != nil {
		return err
	}
	if err := validateGlobalModelBindingContracts(cfg); err != nil {
		return err
	}
	return validateModelBindingContracts(cfg)
}

// validateModelBindingContracts checks the bindings of every recipe; the
// global ones are checked with the global contracts.
func validateModelBindingContracts(cfg *RouterConfig) error {
	if len(cfg.Recipes) == 0 {
		return validateRecipeBindings(cmp.Or(cfg.RoutingScope, DefaultRecipeName), cfg)
	}
	for i := range cfg.Recipes {
		if err := validateRecipeBindings(cfg.Recipes[i].Name, cfg.ConfigForRecipe(&cfg.Recipes[i])); err != nil {
			return err
		}
	}
	return nil
}
