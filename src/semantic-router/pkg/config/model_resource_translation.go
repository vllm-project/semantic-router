package config

import (
	"fmt"
	"net/url"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

// The deprecated ways to choose a model are read, at load, as named
// deployments and bindings, and the fields they came from are cleared, so
// the resolver (model_resolution.go) sees one form:
//
//   - global.model_catalog.system lines, a module's model_id and model_ref,
//     and modality_detector.classifier.model_path become a deployment of the
//     model and a binding of the module's signal;
//   - a module's backend and its external[] model, an external[] model a
//     classifier or safety rule names, and a provider: http deployment
//     become classify, openai_chat or openai_embeddings deployments;
//   - a classifier rule's type, model and model_path, and a safety rule's
//     (or hazard part's) model, become the rule's deployment;
//   - an inline access_key becomes the deployment's credential.
//
// A model path names the declared deployment that serves the same model on
// the same device when there is one, else a new deployment named after the
// model. A model that cannot be resolved fails the consumer that runs it, as
// it did before, rather than the configuration.

const warningDeprecatedModelField = "deprecated_model_field"

// translationState collects one configuration's translation.
type translationState struct {
	cfg      *RouterConfig
	warnings []ConfigWarning
	// errors fail the consumers whose deprecated model fields cannot be
	// translated: a DeferredModelError when the model cannot be resolved,
	// else the configuration error.
	errors map[string]error
	// remote maps a translated remote deployment's identity to its name.
	remote map[string]string
}

func (t *translationState) warn(field, message string) {
	t.warnings = append(t.warnings, ConfigWarning{Code: warningDeprecatedModelField, Field: field, Message: message})
}

// withModelResources returns the configuration with its deprecated model
// fields translated. A configuration the loader returned already is.
func (c *RouterConfig) withModelResources() *RouterConfig {
	if c == nil || c.modelResourcesTranslated {
		return c
	}
	translated := c.cloneModelResources()
	_ = translated.translateModelResources()
	return translated
}

// WithModelResources is withModelResources for callers outside the package
// that read a configuration built in code.
func WithModelResources(cfg *RouterConfig) *RouterConfig { return cfg.withModelResources() }

// WithModule returns a copy of the configuration that also carries a module
// configured in code, apply's change, translated as a loaded configuration
// would be.
func WithModule(cfg *RouterConfig, apply func(*RouterConfig)) *RouterConfig {
	if cfg == nil {
		cfg = &RouterConfig{}
	}
	errs := cfg.modelResourceErrors
	scoped := cfg.cloneModelResources()
	apply(scoped)
	scoped.modelResourcesTranslated = false
	scoped.ModelResourceWarnings = nil
	_ = scoped.translateModelResources()
	for consumer, err := range errs {
		if _, replaced := scoped.modelResourceErrors[consumer]; !replaced {
			if scoped.modelResourceErrors == nil {
				scoped.modelResourceErrors = map[string]error{}
			}
			scoped.modelResourceErrors[consumer] = err
		}
	}
	return scoped
}

// cloneModelResources copies the configuration with every map and rule list
// translation writes to.
func (c *RouterConfig) cloneModelResources() *RouterConfig {
	clone := *c
	clone.ModelDeployments = cloneModelMap(c.ModelDeployments)
	if clone.ModelDeployments == nil {
		clone.ModelDeployments = map[string]ModelDeployment{}
	}
	clone.GlobalModelBindings = cloneModelMap(c.GlobalModelBindings)
	clone.ModelBindings = cloneModelMap(c.ModelBindings)
	clone.scopedBindings = cloneModelMap(c.scopedBindings)
	clone.ModelAdmission = cloneAdmissionMap(c.ModelAdmission)
	clone.ExternalModels = slices.Clone(c.ExternalModels)
	clone.Recipes = slices.Clone(c.Recipes)
	for i := range clone.Recipes {
		profile := &clone.Recipes[i].Profile
		profile.ModelBindings = cloneModelMap(profile.ModelBindings)
	}
	clone.modelResourceErrors = nil
	return &clone
}

// translateModelResources translates the configuration in place.
func (c *RouterConfig) translateModelResources() error {
	t := &translationState{cfg: c, errors: map[string]error{}, remote: map[string]string{}}
	if c.ModelDeployments == nil {
		c.ModelDeployments = map[string]ModelDeployment{}
	}
	var err error
	if c.GlobalModelBindings, err = normalizeBindingKeys("global.model_catalog.bindings", c.GlobalModelBindings); err != nil {
		return err
	}
	if c.ModelBindings, err = normalizeBindingKeys("routing.model_bindings", c.ModelBindings); err != nil {
		return err
	}
	if c.scopedBindings != nil {
		if c.scopedBindings, err = normalizeBindingKeys("routing.model_bindings", c.scopedBindings); err != nil {
			return err
		}
	}
	for i := range c.Recipes {
		recipe := &c.Recipes[i]
		if recipe.Profile.ModelBindings, err = normalizeBindingKeys(fmt.Sprintf("recipes[%s].routing.model_bindings", recipe.Name), recipe.Profile.ModelBindings); err != nil {
			return err
		}
	}
	for _, name := range sortedModelKeys(c.ModelDeployments) {
		deployment := c.ModelDeployments[name]
		path := "global.model_catalog.deployments." + name
		if len(deployment.QuestionTypes) > 0 {
			t.warn(path+".question_types", "question_types is deprecated; write it as capabilities.question_types")
		}
		if deployment.ServedName != "" {
			t.warn(path+".served_name", "served_name is deprecated; write it as model")
		}
		for i, replica := range deployment.Replicas {
			if replica.ServedName != "" {
				t.warn(fmt.Sprintf("%s.replicas[%d].served_name", path, i), "served_name is deprecated; write it as model")
			}
		}
	}
	// A step that fails records the consumers it fails and the rest still
	// run, so a configuration built in code fails only those consumers; a
	// loaded one fails with the first error.
	if err := t.translateHTTPDeployments(); err != nil {
		return err
	}
	t.translateMemoryRewrite()
	c.ModelResourceWarnings = append(c.ModelResourceWarnings, t.warnings...)
	if len(t.errors) > 0 {
		c.modelResourceErrors = t.errors
	}
	c.modelResourcesTranslated = true
	return nil
}

// uniqueName returns base, or base with a numeric suffix when another
// deployment already has that name.
func (t *translationState) uniqueName(base string) string {
	base = strings.Trim(strings.ReplaceAll(strings.TrimSpace(base), "/", "-"), "-")
	if base == "" {
		base = "model"
	}
	base = strings.TrimPrefix(base, ImplicitDeploymentPrefix)
	name := base
	for i := 2; ; i++ {
		if _, taken := t.cfg.ModelDeployments[name]; !taken {
			return name
		}
		name = fmt.Sprintf("%s-%d", base, i)
	}
}

// localDeployment is the deployment of a deprecated model path on the
// device useCPU selects: a declared deployment serving the same model, else
// a new one. The error says why the path names no model.
func (t *translationState) localDeployment(model string, useCPU bool) (string, error) {
	deployment, err := modelRuntimeDeploymentOf(model, useCPU)
	if err != nil {
		return "", err
	}
	deployment = deployment.WithDefaults()
	for _, name := range sortedModelKeys(t.cfg.ModelDeployments) {
		declared := t.cfg.ModelDeployments[name].WithDefaults()
		if declared.IsModelRuntime() && declared.Managed() && len(declared.Replicas) == 0 && !declared.HasEngineArgs() &&
			declared.Artifact == deployment.Artifact && declared.Revision == deployment.Revision && declared.Device == deployment.Device {
			return name, nil
		}
	}
	base := filepath.Base(strings.TrimRight(model, "/"))
	if spec := GetModelByPath(model); spec != nil {
		base = filepath.Base(spec.LocalPath)
	}
	if deployment.Device != "cpu" {
		base += "-" + deployment.Device
	}
	name := t.uniqueName(base)
	deployment.translated = true
	deployment.legacyPath = strings.TrimSpace(model)
	t.cfg.ModelDeployments[name] = deployment
	return name, nil
}

// remoteDeployment is the deployment of an external[] model asked through
// provider, with timeout (zero: the provider's default).
func (t *translationState) remoteDeployment(externalName, provider string, timeout time.Duration) (string, error) {
	external, err := findNamedExternalModel(t.cfg, externalName)
	if err != nil {
		return "", err
	}
	if timeout == 0 {
		timeout = DurationFromSeconds(external.TimeoutSeconds)
	}
	deployment := deploymentFromExternal(*external, provider, timeout)
	identity := fmt.Sprintf("%s\x00%s\x00%s\x00%d\x00%s", provider, deployment.Endpoint, deployment.Model, deployment.TimeoutMs, external.AccessKey)
	if name, ok := t.remote[identity]; ok {
		return name, nil
	}
	name := t.uniqueName(externalName)
	t.cfg.ModelDeployments[name] = deployment
	t.remote[identity] = name
	t.warnExternal(externalName, name, deployment)
	return name, nil
}

func (t *translationState) warnExternal(externalName, name string, deployment ModelDeployment) {
	auth := ""
	if deployment.Auth.HasInlineKey() {
		auth = "; its inline access_key is deprecated: set auth.api_key_env or auth.api_key_file"
	}
	t.warn(fmt.Sprintf("global.model_catalog.external[%q]", externalName), fmt.Sprintf(
		"external models are deprecated; read as deployments.%s {provider: %s, endpoint: %s, model: %q}%s",
		name, deployment.Provider, deployment.Endpoint, deployment.Model, auth))
}

// ExternalModelDeployment is a deprecated external[] model as a deployment
// of provider, its llm_timeout_seconds as the timeout: the one place a
// llm_endpoint becomes a URL.
func ExternalModelDeployment(external ExternalModelConfig, provider string) ModelDeployment {
	return deploymentFromExternal(external, provider, DurationFromSeconds(external.TimeoutSeconds))
}

// deploymentFromExternal is an external[] model as a deployment of provider.
func deploymentFromExternal(external ExternalModelConfig, provider string, timeout time.Duration) ModelDeployment {
	endpoint := LegacyEndpointURL(external.ModelEndpoint.Address, external.ModelEndpoint.Port, external.ModelEndpoint.Protocol)
	if provider == OpenAIChatProvider {
		if parsed, err := url.Parse(endpoint); err == nil && strings.Trim(parsed.Path, "/") == "" {
			endpoint = strings.TrimRight(endpoint, "/") + "/v1"
		}
	}
	deployment := ModelDeployment{Provider: provider, Endpoint: endpoint, Model: external.ModelName}
	if timeout > 0 {
		deployment.TimeoutMs = int(timeout / time.Millisecond)
	}
	if external.AccessKey != "" {
		deployment.Auth = legacyInlineAuth(external.AccessKey)
	}
	if external.MaxRequestBytes > 0 || external.MaxResponseBytes > 0 {
		deployment.Limits = &DeploymentLimits{MaxRequestBytes: external.MaxRequestBytes, MaxResponseBytes: external.MaxResponseBytes}
	}
	copied := external
	deployment.legacyExternal = &copied
	return deployment
}

// LegacyExternal is the external[] model a translated deployment came from,
// whose chat settings (parser, threshold, generation, reasoning, prompt
// template) it keeps; nil for a deployment the configuration declares.
func (d ModelDeployment) LegacyExternal() *ExternalModelConfig { return d.legacyExternal }

// Translated reports whether a deprecated model path declared the
// deployment, whose input policy is then its consumer module's.
func (d ModelDeployment) Translated() bool { return d.translated }

// LegacyModelPath is the model path a deprecated field named for a
// translated model_runtime deployment, which the Router downloads when its
// registry maps it to a repository; empty for any other deployment.
func (d ModelDeployment) LegacyModelPath() string { return d.legacyPath }

// providerOfProtocol is the provider a deprecated backend protocol or
// binding adapter names.
func providerOfProtocol(protocol string) string {
	if protocol == RemoteClassifierProtocolHTTPChat {
		return OpenAIChatProvider
	}
	return ClassifyProvider
}

// translateHTTPDeployments rewrites provider: http deployments as the
// provider their bindings ask through.
func (t *translationState) translateHTTPDeployments() error {
	for _, name := range sortedModelKeys(t.cfg.ModelDeployments) {
		deployment := t.cfg.ModelDeployments[name]
		if deployment.Provider != LegacyHTTPProvider {
			continue
		}
		provider := ClassifyProvider
		for _, bindings := range t.allBindings() {
			for _, binding := range bindings {
				if binding.Deployment == name && binding.Adapter != "" {
					provider = providerOfProtocol(binding.Adapter)
				}
			}
		}
		external, err := findNamedExternalModel(t.cfg, deployment.ExternalModel)
		if err != nil {
			return fmt.Errorf("global.model_catalog.deployments.%s: %w", name, err)
		}
		timeout := time.Duration(0)
		if provider == OpenAIEmbeddingsProvider || provider == OpenAIChatProvider {
			timeout = DurationFromSeconds(external.TimeoutSeconds)
		}
		translated := deploymentFromExternal(*external, provider, timeout)
		translated.Input = deployment.Input
		t.cfg.ModelDeployments[name] = translated
		t.warn("global.model_catalog.deployments."+name, fmt.Sprintf(
			"provider: http with external_model is deprecated; read as {provider: %s, endpoint: %s, model: %q}", provider, translated.Endpoint, translated.Model))
	}
	return nil
}

// allBindings lists the global, top-level and recipe binding maps.
func (t *translationState) allBindings() []map[string]ModelBinding {
	all := []map[string]ModelBinding{t.cfg.GlobalModelBindings, t.cfg.ModelBindings}
	for _, recipe := range t.cfg.Recipes {
		all = append(all, recipe.Profile.ModelBindings)
	}
	return all
}

// ModelResourceTranslation lists the deployment each translated consumer
// came to, for diagnostics.
func (c *RouterConfig) ModelResourceTranslation() []ConfigWarning {
	if c == nil {
		return nil
	}
	warnings := slices.Clone(c.ModelResourceWarnings)
	sort.SliceStable(warnings, func(i, j int) bool { return warnings[i].Field < warnings[j].Field })
	return warnings
}
