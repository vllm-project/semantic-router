package config

import (
	"cmp"
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
//   - embeddings.semantic.endpoint becomes an openai_embeddings deployment
//     and embeddings.default;
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
	clone.ClassifierRules = slices.Clone(c.ClassifierRules)
	clone.SafetyRules = cloneSafetyRules(c.SafetyRules)
	clone.ExternalModels = slices.Clone(c.ExternalModels)
	if c.ModalityDetector.Classifier != nil {
		classifier := *c.ModalityDetector.Classifier
		clone.ModalityDetector.Classifier = &classifier
	}
	clone.Recipes = slices.Clone(c.Recipes)
	for i := range clone.Recipes {
		profile := &clone.Recipes[i].Profile
		profile.ModelBindings = cloneModelMap(profile.ModelBindings)
		profile.Signals.ClassifierRules = slices.Clone(profile.Signals.ClassifierRules)
		profile.Signals.SafetyRules = cloneSafetyRules(profile.Signals.SafetyRules)
	}
	clone.modelResourceErrors = nil
	return &clone
}

func cloneSafetyRules(rules []SafetyRule) []SafetyRule {
	cloned := slices.Clone(rules)
	for i := range cloned {
		if cloned[i].Hazard != nil {
			hazard := *cloned[i].Hazard
			cloned[i].Hazard = &hazard
		}
	}
	return cloned
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
	if err := t.normalizeAdmissionKeys(); err != nil {
		return err
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
	first := t.translateModuleBackends()
	t.translateModuleModels()
	first = cmp.Or(first, t.translateRules())
	if err := t.translateEmbeddings(); err != nil {
		first = cmp.Or(first, err)
	}
	t.translatePreference()
	t.translateMemoryRewrite()
	c.ModelResourceWarnings = append(c.ModelResourceWarnings, t.warnings...)
	if len(t.errors) > 0 {
		c.modelResourceErrors = t.errors
	}
	c.modelResourcesTranslated = true
	return first
}

// normalizeAdmissionKeys keys a signal's own admission gate by its consumer,
// accepting the signal type and, with a warning, the consumer name of earlier
// releases.
func (t *translationState) normalizeAdmissionKeys() error {
	for _, key := range sortedAdmissionKeys(t.cfg.ModelAdmission) {
		if _, declared := t.cfg.ModelDeployments[key]; declared {
			continue
		}
		consumer := BindingConsumer(key)
		if consumer != key && admissionDeploymentKeys[consumer] {
			if _, both := t.cfg.ModelAdmission[consumer]; both {
				return fmt.Errorf("global.model_catalog.admission: %s and %s gate the same signal; keep %s", key, consumer, key)
			}
			t.cfg.ModelAdmission[consumer] = t.cfg.ModelAdmission[key]
			delete(t.cfg.ModelAdmission, key)
			continue
		}
		if public := BindingKey(key); public != key && admissionDeploymentKeys[key] {
			t.warn("global.model_catalog.admission."+key, fmt.Sprintf("%s is deprecated; key the signal's admission as %s", key, public))
		}
	}
	return nil
}

func sortedAdmissionKeys(admission map[string]AdmissionConfig) []string {
	keys := make([]string, 0, len(admission))
	for key := range admission {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// publicAdmission keys a signal's own admission gate by its signal type for
// a written configuration.
func publicAdmission(admission map[string]AdmissionConfig) map[string]AdmissionConfig {
	if admission == nil {
		return nil
	}
	public := make(map[string]AdmissionConfig, len(admission))
	for key, gate := range admission {
		if admissionDeploymentKeys[key] {
			key = BindingKey(key)
		}
		public[key] = gate
	}
	return public
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
			for consumer, binding := range bindings {
				if binding.Deployment != name {
					continue
				}
				switch {
				case consumer == "embedding" || binding.Contract == "embedding.v1":
					provider = OpenAIEmbeddingsProvider
				case binding.Adapter != "":
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

// bind adds a global binding for a module's signal unless the configuration
// binds it already, which then wins as it did over the module's model.
func (t *translationState) bind(consumer string, binding ModelBinding) {
	if t.cfg.GlobalModelBindings == nil {
		t.cfg.GlobalModelBindings = map[string]ModelBinding{}
	}
	if _, bound := t.cfg.GlobalModelBindings[consumer]; bound {
		return
	}
	t.cfg.GlobalModelBindings[consumer] = binding
}

// translateModuleBackends reads modules.*.backend as a remote deployment.
func (t *translationState) translateModuleBackends() error {
	// Each module's backend is checked as it was before translation, so a
	// configuration that failed to load still does, with the same error.
	modules := []struct {
		consumer, field string
		backend         **RemoteClassifierBackend
		validate        func(*RouterConfig) error
	}{
		{"domain_classifier", "global.model_catalog.modules.classifier.domain.backend", &t.cfg.CategoryModel.Backend, ValidateCategoryModelBackend},
		{"pii_classifier", "global.model_catalog.modules.classifier.pii.backend", &t.cfg.PIIModel.Backend, validatePIIModelBackendContracts},
		{"prompt_guard", "global.model_catalog.modules.prompt_guard.backend", &t.cfg.PromptGuard.Backend, validatePromptGuardWiring},
		{"complexity", "global.model_catalog.modules.complexity.backend", &t.cfg.ComplexityModel.Backend, validateComplexityModelBackendContracts},
	}
	var first error
	for _, module := range modules {
		backend := *module.backend
		if backend == nil {
			continue
		}
		name, err := t.moduleBackendDeployment(module.field, backend, module.validate)
		*module.backend = nil
		if err != nil {
			t.errors[module.consumer] = err
			first = cmp.Or(first, err)
			continue
		}
		t.bind(module.consumer, ModelBinding{Deployment: name, Contract: backend.Contract, Adapter: backend.Protocol})
		t.warn(module.field, fmt.Sprintf("backend is deprecated; read as bindings.%s: %s", BindingKey(module.consumer), name))
	}
	return first
}

func (t *translationState) moduleBackendDeployment(field string, backend *RemoteClassifierBackend, validate func(*RouterConfig) error) (string, error) {
	if err := validate(t.cfg); err != nil {
		return "", err
	}
	timeout := time.Duration(backend.EffectiveDeadlineMs()) * time.Millisecond
	name, err := t.remoteDeployment(backend.Model, providerOfProtocol(backend.Protocol), timeout)
	if err != nil {
		return "", fmt.Errorf("%s: %w", strings.TrimSuffix(field, ".backend"), err)
	}
	return name, nil
}

// moduleModel is one module's deprecated model field.
type moduleModelField struct {
	consumer string
	field    string
	model    *string
	useCPU   bool
}

func (t *translationState) moduleModelFields() []moduleModelField {
	c := t.cfg
	fields := []moduleModelField{
		{"domain_classifier", "global.model_catalog.modules.classifier.domain.model_id", &c.CategoryModel.ModelID, c.CategoryModel.UseCPU},
		{"pii_classifier", "global.model_catalog.modules.classifier.pii.model_id", &c.PIIModel.ModelID, c.PIIModel.UseCPU},
		{"prompt_guard", "global.model_catalog.modules.prompt_guard.model_id", &c.PromptGuard.ModelID, c.PromptGuard.UseCPU},
		{"fact_check_classifier", "global.model_catalog.modules.hallucination_mitigation.fact_check.model_id", &c.HallucinationMitigation.FactCheckModel.ModelID, c.HallucinationMitigation.FactCheckModel.UseCPU},
		{"feedback_detector", "global.model_catalog.modules.feedback_detector.model_id", &c.FeedbackDetector.ModelID, c.FeedbackDetector.UseCPU},
		{"hallucination_detector", "global.model_catalog.modules.hallucination_mitigation.detector.model_id", &c.HallucinationMitigation.HallucinationModel.ModelID, c.HallucinationMitigation.HallucinationModel.UseCPU},
		{bindingKeySafety, "global.model_catalog.modules.safety.safety.model_id", &c.SafetyModels.Safety.ModelID, c.SafetyModels.Safety.UseCPU},
		{bindingKeyHazard, "global.model_catalog.modules.safety.hazard.model_id", &c.SafetyModels.Hazard.ModelID, c.SafetyModels.Hazard.UseCPU},
	}
	if classifier := c.ModalityDetector.Classifier; classifier != nil {
		fields = append(fields, moduleModelField{"modality_detector", "global.model_catalog.modules.modality_detector.classifier.model_path", &classifier.ModelPath, classifier.UseCPU})
	}
	return fields
}

// translateModuleModels reads each module's model path (from its model_id,
// or the system line its model_ref names) as a deployment and binding.
func (t *translationState) translateModuleModels() {
	for _, module := range t.moduleModelFields() {
		model := strings.TrimSpace(*module.model)
		*module.model = ""
		if model == "" {
			continue
		}
		field := module.field
		if source, ok := t.cfg.legacyModelSources[module.consumer]; ok {
			field = source
		}
		if _, bound := t.cfg.GlobalModelBindings[module.consumer]; bound {
			t.warn(field, fmt.Sprintf("the model %q is deprecated and ignored: bindings.%s already names the deployment", model, BindingKey(module.consumer)))
			continue
		}
		name, err := t.localDeployment(model, module.useCPU)
		if err != nil {
			t.errors[module.consumer] = &DeferredModelError{Err: fmt.Errorf("%s: %w", field, err), Path: model}
			t.warn(field, fmt.Sprintf("model paths are deprecated; %q names no model: %v", model, err))
			continue
		}
		t.bind(module.consumer, ModelBinding{Deployment: name})
		t.warn(field, fmt.Sprintf("model paths are deprecated; read as bindings.%s: %s (%s)", BindingKey(module.consumer), name, describeDeployment(t.cfg.ModelDeployments[name])))
	}
}

func describeDeployment(deployment ModelDeployment) string {
	switch {
	case deployment.IsModelRuntime() && deployment.Artifact != "":
		return fmt.Sprintf("model_runtime %s on %s", deployment.Artifact, deployment.WithDefaults().Device)
	case deployment.Endpoint != "":
		return fmt.Sprintf("%s %s", deployment.Provider, deployment.Endpoint)
	}
	return deployment.Provider
}

// translateRules reads the deprecated model fields of classifier and safety
// rules, at the top level and in each recipe.
func (t *translationState) translateRules() error {
	scopes := []struct {
		path       string
		classifier []ClassifierSignalRule
		safety     []SafetyRule
		bindings   map[string]ModelBinding
	}{{"routing.signals", t.cfg.ClassifierRules, t.cfg.SafetyRules, t.cfg.ModelBindings}}
	for _, recipe := range t.cfg.Recipes {
		scopes = append(scopes, struct {
			path       string
			classifier []ClassifierSignalRule
			safety     []SafetyRule
			bindings   map[string]ModelBinding
		}{fmt.Sprintf("recipes[%s].routing.signals", recipe.Name), recipe.Profile.Signals.ClassifierRules, recipe.Profile.Signals.SafetyRules, recipe.Profile.ModelBindings})
	}
	var first error
	for _, scope := range scopes {
		bound := func(key string) bool {
			_, local := scope.bindings[key]
			_, global := t.cfg.GlobalModelBindings[key]
			return local || global
		}
		for i := range scope.classifier {
			rule := &scope.classifier[i]
			if err := t.translateClassifierRule(scope.path, rule, bound("classifier."+rule.Name)); err != nil {
				t.errors["classifier."+rule.Name] = err
				first = cmp.Or(first, err)
			}
		}
		for i := range scope.safety {
			rule := &scope.safety[i]
			if err := t.translateSafetyRule(scope.path, rule, bound); err != nil {
				t.errors["safety."+rule.Name] = err
				first = cmp.Or(first, err)
			}
		}
	}
	return first
}

func (t *translationState) translateClassifierRule(path string, rule *ClassifierSignalRule, bound bool) error {
	field := fmt.Sprintf("%s.classifiers[%q]", path, rule.Name)
	legacyType, model, modelPath, useCPU := rule.Type, rule.Model, rule.ModelPath, rule.UseCPU
	if legacyType == "" && model == "" && modelPath == "" && !useCPU {
		return nil
	}
	if rule.Deployment != "" {
		if model != "" || modelPath != "" {
			return fmt.Errorf("%s: deployment replaces type, model and model_path; set one", field)
		}
		rule.Type, rule.UseCPU = "", false
		return nil
	}
	if !bound {
		if err := validateLegacyClassifierRule(t.cfg, *rule); err != nil {
			return err
		}
	}
	rule.legacyType = legacyType
	rule.Type, rule.Model, rule.ModelPath, rule.UseCPU = "", "", "", false
	if bound {
		t.warn(field, "type, model and model_path are deprecated and ignored: a classifier binding names the deployment")
		return nil
	}
	switch legacyType {
	case ClassifierSignalTypeLLM, ClassifierSignalTypeSequenceClassifier:
		if model == "" {
			return fmt.Errorf("%s: a %s classifier names its external model", field, legacyType)
		}
		provider := OpenAIChatProvider
		if legacyType == ClassifierSignalTypeSequenceClassifier {
			provider = ClassifyProvider
		}
		external, err := findNamedExternalModel(t.cfg, model)
		if err != nil {
			return fmt.Errorf("%s: %w", field, err)
		}
		timeout := time.Duration(0)
		if provider == OpenAIChatProvider {
			timeout = DurationFromSeconds(external.TimeoutSeconds)
		}
		name, err := t.remoteDeployment(model, provider, timeout)
		if err != nil {
			return fmt.Errorf("%s: %w", field, err)
		}
		rule.Deployment = name
	case ClassifierSignalTypeLocal, "":
		if modelPath == "" {
			if model != "" {
				return fmt.Errorf("%s: a local classifier takes model_path, not model", field)
			}
			if legacyType != "" {
				t.warn(field, "type: local is deprecated; a classifier without a deployment asks the decision model")
			}
			return nil
		}
		name, err := t.localDeployment(modelPath, useCPU)
		if err != nil {
			t.errors["classifier."+rule.Name] = &DeferredModelError{Err: fmt.Errorf("%s: %w", field, err), Path: modelPath}
			t.warn(field, fmt.Sprintf("model_path is deprecated; %q names no model: %v", modelPath, err))
			return nil
		}
		rule.Deployment = name
	default:
		return fmt.Errorf("%s: unsupported type %q (supported: %s, %s, %s)", field, legacyType, ClassifierSignalTypeLocal, ClassifierSignalTypeLLM, ClassifierSignalTypeSequenceClassifier)
	}
	t.warn(field, fmt.Sprintf("type, model and model_path are deprecated; read as deployment: %s", rule.Deployment))
	return nil
}

func (t *translationState) translateSafetyRule(path string, rule *SafetyRule, bound func(string) bool) error {
	field := fmt.Sprintf("%s.safety[%q]", path, rule.Name)
	if rule.Model != "" {
		model := rule.Model
		rule.Model = ""
		switch {
		case rule.Deployment != "":
			return fmt.Errorf("%s: deployment replaces model; set one", field)
		case bound("safety." + rule.Name):
			t.warn(field+".model", "model is deprecated and ignored: a safety binding names the deployment")
		default:
			legacy := ClassifierSignalRule{Name: rule.Name, Type: ClassifierSignalTypeSequenceClassifier, Model: model, Labels: rule.EffectiveLabels()}
			if err := validateSequenceClassifierSignal(t.cfg, legacy); err != nil {
				return fmt.Errorf("safety %q: %w", rule.Name, err)
			}
			name, err := t.remoteDeployment(model, ClassifyProvider, 0)
			if err != nil {
				return fmt.Errorf("%s: %w", field, err)
			}
			rule.Deployment = name
			t.warn(field+".model", fmt.Sprintf("model is deprecated; read as deployment: %s", name))
		}
	}
	if rule.Hazard != nil && rule.Hazard.Model != "" {
		model := rule.Hazard.Model
		rule.Hazard.Model = ""
		switch {
		case rule.Hazard.Deployment != "":
			return fmt.Errorf("%s.hazard: deployment replaces model; set one", field)
		case bound("safety." + rule.Name + ".hazard"):
			t.warn(field+".hazard.model", "model is deprecated and ignored: a hazard binding names the deployment")
		default:
			legacy := ClassifierSignalRule{Name: rule.Name, Type: ClassifierSignalTypeSequenceClassifier, Model: model, Labels: rule.Hazard.Labels}
			if err := validateSequenceClassifierSignal(t.cfg, legacy); err != nil {
				return fmt.Errorf("safety %q hazard: %w", rule.Name, err)
			}
			name, err := t.remoteDeployment(model, ClassifyProvider, 0)
			if err != nil {
				return fmt.Errorf("%s.hazard: %w", field, err)
			}
			rule.Hazard.Deployment = name
			t.warn(field+".hazard.model", fmt.Sprintf("model is deprecated; read as hazard.deployment: %s", name))
		}
	}
	return nil
}

// translateEmbeddings reads embeddings.semantic.endpoint as an
// openai_embeddings deployment.
func (t *translationState) translateEmbeddings() error {
	models := &t.cfg.EmbeddingModels
	if models.UsesRemoteEmbeddingBackend() && t.cfg.EmbeddingDeployment == "" {
		if _, bound := t.cfg.GlobalModelBindings["embedding"]; !bound && strings.TrimSpace(models.Endpoint.BaseURL) != "" {
			endpoint := models.Endpoint
			deployment := ModelDeployment{Provider: OpenAIEmbeddingsProvider, Endpoint: strings.TrimRight(strings.TrimSpace(endpoint.BaseURL), "/"), Model: endpoint.Model}
			if endpoint.APIKeyEnv != "" {
				deployment.Auth = &DeploymentAuth{APIKeyEnv: endpoint.APIKeyEnv}
			}
			if endpoint.TimeoutSeconds > 0 {
				deployment.TimeoutMs = endpoint.TimeoutSeconds * 1000
			}
			if endpoint.MaxRetries > 0 {
				retries := endpoint.MaxRetries
				deployment.MaxRetries = &retries
			}
			if endpoint.MaxResponseBytes > 0 {
				deployment.Limits = &DeploymentLimits{MaxResponseBytes: endpoint.MaxResponseBytes}
			}
			if endpoint.Dimensions > 0 {
				deployment.Capabilities = &DeploymentCapabilities{Dimensions: endpoint.Dimensions}
			}
			name := t.uniqueName("remote-embedding")
			deployment.translated = true
			t.cfg.ModelDeployments[name] = deployment
			t.cfg.EmbeddingDeployment = name
			t.warn("global.model_catalog.embeddings.semantic.endpoint", fmt.Sprintf(
				"the embedding endpoint is deprecated; read as deployments.%s {provider: openai_embeddings, endpoint: %s, model: %q} and embeddings.default: %s",
				name, deployment.Endpoint, deployment.Model, name))
		}
	}
	if name := t.cfg.EmbeddingDeployment; name != "" {
		if bound, ok := t.cfg.GlobalModelBindings["embedding"]; ok && bound.Deployment != name {
			return fmt.Errorf("global.model_catalog.embeddings.default %q and bindings.embedding %q name different deployments; keep embeddings.default", name, bound.Deployment)
		}
		t.bind("embedding", ModelBinding{Deployment: name})
	}
	return nil
}

// translatePreference binds the preference signal to the external[] model
// with model_role: preference, which it used to find by role.
func (t *translationState) translatePreference() {
	if _, bound := t.cfg.GlobalModelBindings["preference"]; bound {
		return
	}
	if contrastive := t.cfg.PreferenceModel.UseContrastive; contrastive != nil && *contrastive {
		return
	}
	for _, external := range t.cfg.ExternalModels {
		if external.ModelRole != ModelRolePreference || external.Name == "" {
			continue
		}
		name, err := t.remoteDeployment(external.Name, OpenAIChatProvider, DurationFromSeconds(external.TimeoutSeconds))
		if err != nil {
			return
		}
		t.bind("preference", ModelBinding{Deployment: name})
		t.warn(fmt.Sprintf("global.model_catalog.external[%q].model_role", external.Name), fmt.Sprintf("finding the preference model by model_role is deprecated; read as bindings.preference: %s", name))
		return
	}
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
