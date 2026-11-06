package classification

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// classifierModelRuntime is preparation-only state. Requests use typed handles;
// they never resolve catalog paths or scan another recipe's configuration.
type classifierModelRuntime struct {
	runtime *serving.Runtime
	// systemOne asks the systemone deployments' remote APIs.
	systemOne modelservice.Remote
	plan      *config.ModelBindingPlan
	cfg       *config.RouterConfig
	recipe    config.RecipeName
	// Contrastive input policy is captured before the module window defaults.
	jailbreakContrastiveFullContext *bool
}

func newClassifierModelRuntime(cfg *config.RouterConfig, options RecipeRuntimeOptions) (*classifierModelRuntime, error) {
	cfg = config.WithModelResources(cfg)
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	runtime := options.Runtime
	if runtime == nil {
		runtime = serving.New(nil, nil)
	}
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = config.DefaultRecipeName
	}
	models := &classifierModelRuntime{runtime: runtime, systemOne: options.SystemOne, plan: plan, cfg: cfg, recipe: recipe}
	fullContext := (&Classifier{Config: models.cfg}).hasLongContextClassifier(config.SignalTypeJailbreak)
	models.jailbreakContrastiveFullContext = &fullContext
	if err := models.resolveDefaultJailbreakWindow(); err != nil {
		return nil, err
	}
	if err := models.resolveDefaultPIIWindow(); err != nil {
		return nil, err
	}
	return models, nil
}

// decider answers decision signals through the generation's deployments.
func (m *classifierModelRuntime) decider() (modelservice.Decider, bool) {
	if m == nil || m.runtime == nil {
		return nil, false
	}
	decider, ok := m.runtime.Services().(modelservice.Decider)
	return decider, ok
}

// Match registry aliases and equivalent local paths without treating an
// unrelated directory with the same basename as the default artifact.
func isDefaultModelArtifact(selected, defaultPath string) bool {
	if model := config.GetModelByPath(filepath.Clean(selected)); model != nil {
		return model.LocalPath == defaultPath
	}
	if model := config.RegisteredModel(selected); model != nil {
		return model.LocalPath == defaultPath
	}
	selectedPath, err := filepath.Abs(selected)
	if err != nil {
		return false
	}
	registeredPath, err := filepath.Abs(defaultPath)
	return err == nil && selectedPath == registeredPath
}

// admission is a deployment's admission, else the admission an earlier
// release keyed by the consumer's name.
func (m *classifierModelRuntime) admission(deployment, consumer string) config.AdmissionConfig {
	if admission, keyed := m.cfg.ModelAdmission[deployment]; keyed {
		return admission
	}
	return m.cfg.ModelAdmission[consumer]
}

// translatedLocalModel reports whether consumer resolves to a model_runtime
// deployment that a deprecated module path declared for the model at
// defaultPath.
func (m *classifierModelRuntime) translatedLocalModel(consumer, defaultPath string) bool {
	spec, ok, err := m.task(consumer)
	return ok && err == nil && spec.Deployment.Translated() && spec.Deployment.IsModelRuntime() &&
		isDefaultModelArtifact(spec.Deployment.Artifact, defaultPath)
}

// task resolves the deployment that answers consumer in the runtime's
// recipe: the deployment a rule or binding names (from the plan), else the
// decision model, by name (config.RouterConfig.ResolveTask). ok is false
// when nothing answers it.
func (m *classifierModelRuntime) task(consumer string) (config.ResolvedModelBinding, bool, error) {
	if spec, ok := m.plan.Lookup(m.recipe, consumer); ok {
		spec.Admission = m.admission(spec.Binding.Deployment, consumer)
		return spec, true, nil
	}
	resolved, ok, err := m.cfg.ResolveTask(consumer)
	if !ok || err != nil {
		return config.ResolvedModelBinding{}, ok, err
	}
	binding := config.ModelBinding{Deployment: resolved.Name}
	if resolved.Explicit() {
		binding = config.ImpliedBinding(consumer, resolved.Binding, resolved.Deployment)
		binding.Deployment = resolved.Name
	}
	return config.ResolvedModelBinding{
		Recipe: m.recipe, Name: consumer,
		Binding:    binding,
		Deployment: resolved.Deployment, Admission: m.admission(resolved.Name, consumer),
	}, true, nil
}

// localSpec resolves the binding a consumer's model_runtime or systemone
// deployment answers through. A binding the configuration declares is used
// as declared; the decision model answers with the consumer's adapter and
// contract. A deployment a deprecated module path declared answers as the
// module's own did: with the consumer's adapter and contract, and the
// module's input budget, which truncates by default; Vela Halu reads the
// whole grounded input and rejects longer ones.
func (m *classifierModelRuntime) localSpec(name, adapter, contract string, maxTokens ...int) (config.ResolvedModelBinding, error) {
	spec, ok, err := m.task(name)
	if err != nil {
		return config.ResolvedModelBinding{}, fmt.Errorf("%s/%s: %w", m.recipe, name, err)
	}
	if !ok {
		return config.ResolvedModelBinding{}, fmt.Errorf("%s/%s: no model is configured", m.recipe, name)
	}
	if !spec.Deployment.Translated() {
		if spec.Binding.Contract == "" {
			spec.Binding.Adapter, spec.Binding.Contract = adapter, contract
		}
		return spec, nil
	}
	if err := spec.Deployment.MissingLocalModel(); err != nil {
		return config.ResolvedModelBinding{}, fmt.Errorf("%s/%s: %w", m.recipe, name, err)
	}
	limit := 0
	if len(maxTokens) > 0 {
		limit = maxTokens[0]
	}
	overflow := "truncate"
	if model := config.RegisteredModel(spec.Deployment.Artifact); model != nil && model.DefaultAdapter == "vela_halu" {
		adapter = model.DefaultAdapter
		if limit == 0 {
			limit = model.MaxContextLength
		}
		overflow = "reject"
	}
	spec.Binding.Adapter, spec.Binding.Contract = adapter, contract
	spec.Deployment.Input = config.ModelInputBudget{MaxTokens: limit, Overflow: overflow}
	spec.Deployment = spec.Deployment.WithDefaults()
	spec.Admission = m.admission(spec.Binding.Deployment, name)
	return spec, nil
}

// remoteTask reports whether a consumer's deployment answers over a remote
// classify or chat API rather than a model_runtime or systemone deployment.
func remoteTask(spec config.ResolvedModelBinding) bool {
	switch spec.Deployment.Provider {
	case config.ClassifyProvider, config.OpenAIChatProvider, config.LegacyHTTPProvider:
		return true
	}
	return false
}

// ownedSequenceBackend prepares its binding at Init; err is a resolution
// failure that Init reports, so a consumer nobody uses never fails a build.
type ownedSequenceBackend struct {
	runtime        *serving.Runtime
	spec           config.ResolvedModelBinding
	err            error
	labels         []string
	normalizeLabel func(string) string
	mu             sync.RWMutex
	handle         *binding.Resolved[string, tasks.LabelDistribution]
	closed         bool
}

func (b *ownedSequenceBackend) Init(_ string, _ bool, classes ...int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return binding.ErrClosed
	}
	if b.err != nil {
		return b.err
	}
	if b.handle != nil {
		return nil
	}
	handle, err := b.runtime.Sequence(context.Background(), b.spec)
	if err != nil {
		return fmt.Errorf("prepare %s/%s: %w", b.spec.Recipe, b.spec.Name, err)
	}
	if len(classes) > 0 && classes[0] > 0 && len(handle.Capability().Labels) != classes[0] {
		_ = handle.Close()
		return fmt.Errorf("prepared labels do not match consumer mapping: model=%d mapping=%d", len(handle.Capability().Labels), classes[0])
	}
	if err := validateNativeLabelOrder(handle.Capability().Labels, b.labels, b.normalizeLabel); err != nil {
		_ = handle.Close()
		return fmt.Errorf("prepare %s/%s: %w", b.spec.Recipe, b.spec.Name, err)
	}
	b.handle = handle
	return nil
}

// readsWholeText reports whether the prepared binding asks a Vela 2.0 model
// its signal's question, which reads a whole text however long it is.
func (b *ownedSequenceBackend) readsWholeText() bool {
	if b == nil {
		return false
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.handle != nil && b.handle.Capability().Question != ""
}

func (b *ownedSequenceBackend) Classify(ctx context.Context, text string) (tasks.LabelDistribution, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || b.handle == nil {
		return tasks.LabelDistribution{}, binding.ErrClosed
	}
	return b.handle.Call(ctx, string(b.spec.Recipe), text)
}

func (b *ownedSequenceBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	if b.handle == nil {
		return nil
	}
	return b.handle.Close()
}

type ownedCategoryBackend struct{ *ownedSequenceBackend }

func (b ownedCategoryBackend) Classify(ctx context.Context, text string) (tasks.ClassResult, error) {
	result, err := b.ownedSequenceBackend.Classify(ctx, text)
	if err != nil {
		return tasks.ClassResult{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	return tasks.ClassResult{Class: class, Confidence: confidence}, nil
}

func (b ownedCategoryBackend) ClassifyWithProbabilities(ctx context.Context, text string) (tasks.ClassResultWithProbs, error) {
	result, err := b.ownedSequenceBackend.Classify(ctx, text)
	if err != nil {
		return tasks.ClassResultWithProbs{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	return tasks.ClassResultWithProbs{Class: class, Confidence: confidence, Probabilities: result.Probabilities, NumClasses: len(result.Probabilities)}, nil
}

type ownedTokenBackend struct {
	runtime *serving.Runtime
	spec    config.ResolvedModelBinding
	err     error
	labels  []string
	mu      sync.RWMutex
	handle  *binding.Resolved[string, tasks.TokenClassificationResult]
	closed  bool
}

// readsWholeText reports whether the prepared binding asks a decision model's
// ready-made span question, which reads a whole text however long it is.
func (b *ownedTokenBackend) readsWholeText() bool {
	if b == nil {
		return false
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.handle != nil && (b.handle.Capability().Preset != "" || b.handle.Capability().Question != "")
}

func (b *ownedTokenBackend) Init(_ string, _ bool, _ int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return binding.ErrClosed
	}
	if b.err != nil {
		return b.err
	}
	if b.handle != nil {
		return nil
	}
	handle, err := b.runtime.Tokens(context.Background(), b.spec)
	if err != nil {
		return fmt.Errorf("prepare %s/%s: %w", b.spec.Recipe, b.spec.Name, err)
	}
	if err := validateTokenLabels(handle.Capability(), b.labels); err != nil {
		_ = handle.Close()
		return fmt.Errorf("prepare %s/%s: %w", b.spec.Recipe, b.spec.Name, err)
	}
	b.handle = handle
	return nil
}

func (b *ownedTokenBackend) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || b.handle == nil {
		return tasks.TokenClassificationResult{}, binding.ErrClosed
	}
	return b.handle.Call(ctx, string(b.spec.Recipe), text)
}

func (b *ownedTokenBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	if b.handle == nil {
		return nil
	}
	return b.handle.Close()
}

// withModule is the runtime for a module constructed from its own
// configuration, apply's change, rather than from the runtime's: a model the
// module names in a deprecated field resolves through the same translation
// as a loaded configuration's. A module without one is the runtime's.
func (m *classifierModelRuntime) withModule(model string, apply func(*config.RouterConfig)) *classifierModelRuntime {
	if strings.TrimSpace(model) == "" {
		return m
	}
	scoped := *m
	scoped.cfg = config.WithModule(m.cfg, apply)
	scoped.plan = nil
	return &scoped
}

func standaloneModelRuntime() *classifierModelRuntime {
	return &classifierModelRuntime{runtime: serving.New(nil, nil), cfg: &config.RouterConfig{}, recipe: config.DefaultRecipeName}
}

func consumerModelRuntime(models []*classifierModelRuntime) *classifierModelRuntime {
	if len(models) > 0 && models[0] != nil {
		return models[0]
	}
	return standaloneModelRuntime()
}
