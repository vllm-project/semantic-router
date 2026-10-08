package classification

import (
	"context"
	"fmt"
	"path/filepath"
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
	plan    *config.ModelBindingPlan
	cfg     *config.RouterConfig
	recipe  config.RecipeName
	// Contrastive input policy is captured before the module window defaults.
	jailbreakContrastiveFullContext *bool
}

func newClassifierModelRuntime(cfg *config.RouterConfig, options RecipeRuntimeOptions) (*classifierModelRuntime, error) {
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
	models := &classifierModelRuntime{runtime: runtime, plan: plan, cfg: cfg, recipe: recipe}
	if err := models.projectBindings(); err != nil {
		return nil, err
	}
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
	selectedPath, err := filepath.Abs(selected)
	if err != nil {
		return false
	}
	registeredPath, err := filepath.Abs(defaultPath)
	return err == nil && selectedPath == registeredPath
}

// localSpec resolves a consumer's binding: the recipe's declared binding, else
// the implicit model_runtime deployment of the module's model (config owns its
// name and identity, so the manager already runs it in the device's process).
// The module's input budget truncates by default; Vela Halu reads the whole
// grounded input and rejects longer ones.
func (m *classifierModelRuntime) localSpec(name, artifact, adapter, contract string, useCPU bool, maxTokens ...int) (config.ResolvedModelBinding, error) {
	if spec, ok := m.plan.Lookup(m.recipe, name); ok {
		return spec, nil
	}
	if deploymentName, deployment, ok, err := m.cfg.ImplicitTaskDeployment(name); ok && err == nil && deploymentName == m.cfg.DecisionModel {
		return config.ResolvedModelBinding{Recipe: m.recipe, Name: name,
			Binding:    config.ModelBinding{Deployment: deploymentName, Adapter: adapter, Contract: contract},
			Deployment: deployment, Admission: m.cfg.ModelAdmission[deploymentName]}, nil
	}
	limit := 0
	if len(maxTokens) > 0 {
		limit = maxTokens[0]
	}
	overflow := "truncate"
	if model := config.GetModelByPath(artifact); model != nil && model.DefaultAdapter == "vela_halu" {
		adapter = model.DefaultAdapter
		if limit == 0 {
			limit = model.MaxContextLength
		}
		overflow = "reject"
	}
	deployment, err := config.ImplicitModelRuntimeDeployment(artifact, useCPU)
	if err != nil {
		return config.ResolvedModelBinding{}, fmt.Errorf("%s/%s: %w", m.recipe, name, err)
	}
	// The module's own model already runs under its config-owned name; any
	// other artifact gets a recipe-scoped deployment of its own.
	deploymentName := config.ImplicitDeploymentPrefix + string(m.recipe) + "/" + name
	if moduleName, module, ok, moduleErr := m.cfg.ImplicitTaskDeployment(name); ok && moduleErr == nil &&
		module.Artifact == deployment.Artifact && module.Revision == deployment.Revision && module.Device == deployment.Device {
		deploymentName = moduleName
	}
	deployment.Input = config.ModelInputBudget{MaxTokens: limit, Overflow: overflow}
	return config.ResolvedModelBinding{
		Recipe: m.recipe, Name: name,
		Binding:    config.ModelBinding{Deployment: deploymentName, Adapter: adapter, Contract: contract},
		Deployment: deployment.WithDefaults(),
		Admission:  m.cfg.ModelAdmission[name],
	}, nil
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

func standaloneModelRuntime() *classifierModelRuntime {
	return &classifierModelRuntime{runtime: serving.New(nil, nil), cfg: &config.RouterConfig{}, recipe: config.DefaultRecipeName}
}

func consumerModelRuntime(models []*classifierModelRuntime) *classifierModelRuntime {
	if len(models) > 0 && models[0] != nil {
		return models[0]
	}
	return standaloneModelRuntime()
}
