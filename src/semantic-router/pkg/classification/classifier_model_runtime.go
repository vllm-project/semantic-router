package classification

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// classifierModelRuntime is preparation-only state. Requests use typed handles;
// they never resolve catalog paths or scan another recipe's configuration.
type classifierModelRuntime struct {
	runtime *native.Runtime
	plan    *config.ModelBindingPlan
	cfg     *config.RouterConfig
	recipe  config.RecipeName
}

func newClassifierModelRuntime(cfg *config.RouterConfig, runtime *native.Runtime) (*classifierModelRuntime, error) {
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	if runtime == nil {
		runtime = native.New(nil)
	}
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = config.DefaultRecipeName
	}
	models := &classifierModelRuntime{runtime: runtime, plan: plan, cfg: cfg, recipe: recipe}
	if err := models.projectBindings(); err != nil {
		return nil, err
	}
	return models, nil
}

// localSpec materializes the existing canonical module default when no recipe
// override is declared. A module's name is its default binding, not its physical
// identity: native preparation fingerprints the artifact and execution options.
func (m *classifierModelRuntime) localSpec(name, artifact, adapter, contract string, useCPU bool) config.ResolvedModelBinding {
	if spec, ok := m.plan.Lookup(m.recipe, name); ok {
		spec.Deployment.Artifact = config.ResolveModelPath(spec.Deployment.Artifact)
		if spec.Binding.Head != "" {
			spec.Binding.Head = config.ResolveModelPath(spec.Binding.Head)
		}
		return spec
	}
	provider, device := config.DefaultModelExecution(useCPU)
	return config.ResolvedModelBinding{
		Recipe: m.recipe, Name: name,
		Binding:    config.ModelBinding{Deployment: name, Adapter: adapter, Contract: contract},
		Deployment: config.ModelDeployment{Artifact: config.ResolveModelPath(artifact), Provider: provider, Device: device, Precision: "native", Input: config.ModelInputBudget{Overflow: "truncate"}},
		Admission:  m.cfg.ModelAdmission[name],
	}
}

type ownedSequenceBackend struct {
	runtime        *native.Runtime
	spec           config.ResolvedModelBinding
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
	runtime *native.Runtime
	spec    config.ResolvedModelBinding
	labels  []string
	mu      sync.RWMutex
	handle  *binding.Resolved[string, tasks.TokenClassificationResult]
	closed  bool
}

func (b *ownedTokenBackend) Init(_ string, _ bool, _ int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return binding.ErrClosed
	}
	if b.handle != nil {
		return nil
	}
	handle, err := b.runtime.Tokens(context.Background(), b.spec)
	if err != nil {
		return fmt.Errorf("prepare %s/%s: %w", b.spec.Recipe, b.spec.Name, err)
	}
	if err := validateNativeLabelOrder(handle.Capability().Labels, b.labels, nil); err != nil {
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
	return &classifierModelRuntime{runtime: native.New(nil), cfg: &config.RouterConfig{}, recipe: config.DefaultRecipeName}
}

func consumerModelRuntime(models []*classifierModelRuntime) *classifierModelRuntime {
	if len(models) > 0 && models[0] != nil {
		return models[0]
	}
	return standaloneModelRuntime()
}
