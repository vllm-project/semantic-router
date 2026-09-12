package classification

import (
	"fmt"
	"runtime"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type classifierOptionBuilder struct {
	cfg              *config.RouterConfig
	models           *classifierModelRuntime
	embeddingSet     *embedding.Set
	ownsEmbeddingSet bool
	options          []option
	providerInitOnce sync.Once
	provider         embedding.Provider
	providerErr      error
}

func newClassifierOptionBuilder(cfg *config.RouterConfig, options []option) *classifierOptionBuilder {
	return &classifierOptionBuilder{cfg: cfg, options: options}
}

func (b *classifierOptionBuilder) build(categoryMapping *CategoryMapping) ([]option, error) {
	if CurrentNativeBackendCapabilities().Name != "openvino" {
		if err := b.prepareEmbeddingSet(); err != nil {
			return nil, err
		}
	}
	if b.cfg.RoutingScope == config.DefaultRecipeName && !b.cfg.IsRecipeReachableForRouting(config.DefaultRecipeName) {
		return b.options, nil
	}
	steps := []func() (option, error){
		b.buildKeywordClassifierOption,
		b.buildEmbeddingClassifierOption,
		b.buildContextClassifierOption,
		b.buildStructureClassifierOption,
		b.buildReaskClassifierOption,
		b.buildComplexityClassifierOption,
		b.buildContrastiveJailbreakClassifiersOption,
		b.buildAuthzClassifierOption,
		b.buildKBClassifiersOption,
		b.buildEventClassifierOption,
		b.buildGenericClassifiersOption,
		b.buildModalityClassifierOption,
	}
	parallelOptions, err := b.buildParallelOptions(steps)
	b.options = append(b.options, parallelOptions...)
	if err != nil {
		return nil, err
	}
	if err := b.addCategoryClassifier(categoryMapping); err != nil {
		return nil, err
	}
	b.addMCPCategoryClassifier()
	if err := b.addComplexityBackend(); err != nil {
		return nil, err
	}
	return b.options, nil
}

func (b *classifierOptionBuilder) buildParallelOptions(steps []func() (option, error)) ([]option, error) {
	if len(steps) == 0 {
		return nil, nil
	}

	results := make([]option, len(steps))
	var group errgroup.Group
	group.SetLimit(classifierBuildParallelism(len(steps)))

	for i, step := range steps {
		stepIndex := i
		stepFn := step
		group.Go(func() error {
			opt, err := stepFn()
			if err != nil {
				return err
			}
			results[stepIndex] = opt
			return nil
		})
	}

	err := group.Wait()

	options := make([]option, 0, len(results))
	for _, opt := range results {
		if opt != nil {
			options = append(options, opt)
		}
	}
	return options, err
}

// Options carry ownership into the final classifier. A failed build applies
// the completed options to an unpublished shell solely to release that same
// ownership, including successful parallel steps after a sibling failed.
func (b *classifierOptionBuilder) closePending() {
	classifier := &Classifier{Config: b.cfg, embeddingSet: b.embeddingSet, ownsEmbeddingSet: b.ownsEmbeddingSet}
	for _, option := range b.options {
		if option != nil {
			option(classifier)
		}
	}
	_ = classifier.Close()
}

func classifierBuildParallelism(stepCount int) int {
	if stepCount <= 1 {
		return 1
	}
	backend := embeddingBackendOverride()
	if backend == "" || backend == "candle" {
		return 1
	}
	parallelism := runtime.NumCPU()
	if parallelism <= 0 {
		parallelism = 1
	}
	if parallelism > stepCount {
		parallelism = stepCount
	}
	return parallelism
}

func (b *classifierOptionBuilder) initMultiModalIfNeeded(reason string) error {
	_, err := b.embeddingProviderForModel("multimodal", 0, 0)
	if err != nil {
		return fmt.Errorf("%s: %w", reason, err)
	}
	return nil
}

func (b *classifierOptionBuilder) defaultEmbeddingModelType() string {
	modelType := b.cfg.EmbeddingConfig.ModelType
	if modelType == "" {
		return "qwen3"
	}
	return modelType
}

func (c *Classifier) logHeuristicClassifierInitialization() {
	if c.contextClassifier != nil {
		logging.ComponentEvent("classifier", "context_classifier_initialized", map[string]interface{}{
			"rules": len(c.contextClassifier.rules),
		})
	}
	if c.structureClassifier != nil {
		logging.ComponentEvent("classifier", "structure_classifier_initialized", map[string]interface{}{
			"rules": len(c.structureClassifier.rules),
		})
	}
}
