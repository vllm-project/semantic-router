package classification

import (
	"fmt"
	"runtime"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type classifierOptionBuilder struct {
	cfg                *config.RouterConfig
	options            []option
	multiModalInitOnce sync.Once
	multiModalInitErr  error
}

func newClassifierOptionBuilder(cfg *config.RouterConfig, options []option) *classifierOptionBuilder {
	return &classifierOptionBuilder{cfg: cfg, options: options}
}

func (b *classifierOptionBuilder) build(categoryMapping *CategoryMapping) ([]option, error) {
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
	}
	parallelOptions, err := b.buildParallelOptions(steps)
	if err != nil {
		return nil, err
	}
	b.options = append(b.options, parallelOptions...)
	b.addCategoryClassifier(categoryMapping)
	b.addMCPCategoryClassifier()
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

	if err := group.Wait(); err != nil {
		return nil, err
	}

	options := make([]option, 0, len(results))
	for _, opt := range results {
		if opt != nil {
			options = append(options, opt)
		}
	}
	return options, nil
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
	b.multiModalInitOnce.Do(func() {
		mmPath := config.ResolveModelPath(b.cfg.MultiModalModelPath)
		if mmPath == "" {
			b.multiModalInitErr = fmt.Errorf("%s requires embedding_models.multimodal_model_path to be set", reason)
			return
		}
		if err := initMultiModalModel(mmPath, b.cfg.UseCPU); err != nil {
			b.multiModalInitErr = fmt.Errorf("failed to initialize multimodal model for %s: %w", reason, err)
			return
		}
		logging.ComponentEvent("classifier", "multimodal_embedding_initialized", map[string]interface{}{
			"reason":    reason,
			"model_ref": mmPath,
			"use_cpu":   b.cfg.UseCPU,
		})
	})
	return b.multiModalInitErr
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
