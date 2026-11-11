package classification

import (
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
	providerErr      error
}

func newClassifierOptionBuilder(cfg *config.RouterConfig, options []option) *classifierOptionBuilder {
	return &classifierOptionBuilder{cfg: cfg, options: options}
}

func (b *classifierOptionBuilder) build() ([]option, error) {
	if err := b.prepareEmbeddingSet(); err != nil {
		return nil, err
	}
	if b.cfg.IsDefaultRecipe(b.cfg.RoutingScope) && !b.cfg.IsRecipeReachableForRouting(b.cfg.RoutingScope) {
		return b.options, nil
	}
	steps := []func() (option, error){
		b.buildKeywordClassifierOption,
		b.buildContextClassifierOption,
		b.buildAuthzClassifierOption,
	}
	parallelOptions, err := b.buildParallelOptions(steps)
	b.options = append(b.options, parallelOptions...)
	if err != nil {
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

func (c *Classifier) logHeuristicClassifierInitialization() {
	if c.contextClassifier != nil {
		logging.ComponentEvent("classifier", "context_classifier_initialized", map[string]interface{}{
			"rules": len(c.contextClassifier.rules),
		})
	}
}
