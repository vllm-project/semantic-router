package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type routerLearningAdaptationStrategy interface {
	Name() string
	Select(
		*OpenAIRouter,
		routerLearningInput,
		routerLearningProtectionPreflight,
		config.RouterLearningAdaptationConfig,
	) routerLearningDecision
}

type routerLearningAdaptationStrategyRegistry struct {
	strategies map[string]routerLearningAdaptationStrategy
}

func newRouterLearningAdaptationStrategyRegistry(
	strategies ...routerLearningAdaptationStrategy,
) routerLearningAdaptationStrategyRegistry {
	registry := routerLearningAdaptationStrategyRegistry{
		strategies: map[string]routerLearningAdaptationStrategy{},
	}
	for _, strategy := range strategies {
		if strategy == nil {
			continue
		}
		name := strings.TrimSpace(strategy.Name())
		if name == "" {
			continue
		}
		registry.strategies[name] = strategy
	}
	return registry
}

func (r routerLearningAdaptationStrategyRegistry) Strategy(
	cfg config.RouterLearningAdaptationConfig,
) (routerLearningAdaptationStrategy, bool) {
	name := strings.TrimSpace(cfg.EffectiveStrategy())
	if name == "" {
		name = config.RouterLearningStrategyRoutingSampling
	}
	strategy, ok := r.strategies[name]
	return strategy, ok
}

var routerLearningAdaptationStrategies = newRouterLearningAdaptationStrategyRegistry(
	routingSamplingAdaptationStrategy{},
)

type routingSamplingAdaptationStrategy struct{}

func (routingSamplingAdaptationStrategy) Name() string {
	return config.RouterLearningStrategyRoutingSampling
}

func (routingSamplingAdaptationStrategy) Select(
	router *OpenAIRouter,
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	cfg config.RouterLearningAdaptationConfig,
) routerLearningDecision {
	if router == nil {
		return routerLearningDecision{}
	}
	return router.applyRoutingSamplingAdaptation(input, preflight, cfg)
}
