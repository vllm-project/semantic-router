package modelruntime

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// PairScorer is the retrieval-facing port. Scores stay in query/document input
// order; the retrieval consumer owns document IDs, ordering and selection.
type PairScorer interface {
	ScorePairs(context.Context, string, []tasks.QueryDocument) (tasks.RelevanceScores, error)
	CacheIdentity() string
	Close() error
}

// PrepareRerankers loads only enabled, request-reachable RAG consumers. It uses
// the generation's native runtime and releases candidate references on failure.
func PrepareRerankers(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (map[config.RecipeName]PairScorer, error) {
	return prepareRerankers(ctx, cfg, func(ctx context.Context, spec config.ResolvedModelBinding) (PairScorer, error) {
		return runtime.Relevance(ctx, spec)
	})
}

func prepareRerankers(ctx context.Context, cfg *config.RouterConfig, load func(context.Context, config.ResolvedModelBinding) (PairScorer, error)) (map[config.RecipeName]PairScorer, error) {
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	prepared := make(map[config.RecipeName]PairScorer)
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		scope := cfg.ConfigForRecipe(recipe)
		if !scope.NeedsRAGReranker() {
			continue
		}
		spec, exists := plan.Lookup(recipe.Name, config.RAGRerankerConsumer)
		if !exists {
			return nil, errors.Join(fmt.Errorf("recipe %q requires %s", recipe.Name, config.RAGRerankerConsumer), CloseRerankers(prepared))
		}
		spec.Deployment.Artifact = config.ResolveModelPath(spec.Deployment.Artifact)
		model, loadErr := load(ctx, spec)
		if loadErr != nil {
			return nil, errors.Join(loadErr, CloseRerankers(prepared))
		}
		prepared[recipe.Name] = model
	}
	return prepared, nil
}

func CloseRerankers(models map[config.RecipeName]PairScorer) error {
	var errs []error
	for _, model := range models {
		errs = append(errs, model.Close())
	}
	return errors.Join(errs...)
}
