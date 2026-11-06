package extproc

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// signalRuntime is the part of a router generation that extracts signals:
// the model runtime deployments it leases and the embeddings, rerankers and
// recipe classifiers built on them. Building one waits until every
// Router-managed deployment it leases is ready. When a configuration change
// leaves the signals component's resources unchanged, the next generation
// shares this runtime instead of building another, and the last generation
// that uses it closes it.
type signalRuntime struct {
	// key identifies what the runtime was built from; empty until the
	// configuration lifecycle names it, and never shared while empty.
	key string

	modelLease *modelservice.Lease
	// systemOne asks the systemone deployments' remote APIs. Their
	// definitions are runtime models, so a change builds another runtime.
	systemOne         *modelservice.RemoteSystemOne
	serving           *serving.Runtime
	embeddings        *embedding.Set
	serviceEmbeddings *embedding.Set
	cacheEmbeddings   *embedding.Set
	rerankers         map[config.RecipeName]modelruntime.PairScorer
	recipeClassifiers *classification.RecipeClassifiers
	classifier        *classification.Classifier

	resources *resourceScope
	refs      atomic.Int32
}

// buildSignalRuntime builds the signal runtime for cfg, held once.
func buildSignalRuntime(cfg *config.RouterConfig, pool *binding.Pool) (*signalRuntime, error) {
	signals := &signalRuntime{resources: newResourceScope()}
	signals.refs.Store(1)
	if err := signals.build(cfg, pool); err != nil {
		return nil, rollbackResources(signals.resources, err)
	}
	return signals, nil
}

func (s *signalRuntime) build(cfg *config.RouterConfig, pool *binding.Pool) error {
	var err error
	if s.systemOne, err = modelservice.NewRemoteSystemOne(cfg); err != nil {
		return err
	}
	s.resources.add(s.systemOne.Close)
	var services serving.Services
	if manager := modelservice.DefaultManager(); manager != nil {
		if s.modelLease, err = manager.Acquire(cfg); err != nil {
			return err
		}
		services = s.modelLease
		s.resources.add(s.modelLease.Close)
		// Decision signals and the decision selector fail open while their
		// model loads, so a generation that served before its Router-managed
		// models are ready would route on unknown answers.
		if err = s.modelLease.WaitManaged(context.Background()); err != nil {
			return err
		}
	}
	// Built-in signals that resolve to a systemone deployment ask it through
	// the same runtime as the model_runtime ones.
	if !s.systemOne.Empty() {
		services = serving.WithRemote(services, s.systemOne)
	}
	s.serving = serving.New(services, pool)

	if s.embeddings, err = modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, s.serving); err != nil {
		return err
	}
	s.resources.add(s.embeddings.Close)
	servicesConfig := *cfg
	// Ingestion owns an independent handle for its longer worker lifetime.
	servicesConfig.VectorStore = nil
	if s.serviceEmbeddings, err = modelruntime.PrepareOwnedGlobalServiceEmbeddings(context.Background(), &servicesConfig, s.serving); err != nil {
		return err
	}
	s.resources.add(s.serviceEmbeddings.Close)
	s.cacheEmbeddings = s.embeddings
	if cfg.NeedsSemanticResponseCache() {
		if s.cacheEmbeddings, err = modelruntime.PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, s.serving); err != nil {
			return err
		}
		s.resources.add(s.cacheEmbeddings.Close)
	}
	if s.rerankers, err = modelruntime.PrepareRerankers(context.Background(), cfg, s.serving); err != nil {
		return err
	}
	s.resources.add(func() error { return modelruntime.CloseRerankers(s.rerankers) })

	s.recipeClassifiers, s.classifier, err = buildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{
		Runtime: s.serving, Embeddings: s.embeddings, SystemOne: s.systemOne,
	})
	if err != nil {
		return err
	}
	s.resources.add(s.recipeClassifiers.Close)
	return nil
}

// share holds the runtime for one more generation. The remote answers
// cached for the generation before it are dropped.
func (s *signalRuntime) share() *signalRuntime {
	s.refs.Add(1)
	s.systemOne.ResetCache()
	return s
}

// release lets go of one generation's hold and closes the runtime when no
// generation holds it any more.
func (s *signalRuntime) release() error {
	if s.refs.Add(-1) == 0 {
		return s.resources.close()
	}
	return nil
}

// sharedWith returns the runtime when a generation built from key can share
// it.
func (s *signalRuntime) sharedWith(key string) (*signalRuntime, bool) {
	if s == nil || s.key == "" || s.key != key {
		return nil, false
	}
	return s, true
}

func buildRecipeClassifiers(
	cfg *config.RouterConfig,
	runtimeOptions classification.RecipeRuntimeOptions,
) (*classification.RecipeClassifiers, *classification.Classifier, error) {
	classifiers, err := classification.BuildRecipeClassifiers(cfg, runtimeOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build recipe classifiers: %w", err)
	}
	if err := classifiers.InitializeRuntime(); err != nil {
		_ = classifiers.Close()
		return nil, nil, fmt.Errorf("failed to initialize recipe classifiers: %w", err)
	}
	defaultClassifier := classifiers.Default()
	if defaultClassifier == nil {
		_ = classifiers.Close()
		return nil, nil, fmt.Errorf("default routing recipe classifier is unavailable")
	}
	return classifiers, defaultClassifier, nil
}
