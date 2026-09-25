package services

import (
	"context"
	"errors"
	"io"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// SetGlobalEmbeddings binds a borrowed global service view before publishing a
// router generation. The generation owns its lifetime; this service never closes it.
func (s *ClassificationService) SetGlobalEmbeddings(prepared *embedding.Set) {
	s.configMutex.Lock()
	s.globalEmbeddings = prepared
	s.configMutex.Unlock()
}

// AcquireEmbeddingAPISnapshot pins the selected embedding view until release.
// The opt-in API uses global execution settings; otherwise existing default
// recipe diagnostics keep their scope. No model is loaded by a request.
func (s *ClassificationService) AcquireEmbeddingAPISnapshot() (*config.RouterConfig, *embedding.Set, func(), error) {
	if s == nil {
		return nil, nil, func() {}, binding.ErrNotPrepared
	}
	s.runtimeMutex.RLock()
	var once sync.Once
	release := func() { once.Do(s.runtimeMutex.RUnlock) }
	if s.closed {
		release()
		return nil, nil, func() {}, binding.ErrClosed
	}
	s.configMutex.RLock()
	cfg, classifier, prepared := s.config, s.classifier, s.globalEmbeddings
	s.configMutex.RUnlock()
	if cfg != nil && cfg.API.Embeddings.Enabled {
		if prepared == nil || !prepared.Ready() {
			return cfg, nil, release, binding.ErrNotPrepared
		}
		return cfg.ConfigForGlobalModelServices(), prepared, release, nil
	}
	if classifier == nil || classifier.PreparedEmbeddings() == nil {
		return cfg, nil, release, binding.ErrNotPrepared
	}
	return classifier.Config, classifier.PreparedEmbeddings(), release, nil
}

type classifierAPIResources struct {
	classifiers io.Closer
	embeddings  *embedding.Set
}

func (r classifierAPIResources) Close() error {
	return errors.Join(r.classifiers.Close(), r.embeddings.Close())
}

func (s *ClassificationService) prepareAndPublishClassifiers(cfg *config.RouterConfig, classifier *classification.Classifier, recipes *classification.RecipeClassifiers, owner io.Closer, runtime *native.Runtime) error {
	prepared, err := modelruntime.PrepareOwnedEmbeddingAPI(context.Background(), cfg, runtime)
	if err != nil {
		_ = owner.Close()
		return err
	}
	s.publishClassifiers(cfg, classifier, recipes, classifierAPIResources{classifiers: owner, embeddings: prepared}, prepared)
	return nil
}
