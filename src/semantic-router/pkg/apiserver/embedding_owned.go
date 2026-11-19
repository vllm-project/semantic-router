//go:build !windows

package apiserver

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func preparedEmbeddings(service classificationService) (*embedding.Set, error) {
	source, ok := service.(interface {
		GetClassifier() *classification.Classifier
	})
	if !ok {
		return nil, fmt.Errorf("embedding runtime is unavailable")
	}
	return classifierEmbeddings(source.GetClassifier())
}

func classifierEmbeddings(classifier *classification.Classifier) (*embedding.Set, error) {
	if classifier == nil || classifier.PreparedEmbeddings() == nil {
		return nil, fmt.Errorf("embedding runtime is unavailable")
	}
	return classifier.PreparedEmbeddings(), nil
}

// Direct provider consumers need a lease over the entire operation, including
// gaps between forwards. Service methods take their own runtime lock; this
// helper must not be used to wrap normal classification/readiness methods.
func (s *ClassificationAPIServer) acquireEmbeddingRuntime() (*config.RouterConfig, *embedding.Set, func(), error) {
	if s != nil && s.runtimeRegistry != nil {
		if cfg, service, release, ok := s.runtimeRegistry.AcquireClassificationRuntime(); ok {
			prepared, err := preparedEmbeddings(service)
			return cfg, prepared, release, err
		}
	}
	service, release := s.acquireClassificationService()
	if source, ok := service.(interface {
		AcquireRuntimeSnapshot() (*config.RouterConfig, *classification.Classifier, func())
	}); ok {
		cfg, classifier, snapshotRelease := source.AcquireRuntimeSnapshot()
		prepared, err := classifierEmbeddings(classifier)
		var once sync.Once
		return cfg, prepared, func() { once.Do(func() { snapshotRelease(); release() }) }, err
	}
	prepared, err := preparedEmbeddings(service)
	return s.currentConfig(), prepared, release, err
}
