//go:build !windows && cgo

package apiserver

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

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

func (s *ClassificationAPIServer) acquireEmbeddings() (*embedding.Set, func(), error) {
	_, prepared, release, err := s.acquireEmbeddingRuntime()
	return prepared, release, err
}

func ownedEmbeddingOutput(ctx context.Context, set *embedding.Set, request EmbeddingRequest, text string) (EmbeddingResult, error) {
	model := request.Model
	if model == "auto" || model == "" {
		var err error
		model, err = set.Select(text, request.QualityPriority, request.LatencyPriority, request.Dimension)
		if err != nil {
			return EmbeddingResult{}, err
		}
	}
	provider, err := set.Get(model, request.Dimension, request.TargetLayer)
	if err != nil {
		return EmbeddingResult{}, err
	}
	start := time.Now()
	vector, err := provider.Embed(ctx, text)
	return EmbeddingResult{Text: text, Embedding: vector, Dimension: len(vector), ModelUsed: model, ProcessingTimeMs: time.Since(start).Milliseconds()}, err
}

func buildOwnedEmbeddingResults(ctx context.Context, set *embedding.Set, request EmbeddingRequest) ([]EmbeddingResult, int64, error) {
	results := make([]EmbeddingResult, 0, len(request.Texts)+len(request.Images))
	var elapsed int64
	for _, text := range request.Texts {
		output, err := ownedEmbeddingOutput(ctx, set, request, text)
		if err != nil {
			return nil, 0, err
		}
		elapsed += output.ProcessingTimeMs
		results = append(results, output)
	}
	for index, image := range request.Images {
		provider, err := set.Get("multimodal", request.Dimension, 0)
		if err != nil {
			return nil, 0, err
		}
		start := time.Now()
		vector, err := embedding.Image(ctx, provider, image, request.Dimension)
		if err != nil {
			return nil, 0, &imageEncodeError{index: index, err: err}
		}
		duration := time.Since(start).Milliseconds()
		elapsed += duration
		results = append(results, EmbeddingResult{Modality: "image", Embedding: vector, Dimension: len(vector), ModelUsed: "multi-modal-embed", ProcessingTimeMs: duration})
	}
	return results, elapsed, nil
}

func embeddingCosine(a, b []float32) (float32, error) {
	if len(a) != len(b) || len(a) == 0 {
		return 0, fmt.Errorf("embedding dimensions differ or are empty")
	}
	var dot, aa, bb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	if aa == 0 || bb == 0 {
		return 0, fmt.Errorf("zero-norm embedding")
	}
	return float32(dot / math.Sqrt(aa*bb)), nil
}

func ownedBatchSimilarity(ctx context.Context, set *embedding.Set, request BatchSimilarityRequest) (BatchSimilarityResponse, error) {
	req := EmbeddingRequest{Model: request.Model, Dimension: request.Dimension, QualityPriority: request.QualityPriority, LatencyPriority: request.LatencyPriority}
	start := time.Now()
	query, err := ownedEmbeddingOutput(ctx, set, req, request.Query)
	if err != nil {
		return BatchSimilarityResponse{}, err
	}
	// A batch compares all candidates in the query's selected vector space.
	req.Model = query.ModelUsed
	matches := make([]BatchSimilarityMatch, len(request.Candidates))
	for index, text := range request.Candidates {
		candidate, err := ownedEmbeddingOutput(ctx, set, req, text)
		if err != nil {
			return BatchSimilarityResponse{}, err
		}
		score, err := embeddingCosine(query.Embedding, candidate.Embedding)
		if err != nil {
			return BatchSimilarityResponse{}, err
		}
		matches[index] = BatchSimilarityMatch{Index: index, Text: text, Similarity: score}
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].Similarity > matches[j].Similarity })
	if len(matches) > request.TopK {
		matches = matches[:request.TopK]
	}
	return BatchSimilarityResponse{Matches: matches, TotalCandidates: len(request.Candidates), ModelUsed: query.ModelUsed, ProcessingTimeMs: float32(time.Since(start).Microseconds()) / 1000}, nil
}
