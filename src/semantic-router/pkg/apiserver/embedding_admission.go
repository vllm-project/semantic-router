//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

const embeddingProcessAdmissionCapacity = embedding.DefaultProcessAdmissionCapacity

var errEmbeddingOverloaded = embedding.ErrOverloaded

func newEmbeddingProcessAdmission(capacity int) *embedding.ProcessAdmission {
	return embedding.NewProcessAdmission(capacity)
}

func (s *ClassificationAPIServer) embeddingProcessAdmission() *embedding.ProcessAdmission {
	if s != nil && s.embeddingAdmission != nil {
		return s.embeddingAdmission
	}
	return embedding.DefaultProcessAdmission
}

func (s *ClassificationAPIServer) admitEmbeddingNative(
	w http.ResponseWriter,
	ctx context.Context,
) (func(), bool) {
	release, err := s.embeddingProcessAdmission().TryAcquire(ctx)
	if err == nil {
		return release, true
	}
	if errors.Is(err, errEmbeddingOverloaded) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Retry-After", "1")
		s.writeErrorResponse(
			w,
			http.StatusServiceUnavailable,
			"EMBEDDING_OVERLOADED",
			"embedding inference is temporarily overloaded",
		)
		return nil, false
	}
	s.writeErrorResponse(w, http.StatusRequestTimeout, "REQUEST_CANCELED", "request canceled")
	return nil, false
}

type localNativeEmbeddingCapability interface {
	UsesLocalNativeEmbeddings(hasImage bool) bool
}

func (s *ClassificationAPIServer) admitIntentClassificationNative(
	w http.ResponseWriter,
	ctx context.Context,
	req services.IntentRequest,
) (func(), bool) {
	hasImage := req.HasInlineImageInput()
	// Classify/eval performs strict full-image validation before it knows
	// whether an image rule will run, so every inline image consumes admission
	// even when all configured text embeddings are provider-backed.
	if !hasImage {
		if capability, ok := s.classificationSvc.(localNativeEmbeddingCapability); ok &&
			!capability.UsesLocalNativeEmbeddings(hasImage) {
			return func() {}, true
		}
	}
	return s.admitEmbeddingNative(w, ctx)
}
