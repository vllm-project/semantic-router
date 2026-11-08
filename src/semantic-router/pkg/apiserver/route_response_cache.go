//go:build !windows

package apiserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type responseCacheTestRequest struct {
	Configuration json.RawMessage `json:"configuration"`
}

type responseCacheTestResponse struct {
	Valid        bool                      `json:"valid"`
	Healthy      bool                      `json:"healthy"`
	Capabilities cache.BackendCapabilities `json:"capabilities"`
}

type responseCacheInvalidateRequest struct {
	Selector cache.CacheSelector `json:"selector"`
	DryRun   *bool               `json:"dry_run,omitempty"`
}

type responseCacheFlushRequest struct {
	Selector cache.CacheSelector `json:"selector"`
	Confirm  string              `json:"confirm"`
}

func (s *ClassificationAPIServer) handleResponseCacheCapabilities(
	w http.ResponseWriter,
	_ *http.Request,
) {
	service, release := s.currentResponseCache()
	defer release()
	if service == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNAVAILABLE", "Response cache is unavailable")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, service.Capabilities())
}

func (s *ClassificationAPIServer) handleResponseCacheHealth(
	w http.ResponseWriter,
	r *http.Request,
) {
	service, release := s.currentResponseCache()
	defer release()
	if service == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNAVAILABLE", "Response cache is unavailable")
		return
	}
	if err := service.Health(r.Context()); err != nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNHEALTHY", "Response cache health check failed")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, cacheHealthResponse{
		Status:       "healthy",
		Capabilities: service.Capabilities(),
	})
}

func (s *ClassificationAPIServer) handleResponseCacheStats(
	w http.ResponseWriter,
	r *http.Request,
) {
	service, release := s.currentResponseCache()
	defer release()
	if service == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNAVAILABLE", "Response cache is unavailable")
		return
	}
	stats, err := service.Stats(r.Context())
	if err != nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_STATS_FAILED", "Response cache statistics are unavailable")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, stats)
}

func (s *ClassificationAPIServer) handleResponseCacheTest(
	w http.ResponseWriter,
	r *http.Request,
) {
	var request responseCacheTestRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	var candidate cache.CacheConfig
	var store struct {
		Embedding *config.EmbeddingBinding `yaml:"embedding"`
	}
	if err := yaml.Unmarshal(request.Configuration, &candidate); err != nil || yaml.Unmarshal(request.Configuration, &store) != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CACHE_CONFIG", "Invalid response cache configuration")
		return
	}
	if candidate.Enabled {
		deployment, provider, release, prepareErr := s.acquireResponseCacheEmbedding(store.Embedding)
		defer release()
		if prepareErr != nil {
			s.writeErrorResponse(w, http.StatusServiceUnavailable, "EMBEDDING_UNAVAILABLE", "Response cache embedding provider is unavailable")
			return
		}
		candidate.EmbeddingProvider, candidate.EmbeddingDeployment = provider, deployment
		if store.Embedding != nil {
			candidate.EmbeddingDimension = store.Embedding.Dimension
		}
	}
	backend, err := cache.NewCacheBackend(candidate)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CACHE_CONFIG", err.Error())
		return
	}
	defer func() { _ = backend.Close() }()
	if err = cache.ValidateBackendEmbedding(r.Context(), backend); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CACHE_CONFIG", err.Error())
		return
	}
	healthErr := backend.CheckConnection(r.Context())
	response := responseCacheTestResponse{
		Valid:        true,
		Healthy:      healthErr == nil,
		Capabilities: cache.CapabilitiesForBackend(candidate.BackendType),
	}
	if healthErr != nil {
		s.writeJSONResponse(w, http.StatusServiceUnavailable, response)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

// acquireResponseCacheEmbedding borrows the provider of the deployment a
// candidate cache embeds with, its binding's or embeddings.default: the live
// cache's, else the embeddings API's.
func (s *ClassificationAPIServer) acquireResponseCacheEmbedding(binding *config.EmbeddingBinding) (string, embedding.Provider, func(), error) {
	deployment := ""
	if binding != nil {
		deployment = binding.Deployment
	}
	if deployment == "" {
		cfg := s.currentConfig()
		if cfg == nil {
			return "", nil, func() {}, fmt.Errorf("router configuration is unavailable")
		}
		deployment = cfg.EmbeddingDeployment
	}
	service, release := s.currentResponseCache()
	if service != nil {
		if provider, err := service.PreparedEmbedding(deployment); err == nil {
			return deployment, provider, release, nil
		}
	}
	release()
	_, prepared, release, err := s.acquireEmbeddingRuntime()
	if err != nil {
		return deployment, nil, release, err
	}
	provider, err := prepared.Get(deployment, 0, 0)
	return deployment, provider, release, err
}

func (s *ClassificationAPIServer) handleResponseCacheInvalidate(
	w http.ResponseWriter,
	r *http.Request,
) {
	var request responseCacheInvalidateRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	service, release := s.currentResponseCache()
	defer release()
	if service == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNAVAILABLE", "Response cache is unavailable")
		return
	}
	dryRun := true
	if request.DryRun != nil {
		dryRun = *request.DryRun
	}
	result, err := service.Invalidate(r.Context(), request.Selector, dryRun)
	if err != nil {
		s.writeCacheAdminError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, result)
}

func (s *ClassificationAPIServer) handleResponseCacheFlush(
	w http.ResponseWriter,
	r *http.Request,
) {
	var request responseCacheFlushRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if request.Confirm != "flush response cache" {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIRMATION_REQUIRED", "confirm must equal \"flush response cache\"")
		return
	}
	service, release := s.currentResponseCache()
	defer release()
	if service == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CACHE_UNAVAILABLE", "Response cache is unavailable")
		return
	}
	result, err := service.Flush(r.Context(), request.Selector)
	if err != nil {
		s.writeCacheAdminError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, result)
}

func (s *ClassificationAPIServer) writeCacheAdminError(w http.ResponseWriter, err error) {
	if errors.Is(err, cache.ErrUnsupported) {
		s.writeErrorResponse(w, http.StatusNotImplemented, "CACHE_OPERATION_UNSUPPORTED", "Response cache backend does not support this operation")
		return
	}
	s.writeErrorResponse(w, http.StatusBadRequest, "CACHE_OPERATION_FAILED", err.Error())
}
