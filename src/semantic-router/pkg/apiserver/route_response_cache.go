//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"net/http"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
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
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status":       "healthy",
		"capabilities": service.Capabilities(),
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
	if err := yaml.Unmarshal(request.Configuration, &candidate); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CACHE_CONFIG", "Invalid response cache configuration")
		return
	}
	backend, err := cache.NewCacheBackend(candidate)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_CACHE_CONFIG", err.Error())
		return
	}
	defer func() { _ = backend.Close() }()
	healthErr := backend.CheckConnection()
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
