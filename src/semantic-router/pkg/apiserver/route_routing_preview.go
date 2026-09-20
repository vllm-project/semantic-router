//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type routingPreviewResult struct {
	response *services.EvalResponse
	err      error
}

func (s *ClassificationAPIServer) initRoutingPreviewAdmission(cfg *config.RouterConfig) {
	s.previewAdmissionOnce.Do(func() {
		settings := config.RoutingPreviewConfig{}
		if cfg != nil {
			settings = cfg.API.RoutingPreview
		}
		s.previewAdmission = admission.NewSemaphore(settings.ConcurrencyLimit(), 0, 0, admission.OverflowShed)
	})
}

func (s *ClassificationAPIServer) runRoutingPreview(w http.ResponseWriter, r *http.Request, req services.IntentRequest) {
	cfg, service, release := s.acquireClassificationRuntime()
	settings := config.RoutingPreviewConfig{}
	if cfg != nil {
		settings = cfg.API.RoutingPreview
	}
	s.initRoutingPreviewAdmission(cfg)
	ctx, cancel := context.WithTimeout(r.Context(), settings.RequestTimeout())
	defer cancel()
	deadline, _ := ctx.Deadline()
	if err := http.NewResponseController(w).SetWriteDeadline(deadline.Add(config.RoutingPreviewResponseWriteAllowance)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		release()
		s.writeErrorResponse(w, http.StatusInternalServerError, "RESPONSE_DEADLINE_ERROR", "could not configure Preview response deadline")
		return
	}
	if ctx.Err() != nil {
		release()
		s.writeRoutingPreviewContextError(w, ctx.Err())
		return
	}
	ticket, err := s.previewAdmission.Acquire(ctx)
	if err != nil {
		release()
		s.writeClassificationError(w, err)
		return
	}
	workerDone, ok := retainAPIWorker(ctx)
	if !ok {
		ticket()
		release()
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SHUTTING_DOWN", "API server is shutting down")
		return
	}
	completed := make(chan routingPreviewResult, 1)
	go func() {
		var result routingPreviewResult
		defer func() {
			if recovered := recover(); recovered != nil {
				logging.Errorf("Routing preview worker panicked: %v", recovered)
				result = routingPreviewResult{err: fmt.Errorf("routing preview inference failed")}
			}
			// Ownership remains with this worker after the handler returns 504.
			release()
			ticket()
			workerDone()
			completed <- result
		}()
		if result.err = ctx.Err(); result.err == nil {
			result.response, result.err = service.ClassifyIntentForEval(ctx, req)
		}
	}()
	select {
	case <-ctx.Done():
		s.writeRoutingPreviewContextError(w, ctx.Err())
	case result := <-completed:
		// Cancellation wins over late success or partial decision diagnostics.
		if ctx.Err() != nil {
			s.writeRoutingPreviewContextError(w, ctx.Err())
			return
		}
		s.writeRoutingPreviewResult(w, result)
	}
}

func (s *ClassificationAPIServer) writeRoutingPreviewContextError(w http.ResponseWriter, err error) {
	if errors.Is(err, context.DeadlineExceeded) {
		s.writeErrorResponse(w, http.StatusGatewayTimeout, "REQUEST_TIMEOUT", "routing preview exceeded its request deadline")
	}
	// A disconnected caller cannot receive a response. Its worker still drains.
}

func (s *ClassificationAPIServer) writeRoutingPreviewResult(w http.ResponseWriter, result routingPreviewResult) {
	if result.response != nil && headers.ValidConfigHash(result.response.ConfigHash) {
		w.Header().Set(headers.VSRConfigHash, result.response.ConfigHash)
	}
	if errors.Is(result.err, context.DeadlineExceeded) {
		s.writeRoutingPreviewContextError(w, result.err)
		return
	}
	if errors.Is(result.err, context.Canceled) {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "INFERENCE_CANCELED", "routing preview inference was canceled")
		return
	}
	if result.err != nil {
		if result.response != nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, result.response)
			return
		}
		s.writeClassificationError(w, result.err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, result.response)
}
