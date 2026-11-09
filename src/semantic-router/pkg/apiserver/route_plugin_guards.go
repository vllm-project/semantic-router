//go:build !windows

package apiserver

import (
	"context"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func apiPluginGuardRoutes() []apiRoute {
	policy := routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/response_guard/preview", Method: "POST", Description: "Preview what an active binding's response_guard does with an answer; mode=probe reads the guard's request-stage conditions on the question and its response-stage signals on the answer and context, without generation or persistence"}, policy, (*ClassificationAPIServer).handleResponseGuardPreview, pluginOperationFor(config.DecisionPluginResponseGuard, "preview"), pluginOperationFor(config.DecisionPluginResponseGuard, "probe"), strictJSONBodyFor[pluginruntime.ResponseGuardPreviewRequest](), jsonResponse[pluginruntime.ResponseGuardPreviewResponse](http.StatusOK, "Guard action, rule outcomes and the response-stage signals they read"), errorResponses(http.StatusBadRequest, http.StatusNotFound, http.StatusRequestEntityTooLarge, http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable, http.StatusGatewayTimeout)),
	}
}

func (s *ClassificationAPIServer) handleResponseGuardPreview(w http.ResponseWriter, r *http.Request) {
	var request pluginruntime.ResponseGuardPreviewRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	runPluginPreview(s, w, r, func(ctx context.Context, runtime pluginruntime.Capabilities) (pluginruntime.ResponseGuardPreviewResponse, error) {
		if runtime.Guards == nil {
			return pluginruntime.ResponseGuardPreviewResponse{}, pluginruntime.ErrUnavailable
		}
		return runtime.Guards.PreviewResponseGuard(ctx, request)
	})
}

// runPluginPreview shares the configured preview admission/deadline budget.
// On timeout the worker retains its generation lease until the real operation
// finishes, so reload/shutdown cannot close an in-flight classifier or store.
func runPluginPreview[T any](s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request, invoke func(context.Context, pluginruntime.Capabilities) (T, error)) {
	if s.runtimeRegistry == nil {
		s.writeRuntimePluginPreviewError(w, pluginruntime.ErrUnavailable)
		return
	}
	cfg, runtime, release, ok := s.runtimeRegistry.AcquirePluginRuntime()
	if !ok {
		s.writeRuntimePluginPreviewError(w, pluginruntime.ErrUnavailable)
		return
	}
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
		s.writeErrorResponse(w, http.StatusInternalServerError, "RESPONSE_DEADLINE_ERROR", "could not configure plugin preview response deadline")
		return
	}
	if ctx.Err() != nil {
		release()
		s.writeRuntimePluginPreviewError(w, ctx.Err())
		return
	}
	ticket, err := s.previewAdmission.Acquire(ctx)
	if err != nil {
		release()
		s.writeClassificationError(w, err)
		return
	}
	response, err := runRetainedAPIWork(ctx, func() { release(); ticket() }, func(ctx context.Context) (T, error) { return invoke(ctx, runtime) })
	if err != nil {
		s.writeRuntimePluginPreviewError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) writeRuntimePluginPreviewError(w http.ResponseWriter, err error) {
	status, code, message := http.StatusBadRequest, "INVALID_PLUGIN_PREVIEW", "Plugin preview could not be performed"
	switch {
	case errors.Is(err, pluginruntime.ErrInvalidBinding):
		status, code, message = http.StatusNotFound, "PLUGIN_BINDING_NOT_FOUND", pluginruntime.ErrInvalidBinding.Error()
	case errors.Is(err, pluginruntime.ErrUnavailable), errors.Is(err, errAPIWorkerUnavailable):
		status, code, message = http.StatusServiceUnavailable, "PLUGIN_RUNTIME_UNAVAILABLE", pluginruntime.ErrUnavailable.Error()
	case errors.Is(err, pluginruntime.ErrProbeRequired):
		code, message = "PLUGIN_PROBE_REQUIRED", pluginruntime.ErrProbeRequired.Error()
	case errors.Is(err, context.DeadlineExceeded):
		status, code, message = http.StatusGatewayTimeout, "REQUEST_TIMEOUT", "Plugin preview exceeded its request deadline"
	case errors.Is(err, context.Canceled):
		return
	}
	s.writeErrorResponse(w, status, code, message)
}
