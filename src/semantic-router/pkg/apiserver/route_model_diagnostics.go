//go:build !windows

package apiserver

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func apiModelDiagnosticRoutes() []apiRoute {
	path := apiDiagnosticsPath + "/models"
	policy := routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: path, Method: http.MethodGet, Description: "List prepared model bindings in an explicitly selected recipe", Parameters: []OpenAPIParameter{{Name: "recipe", In: "query", Required: true, Description: "Configured recipe name", Schema: OpenAPISchema{Type: "string"}}}}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleModelDiagnosticInventory, jsonResponse[ModelDiagnosticInventory](http.StatusOK, "Prepared binding inventory"), errorResponses(400, 404, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/embeddings", Method: http.MethodPost, Description: "Run the explicitly selected prepared embedding binding at its published representation"}, policy, (*ClassificationAPIServer).handleModelDiagnosticEmbedding, strictJSONBodyFor[ModelTextDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]](http.StatusOK, "Embedding vector and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
	}
}

func (s *ClassificationAPIServer) acquireModelDiagnostics(recipe string) (*config.RouterConfig, services.ModelDiagnostics, func(), error) {
	cfg, service, releaseGeneration := s.acquireClassificationRuntime()
	source, ok := service.(interface {
		AcquireModelDiagnostics(string) (services.ModelDiagnostics, func(), error)
	})
	if !ok {
		releaseGeneration()
		return nil, services.ModelDiagnostics{}, func() {}, binding.ErrNotPrepared
	}
	diagnostics, releaseService, err := source.AcquireModelDiagnostics(recipe)
	if err != nil {
		releaseGeneration()
		return nil, services.ModelDiagnostics{}, func() {}, err
	}
	return cfg, diagnostics, func() { releaseService(); releaseGeneration() }, nil
}

func (s *ClassificationAPIServer) handleModelDiagnosticInventory(w http.ResponseWriter, r *http.Request) {
	recipe := r.URL.Query().Get("recipe")
	_, runtime, release, err := s.acquireModelDiagnostics(recipe)
	defer release()
	if err != nil {
		s.writeModelDiagnosticError(w, err)
		return
	}
	response := ModelDiagnosticInventory{Recipe: recipe, Bindings: []ModelDiagnosticBinding{}}
	for _, entry := range runtime.PreparedBindings() {
		if entry.Identity.Recipe == recipe {
			response.Bindings = append(response.Bindings, diagnosticBinding(entry))
		}
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func runModelTextDiagnostic[T any](s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request, call func(context.Context, services.ModelDiagnostics, ModelTextDiagnosticRequest) (ModelDiagnosticResponse[T], error)) {
	var request ModelTextDiagnosticRequest
	if err := s.parseStrictJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if strings.TrimSpace(request.Binding) == "" || strings.TrimSpace(request.Text) == "" {
		s.writeModelDiagnosticError(w, binding.ErrInvalidInput)
		return
	}
	_, runtime, release, err := s.acquireModelDiagnostics(request.Recipe)
	if err != nil {
		s.writeModelDiagnosticError(w, err)
		return
	}
	response, err := executeModelDiagnostic(w, r, release, func(ctx context.Context) (ModelDiagnosticResponse[T], error) { return call(ctx, runtime, request) })
	if err != nil {
		s.writeModelDiagnosticError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleModelDiagnosticEmbedding(w http.ResponseWriter, r *http.Request) {
	runModelTextDiagnostic(s, w, r, func(ctx context.Context, runtime services.ModelDiagnostics, req ModelTextDiagnosticRequest) (ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult], error) {
		result, err := runtime.Tasks.DiagnoseEmbedding(ctx, req.Recipe, req.Binding, req.Text)
		return ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]{Binding: diagnosticBinding(result.Binding), Result: ModelDiagnosticEmbeddingResult{Embedding: result.Result.Embedding, Input: diagnosticInput(result.Result.Input)}}, err
	})
}

func (s *ClassificationAPIServer) writeModelDiagnosticError(w http.ResponseWriter, err error) {
	status, code, message := http.StatusInternalServerError, "MODEL_DIAGNOSTIC_ERROR", "Model diagnostic execution failed"
	switch {
	case errors.Is(err, binding.ErrInvalidInput), errors.Is(err, binding.ErrInputLimit):
		status, code, message = http.StatusBadRequest, "INVALID_MODEL_INPUT", err.Error()
	case errors.Is(err, binding.ErrNotPrepared), errors.Is(err, services.ErrUnknownDiagnosticRecipe):
		status, code, message = http.StatusNotFound, "PREPARED_BINDING_NOT_FOUND", "The selected recipe has no prepared binding for this operation"
	case errors.Is(err, binding.ErrCapability):
		status, code, message = http.StatusConflict, "MODEL_CAPABILITY_MISMATCH", err.Error()
	case errors.Is(err, admission.ErrQueueFull):
		status, code, message = http.StatusTooManyRequests, "OVERLOADED", "Model admission queue is full"
	case errors.Is(err, errAPIWorkerUnavailable), errors.Is(err, binding.ErrClosed), errors.Is(err, services.ErrClassifierUnavailable), errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		status, code, message = http.StatusServiceUnavailable, "MODEL_DIAGNOSTIC_UNAVAILABLE", "The model runtime is unavailable or the request was canceled"
	}
	s.writeErrorResponse(w, status, code, message)
}

// executeModelDiagnostic bounds the response while transferring the lease to
// the actual inference, which keeps the binding's one shared admission ticket
// until the model call returns; the API does not add a competing admission
// policy or discover another model instance.
func executeModelDiagnostic[T any](w http.ResponseWriter, r *http.Request, release func(), invoke func(context.Context) (T, error)) (T, error) {
	var zero T
	ctx, cancel := context.WithTimeout(r.Context(), apiWriteTimeout)
	defer cancel()
	deadline, _ := ctx.Deadline()
	if err := http.NewResponseController(w).SetWriteDeadline(deadline.Add(config.RoutingPreviewResponseWriteAllowance)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		release()
		return zero, err
	}
	return runRetainedAPIWork(ctx, release, invoke)
}
