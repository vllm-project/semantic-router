//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// defaultModelDiagnosticBatchSize bounds one native pair batch when the operator
// has not configured global.services.api.batch_classification.max_batch_size.
const defaultModelDiagnosticBatchSize = 100

func apiModelDiagnosticRoutes() []apiRoute {
	path := apiDiagnosticsPath + "/models"
	policy := routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: path, Method: http.MethodGet, Description: "List prepared model bindings in an explicitly selected recipe", Parameters: []OpenAPIParameter{{Name: "recipe", In: "query", Required: true, Description: "Configured recipe name", Schema: OpenAPISchema{Type: "string"}}}}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleModelDiagnosticInventory, jsonResponse[ModelDiagnosticInventory](http.StatusOK, "Prepared binding inventory"), errorResponses(400, 404, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/labels", Method: http.MethodPost, Description: "Inspect a prepared label distribution; windowed bindings preserve their configured scan"}, policy, (*ClassificationAPIServer).handleModelDiagnosticLabels, strictJSONBodyFor[ModelTextDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticLabelsResult]](http.StatusOK, "Label probabilities and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/label-scores", Method: http.MethodPost, Description: "Inspect independent label scores using the prepared operating point when configured"}, policy, (*ClassificationAPIServer).handleModelDiagnosticScores, strictJSONBodyFor[ModelTextDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticScoresResult]](http.StatusOK, "Independent scores, operating policy and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/tokens", Method: http.MethodPost, Description: "Inspect prepared token spans with original UTF-8 byte offsets and complete configured window scanning"}, policy, (*ClassificationAPIServer).handleModelDiagnosticTokens, strictJSONBodyFor[ModelTextDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticTokensResult]](http.StatusOK, "Token spans and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/embeddings", Method: http.MethodPost, Description: "Run the explicitly selected prepared embedding binding at its published representation"}, policy, (*ClassificationAPIServer).handleModelDiagnosticEmbedding, strictJSONBodyFor[ModelTextDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]](http.StatusOK, "Embedding vector and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
		managedRoute(EndpointMetadata{Path: path + "/rerank", Method: http.MethodPost, Description: "Score query-document pairs using the selected prepared relevance binding without running a RAG request; max_batch_size applies (default 100 pairs)"}, policy, (*ClassificationAPIServer).handleModelDiagnosticRerank, strictJSONBodyFor[ModelRerankDiagnosticRequest](), jsonResponse[ModelDiagnosticResponse[ModelDiagnosticRerankResult]](http.StatusOK, "Relevance logits in input order and actual binding identity"), errorResponses(400, 404, 409, 413, 429, 503, 500)),
	}
}

func (s *ClassificationAPIServer) acquireModelDiagnostics(recipe string) (*config.RouterConfig, *native.Runtime, func(), error) {
	cfg, service, releaseGeneration := s.acquireClassificationRuntime()
	source, ok := service.(interface {
		AcquireModelDiagnostics(string) (*native.Runtime, func(), error)
	})
	if !ok {
		releaseGeneration()
		return nil, nil, func() {}, binding.ErrNotPrepared
	}
	runtime, releaseService, err := source.AcquireModelDiagnostics(recipe)
	if err != nil {
		releaseGeneration()
		return nil, nil, func() {}, err
	}
	return cfg, runtime, func() { releaseService(); releaseGeneration() }, nil
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

func runModelTextDiagnostic[T any](s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request, call func(context.Context, *native.Runtime, ModelTextDiagnosticRequest) (ModelDiagnosticResponse[T], error)) {
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
	if err != nil && !errors.Is(err, tasks.ErrTokenSpansTruncated) {
		s.writeModelDiagnosticError(w, err)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleModelDiagnosticLabels(w http.ResponseWriter, r *http.Request) {
	runModelTextDiagnostic(s, w, r, func(ctx context.Context, runtime *native.Runtime, req ModelTextDiagnosticRequest) (ModelDiagnosticResponse[ModelDiagnosticLabelsResult], error) {
		result, err := runtime.DiagnoseLabels(ctx, req.Recipe, req.Binding, req.Text)
		out := ModelDiagnosticLabelsResult{}
		if d := result.Result.Distribution; d != nil {
			out.Probabilities = d.Probabilities
			out.Input = diagnosticInput(d.Input)
		}
		if d := result.Result.Windows; d != nil {
			out.ContentTokens = d.ContentTokens
			out.Input = diagnosticInput(d.Input)
			for _, v := range d.Windows {
				out.Windows = append(out.Windows, ModelDiagnosticLabelWindow{Start: v.Start, End: v.End, Probabilities: v.Probabilities})
			}
		}
		return ModelDiagnosticResponse[ModelDiagnosticLabelsResult]{Binding: diagnosticBinding(result.Binding), Result: out}, err
	})
}

func (s *ClassificationAPIServer) handleModelDiagnosticScores(w http.ResponseWriter, r *http.Request) {
	runModelTextDiagnostic(s, w, r, func(ctx context.Context, runtime *native.Runtime, req ModelTextDiagnosticRequest) (ModelDiagnosticResponse[ModelDiagnosticScoresResult], error) {
		result, err := runtime.DiagnoseScores(ctx, req.Recipe, req.Binding, req.Text)
		d := result.Result
		out := ModelDiagnosticScoresResult{Scores: d.Scores, Input: diagnosticInput(d.Input), PolicySHA256: d.PolicySHA256, Thresholds: d.Thresholds, WindowRanges: d.WindowRanges}
		if d.Windows != nil {
			out.Input = diagnosticInput(d.Windows.Input)
			for _, v := range d.Windows.Windows {
				out.Windows = append(out.Windows, ModelDiagnosticScoreWindow{Start: v.Start, End: v.End, Scores: v.Scores})
			}
		}
		return ModelDiagnosticResponse[ModelDiagnosticScoresResult]{Binding: diagnosticBinding(result.Binding), Result: out}, err
	})
}

func (s *ClassificationAPIServer) handleModelDiagnosticTokens(w http.ResponseWriter, r *http.Request) {
	runModelTextDiagnostic(s, w, r, func(ctx context.Context, runtime *native.Runtime, req ModelTextDiagnosticRequest) (ModelDiagnosticResponse[ModelDiagnosticTokensResult], error) {
		result, err := runtime.DiagnoseTokens(ctx, req.Recipe, req.Binding, req.Text)
		d := result.Result
		out := ModelDiagnosticTokensResult{ScanIncomplete: errors.Is(err, tasks.ErrTokenSpansTruncated), Entities: []ModelDiagnosticEntity{}, Input: diagnosticInput(d.Spans.Input), ScoresAvailable: d.Spans.HasScores(), TruncatedAt: d.Spans.TruncatedAt, Windows: d.Windows, ContentTokens: d.ContentTokens}
		for _, v := range d.Spans.Entities {
			e := ModelDiagnosticEntity{Type: v.EntityType, Start: v.Start, End: v.End, Text: v.Text}
			if d.Spans.HasScores() {
				score := v.Confidence
				e.Confidence = &score
			}
			out.Entities = append(out.Entities, e)
		}
		return ModelDiagnosticResponse[ModelDiagnosticTokensResult]{Binding: diagnosticBinding(result.Binding), Result: out}, err
	})
}

func (s *ClassificationAPIServer) handleModelDiagnosticEmbedding(w http.ResponseWriter, r *http.Request) {
	runModelTextDiagnostic(s, w, r, func(ctx context.Context, runtime *native.Runtime, req ModelTextDiagnosticRequest) (ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult], error) {
		result, err := runtime.DiagnoseEmbedding(ctx, req.Recipe, req.Binding, req.Text)
		return ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]{Binding: diagnosticBinding(result.Binding), Result: ModelDiagnosticEmbeddingResult{Embedding: result.Result.Embedding, Input: diagnosticInput(result.Result.Input)}}, err
	})
}

func (s *ClassificationAPIServer) handleModelDiagnosticRerank(w http.ResponseWriter, r *http.Request) {
	var req ModelRerankDiagnosticRequest
	if err := s.parseStrictJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if strings.TrimSpace(req.Binding) == "" || len(req.Pairs) == 0 {
		s.writeModelDiagnosticError(w, binding.ErrInvalidInput)
		return
	}
	cfg, runtime, release, err := s.acquireModelDiagnostics(req.Recipe)
	if err != nil {
		s.writeModelDiagnosticError(w, err)
		return
	}
	limit := defaultModelDiagnosticBatchSize
	if cfg != nil && cfg.API.BatchClassification.MaxBatchSize > 0 {
		limit = cfg.API.BatchClassification.MaxBatchSize
	}
	if len(req.Pairs) > limit {
		release()
		s.writeModelDiagnosticError(w, fmt.Errorf("%w: pairs exceeds max_batch_size %d", binding.ErrInvalidInput, limit))
		return
	}
	pairs := make([]tasks.QueryDocument, len(req.Pairs))
	for i, p := range req.Pairs {
		pairs[i] = tasks.QueryDocument{Query: p.Query, Document: p.Document}
	}
	result, err := executeModelDiagnostic(w, r, release, func(ctx context.Context) (native.DiagnosticResult[tasks.RelevanceScores], error) {
		return runtime.DiagnoseRerank(ctx, req.Recipe, req.Binding, pairs)
	})
	if err != nil {
		s.writeModelDiagnosticError(w, err)
		return
	}
	out := ModelDiagnosticRerankResult{Scores: result.Result.Scores, ScoreType: "relevance_logit", Inputs: make([]*ModelDiagnosticInputUsage, len(result.Result.Inputs))}
	for i := range result.Result.Inputs {
		out.Inputs[i] = diagnosticInput(&result.Result.Inputs[i])
	}
	s.writeJSONResponse(w, http.StatusOK, ModelDiagnosticResponse[ModelDiagnosticRerankResult]{Binding: diagnosticBinding(result.Binding), Result: out})
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
// the actual inference. A canceled native forward cannot be unloaded early.
// Its prepared resource retains the one shared admission ticket; the API does
// not add a competing admission policy or discover another model instance.
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
