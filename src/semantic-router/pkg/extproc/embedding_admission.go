package extproc

import (
	"context"
	"errors"
	"net/http"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

const embeddingOverloadMessage = "router inference is temporarily overloaded"

func (r *OpenAIRouter) embeddingProcessAdmission() *embedding.ProcessAdmission {
	if r != nil && r.embeddingAdmission != nil {
		return r.embeddingAdmission
	}
	return embedding.DefaultProcessAdmission
}

func (r *OpenAIRouter) admitDecisionEvaluation(
	ctx context.Context,
	originalModel string,
	imageURL string,
) (func(), error) {
	if !r.decisionEvaluationUsesLocalNativeEmbeddings(originalModel, imageURL != "") {
		return func() {}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return r.embeddingProcessAdmission().TryAcquire(ctx)
}

func (r *OpenAIRouter) decisionEvaluationUsesLocalNativeEmbeddings(originalModel string, hasImage bool) bool {
	if r != nil && r.Classifier != nil && r.Classifier.UsesLocalNativeEmbeddings(hasImage) {
		return true
	}
	if r == nil {
		return false
	}
	plan, err := r.resolvedEmbeddingRuntimePlan()
	if err != nil {
		// Router construction rejects an invalid runtime plan. Keep this
		// request-time fallback conservative as defense in depth for tests or
		// callers that assemble OpenAIRouter directly: an invalid plan must not
		// bypass the shared native-inference admission gate.
		return true
	}
	return configDecisionEvaluationUsesLocalNativeEmbeddings(r.Config, originalModel, plan)
}

func configDecisionEvaluationUsesLocalNativeEmbeddings(
	routerConfig *config.RouterConfig,
	originalModel string,
	plan embedding.RuntimePlan,
) bool {
	if routerConfig == nil || !routerConfig.IsAutoModelName(originalModel) {
		return false
	}

	if plan.Backend == config.EmbeddingBackendOpenAICompatible {
		return false
	}
	return configuredDecisionsUseEmbeddings(routerConfig.Decisions)
}

func configuredDecisionsUseEmbeddings(decisions []config.Decision) bool {
	for _, decision := range decisions {
		if len(decision.ModelRefs) < 2 || decision.Algorithm == nil {
			continue
		}
		if algorithmUsesEmbeddings(decision.Algorithm.Type) {
			return true
		}
	}
	return false
}

func algorithmUsesEmbeddings(algorithmType string) bool {
	switch algorithmType {
	case "router_dc", "hybrid", "knn", "kmeans", "svm", "mlp":
		return true
	default:
		return false
	}
}

func (r *OpenAIRouter) classificationEvaluationErrorResponse(err error) *ext_proc.ProcessingResponse {
	if errors.Is(err, embedding.ErrOverloaded) {
		response := r.createErrorResponse(http.StatusServiceUnavailable, embeddingOverloadMessage)
		if immediate := response.GetImmediateResponse(); immediate != nil {
			immediate.Headers.SetHeaders = append(
				immediate.Headers.SetHeaders,
				&core.HeaderValueOption{Header: &core.HeaderValue{Key: "cache-control", RawValue: []byte("no-store")}},
				&core.HeaderValueOption{Header: &core.HeaderValue{Key: "retry-after", RawValue: []byte("1")}},
			)
		}
		return response
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return r.createErrorResponse(http.StatusServiceUnavailable, "request canceled")
	}
	if errors.Is(err, classification.ErrInvalidImageSignalInput) {
		return r.createNonCacheableEvaluationError(
			http.StatusBadRequest,
			"image input must contain a decodable JPEG or PNG image within the supported limits",
		)
	}
	if errors.Is(err, classification.ErrImageSignalEvaluation) {
		return r.createNonCacheableEvaluationError(
			http.StatusInternalServerError,
			"router image evaluation failed",
		)
	}
	if errors.Is(err, classification.ErrTextSignalEvaluation) {
		response := r.createNonCacheableEvaluationError(
			http.StatusServiceUnavailable,
			"router signal evaluation temporarily unavailable",
		)
		if immediate := response.GetImmediateResponse(); immediate != nil {
			immediate.Headers.SetHeaders = append(
				immediate.Headers.SetHeaders,
				&core.HeaderValueOption{Header: &core.HeaderValue{Key: "retry-after", RawValue: []byte("1")}},
			)
		}
		return response
	}
	return r.createNonCacheableEvaluationError(http.StatusInternalServerError, "router evaluation failed")
}

func (r *OpenAIRouter) createNonCacheableEvaluationError(statusCode int, message string) *ext_proc.ProcessingResponse {
	response := r.createErrorResponse(statusCode, message)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		immediate.Headers.SetHeaders = append(
			immediate.Headers.SetHeaders,
			&core.HeaderValueOption{Header: &core.HeaderValue{Key: "cache-control", RawValue: []byte("no-store")}},
		)
	}
	return response
}
