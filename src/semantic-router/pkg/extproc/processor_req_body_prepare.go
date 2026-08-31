package extproc

import (
	"context"
	"errors"
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) extractRequestSignalSnapshot(
	ctx *RequestContext,
) (*requestSignalSnapshot, error) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil, status.Error(codes.InvalidArgument, "neutral inference request is unavailable")
	}
	snapshot := extractSemanticRequestSignals(ctx.SemanticRequest)
	if snapshot.Stream {
		logging.ComponentDebugEvent("extproc", "stream_parameter_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
		})
		ctx.ExpectStreamingResponse = true
	}
	return snapshot, nil
}

func (r *OpenAIRouter) runRequestPreRoutingStages(
	originalModel string,
	snapshot *requestSignalSnapshot,
	ctx *RequestContext,
) (requestDecisionState, *ext_proc.ProcessingResponse) {
	if !ctx.Routing.IsResolved() {
		r.resolveEntrypointForRequest(originalModel, ctx)
	}
	populatePinnedSessionFromHeaders(ctx)
	history := signalConversationHistoryFromSnapshot(snapshot)
	applyRequestContextEstimate(snapshot, ctx)
	decisionName, _, reasoningDecision, selectedModel, decisionErr := r.performDecisionEvaluation(
		originalModel,
		history,
		ctx,
	)
	if decisionErr != nil {
		if errors.Is(decisionErr, context.Canceled) ||
			errors.Is(decisionErr, context.DeadlineExceeded) {
			return requestDecisionState{}, r.createErrorResponse(499, "request canceled")
		}
		if errors.Is(decisionErr, errNoContextEligibleDecisionModel) {
			logging.Warnf("[Request Body] Decision candidates cannot satisfy request context: %v", decisionErr)
			return requestDecisionState{}, r.createErrorResponse(422, decisionErr.Error())
		}
		logging.Errorf("[Request Body] Decision evaluation failed: %v", decisionErr)
		return requestDecisionState{}, r.createErrorResponse(403, decisionErr.Error())
	}
	metrics.RecordModelRequest(selectedModel)
	ctx.InflightToken = inflight.Begin(selectedModel)
	if resp := r.handleFastResponse(ctx, decisionName); resp != nil {
		inflight.End(selectedModel, ctx.InflightToken)
		ctx.InflightToken = 0
		r.startRouterReplay(ctx, originalModel, selectedModel, decisionName)
		r.updateRouterReplayStatus(ctx, 200, false)
		r.attachRouterReplayResponse(
			ctx,
			resp.GetImmediateResponse().GetBody(),
			true,
		)
		addRouterReplayHeaderToImmediateResponse(resp, ctx.RouterReplayID)
		return requestDecisionState{}, resp
	}
	if resp := r.applyRateLimit(ctx, selectedModel); resp != nil {
		inflight.End(selectedModel, ctx.InflightToken)
		ctx.InflightToken = 0
		return requestDecisionState{}, resp
	}
	if resp := r.applyCacheChecks(ctx, selectedModel, decisionName); resp != nil {
		inflight.End(selectedModel, ctx.InflightToken)
		ctx.InflightToken = 0
		return requestDecisionState{}, resp
	}
	if ragErr := r.executeRAGPlugin(ctx, decisionName); ragErr != nil {
		inflight.End(selectedModel, ctx.InflightToken)
		ctx.InflightToken = 0
		return requestDecisionState{}, r.createErrorResponse(503, fmt.Sprintf("RAG retrieval failed: %v", ragErr))
	}

	return requestDecisionState{
		decisionName:      decisionName,
		reasoningDecision: reasoningDecision,
		selectedModel:     selectedModel,
	}, nil
}

func applyRequestContextEstimate(snapshot *requestSignalSnapshot, ctx *RequestContext) {
	if snapshot == nil || ctx == nil {
		return
	}
	ctx.VSRContextTokenCount = snapshot.ContextTokenFloor
	ctx.VSRContextTextBytes = snapshot.ContextTextBytes
	ctx.VSRContextEquivalentBytes = snapshot.ContextEquivalentBytes
	ctx.VSRContextHasNonText = snapshot.ContextHasNonText
}

func (r *OpenAIRouter) applyCacheChecks(
	ctx *RequestContext,
	selectedModel string,
	decisionName string,
) *ext_proc.ProcessingResponse {
	if response, shouldReturn := r.handleCaching(ctx, decisionName, selectedModel); shouldReturn {
		logging.ComponentDebugEvent("extproc", "cache_short_circuit", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   decisionName,
		})
		return response
	}
	return nil
}

func (r *OpenAIRouter) prepareRequestForModelRouting(
	request *llmprotocol.Request,
	userContent string,
	ctx *RequestContext,
) (*llmprotocol.Request, *ext_proc.ProcessingResponse, error) {
	if request == nil {
		return nil, nil, status.Error(codes.InvalidArgument, "neutral inference request is unavailable")
	}
	populateSessionTransitionFields(ctx)
	memErr := r.handleMemoryRetrieval(ctx, userContent, request)
	if memErr != nil {
		logging.ComponentWarnEvent("extproc", "memory_retrieval_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      memErr.Error(),
			"fallback":   "continue_without_memory",
		})
	}
	if compressionErr := r.applySemanticContextCompression(ctx, request); compressionErr != nil {
		return nil, r.createErrorResponse(500, "Context compression failed under fail_closed policy"), nil
	}
	return request, nil, nil
}
