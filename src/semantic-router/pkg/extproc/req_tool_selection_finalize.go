package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// toolSelectionResult is one retrieval strategy's ranked proposal. Strategies
// only propose; finalizeToolSelection decides what reaches the request, so
// every selection path shares one emission, fallback, and sticky boundary.
type toolSelectionResult struct {
	// tools are the ranked candidates, best first.
	tools              []llmprotocol.Tool
	strategyID         string
	confidence         float32
	latency            time.Duration
	classificationText string
	err                error
	// fallbackOverride is the plugin's fallback_to_empty for an empty
	// selection; nil uses the global setting.
	fallbackOverride *bool
	// errorFallbackToEmpty is the effective fallback for a retrieval error.
	errorFallbackToEmpty bool
}

// finalizeToolSelection reports the strategy's observability and publishes
// its result. A sticky scope routes the result through session-state
// reconciliation; otherwise the ordinary stateless selection applies.
func (r *OpenAIRouter) finalizeToolSelection(
	request *llmprotocol.Request,
	response **ext_proc.ProcessingResponse,
	ctx *RequestContext,
	result toolSelectionResult,
	sticky *stickyToolScope,
) error {
	emitToolObservability(response, ctx, result.strategyID, result.confidence, result.latency)
	metrics.RecordToolsRetrieval(result.strategyID, result.latency.Seconds())
	if sticky != nil {
		return r.finalizeStickyToolSelection(request, ctx, result, sticky)
	}
	if result.err != nil {
		return r.handleToolSelectionError(request, response, ctx, result.err, result.errorFallbackToEmpty)
	}
	if err := r.applySelectedTools(request, result.tools, result.strategyID, result.confidence, result.latency, result.classificationText, result.fallbackOverride); err != nil {
		return err
	}
	return commitToolSelection(request, ctx)
}
