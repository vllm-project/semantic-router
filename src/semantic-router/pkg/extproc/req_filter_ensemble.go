package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ensemble"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleEnsembleRequest processes a request with ensemble orchestration
func (r *OpenAIRouter) handleEnsembleRequest(ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	// Check if ensemble is enabled and requested
	if !ctx.EnsembleEnabled || r.EnsembleFactory == nil {
		return nil, false
	}

	// Validate that we have models to query
	if len(ctx.EnsembleModels) == 0 {
		logging.Warnf("Ensemble requested but no models specified")
		return nil, false
	}

	logging.Infof("Processing ensemble request with %d models: %v", len(ctx.EnsembleModels), ctx.EnsembleModels)

	// Build ensemble request
	ensembleReq := &ensemble.Request{
		Models:           ctx.EnsembleModels,
		Strategy:         ensemble.Strategy(ctx.EnsembleStrategy),
		MinResponses:     ctx.EnsembleMinResponses,
		OriginalRequest:  ctx.OriginalRequestBody,
		Context:          ctx.TraceContext,
	}

	// Execute ensemble orchestration
	ensembleResp := r.EnsembleFactory.Execute(ensembleReq)

	// Track ensemble metadata
	ctx.VSREnsembleUsed = true
	ctx.VSREnsembleModelsQueried = ensembleResp.ModelsQueried
	ctx.VSREnsembleResponsesReceived = ensembleResp.ResponsesReceived

	// Check for errors
	if ensembleResp.Error != nil {
		logging.Errorf("Ensemble execution failed: %v", ensembleResp.Error)
		errorMsg := fmt.Sprintf("Ensemble orchestration failed: %v", ensembleResp.Error)
		return r.createErrorResponse(500, errorMsg), true
	}

	// Return the aggregated response
	logging.Infof("Ensemble execution successful: queried=%d, received=%d, strategy=%s",
		ensembleResp.ModelsQueried, ensembleResp.ResponsesReceived, ensembleResp.Strategy)

	return r.createJSONResponseWithBody(200, ensembleResp.FinalResponse), true
}
