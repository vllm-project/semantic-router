package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func captureOriginalContextHistory(ctx *RequestContext) {
	if ctx.OriginalContextHistory == nil {
		ctx.OriginalContextHistory = contextcompression.CaptureHistory(ctx.SemanticRequest)
	}
}

func contextRequestIR(ctx *RequestContext, request *llmprotocol.Request) *contextcompression.RequestIR {
	if ctx.ContextRequestIR == nil {
		ctx.ContextRequestIR = contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{
			RAGToolCallIDs:       ctx.RAGToolCallIDs,
			MemoryMessageIndexes: ctx.MemoryMessageIndexes,
			OriginalHistory:      ctx.OriginalContextHistory,
			ProtectedMessages:    ctx.ProtectedContextMessages,
		})
	}
	return ctx.ContextRequestIR
}

// applyContextTransformationPlan runs after RAG and Memory enrichment. Child
// policies register history steps during decision preparation; compression is
// always last and continues to use the existing plugin configuration.
func (r *OpenAIRouter) applyContextTransformationPlan(ctx *RequestContext, request *llmprotocol.Request) error {
	ir := contextRequestIR(ctx, request)
	callContext := ctx.TraceContext
	if callContext == nil {
		callContext = context.Background()
	}
	if err := ir.ApplySteps(callContext, ctx.ContextHistorySteps); err != nil {
		return err
	}
	if err := r.applySemanticContextCompression(ctx, request); err != nil {
		return err
	}
	return ir.SkipCompression(callContext)
}
