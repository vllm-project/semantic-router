package extproc

import (
	"encoding/json"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func responseObjectPublicID(ctx *RequestContext) string {
	if ctx == nil || ctx.SourceFormat != llmprotocol.OpenAIResponsesV1 || ctx.ResponseObjectState == nil {
		return ""
	}
	return strings.TrimSpace(ctx.ResponseObjectState.GeneratedResponseID)
}

func responseObjectPreviousID(ctx *RequestContext) string {
	if ctx == nil || ctx.SourceFormat != llmprotocol.OpenAIResponsesV1 || ctx.ResponseObjectState == nil {
		return ""
	}
	return strings.TrimSpace(ctx.ResponseObjectState.PreviousResponseID)
}

// persistImmediateResponseObject retains successful inference short-circuits
// after their final client representation has been selected.
func (r *OpenAIRouter) persistImmediateResponseObject(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response == nil {
		return
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return
	}
	status := int(immediate.GetStatus().GetCode())
	if status < 200 || status >= 300 {
		return
	}
	r.persistResponseObject(ctx)
}

// persistResponseObject retains the final neutral Responses result. Retention
// is optional and never changes an otherwise successful inference response.
//
//nolint:cyclop // Persistence maps the complete neutral response lifecycle into one stored object transaction.
func (r *OpenAIRouter) persistResponseObject(ctx *RequestContext) {
	if r == nil || r.ResponseAPIFilter == nil || !r.ResponseAPIFilter.IsEnabled() ||
		ctx == nil || ctx.SourceFormat != llmprotocol.OpenAIResponsesV1 ||
		ctx.ResponseObjectState == nil || ctx.SemanticResponse == nil {
		return
	}
	state := ctx.ResponseObjectState
	if state.PersistenceAttempted {
		return
	}
	state.PersistenceAttempted = true
	if ctx.UpstreamStatusCode != 0 && (ctx.UpstreamStatusCode < 200 || ctx.UpstreamStatusCode >= 300) {
		return
	}
	if !state.ShouldStore || ctx.SemanticResponse.Error != nil ||
		strings.TrimSpace(ctx.SemanticResponse.ID) == "" {
		return
	}

	engine, err := r.protocolEngine()
	if err != nil {
		logResponseObjectPersistenceFailure(ctx, err)
		return
	}
	encoded, err := engine.EncodeResponse(
		llmprotocol.OpenAIResponsesV1,
		*ctx.SemanticResponse,
		llmprotocol.Envelope{},
	)
	if err != nil {
		logResponseObjectPersistenceFailure(ctx, err)
		return
	}
	var response responseapi.ResponseAPIResponse
	if err := json.Unmarshal(encoded.Body, &response); err != nil {
		logResponseObjectPersistenceFailure(ctx, err)
		return
	}
	response.PreviousResponseID = state.PreviousResponseID
	response.ConversationID = state.ConversationID
	response.Instructions = state.Instructions
	response.Metadata = cloneResponseMetadata(state.Metadata)
	if response.OutputText == "" {
		response.OutputText = responseObjectOutputText(response.Output)
	}

	stored := &responseapi.StoredResponse{
		ID: response.ID, Object: "response",
		CreatedAt: response.CreatedAt, Model: response.Model, Status: response.Status,
		Input: cloneResponseInputItems(state.Input), Output: response.Output,
		OutputText: response.OutputText, PreviousResponseID: response.PreviousResponseID,
		ConversationID: response.ConversationID, Usage: response.Usage,
		Instructions: response.Instructions, Metadata: response.Metadata, Error: response.Error,
	}
	if err := r.ResponseAPIFilter.store.StoreResponse(ctx.TraceContext, stored); err != nil {
		logResponseObjectPersistenceFailure(ctx, err)
	}
}

func cloneResponseInputItems(items []responseapi.InputItem) []responseapi.InputItem {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]responseapi.InputItem, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].Content = append(json.RawMessage(nil), cloned[index].Content...)
		cloned[index].Output = append(json.RawMessage(nil), cloned[index].Output...)
		cloned[index].Summary = append(json.RawMessage(nil), cloned[index].Summary...)
	}
	return cloned
}

func responseObjectOutputText(items []responseapi.OutputItem) string {
	var result strings.Builder
	for _, item := range items {
		for _, content := range item.Content {
			if content.Type == responseapi.ContentTypeOutputText {
				result.WriteString(content.Text)
			}
		}
	}
	return result.String()
}

func logResponseObjectPersistenceFailure(ctx *RequestContext, err error) {
	fields := map[string]interface{}{"error": err.Error()}
	if ctx != nil {
		fields["request_id"] = ctx.RequestID
	}
	logging.ComponentWarnEvent("extproc", "response_object_persistence_failed", fields)
}
