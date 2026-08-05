package extproc

import (
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func (r *OpenAIRouter) maybeStoreResponseAPIStreamingResponse(ctx *RequestContext) {
	if r == nil || r.ResponseAPIFilter == nil || !isResponseAPIRequest(ctx) {
		return
	}
	response := responseAPIStreamingResponseForStore(ctx)
	if response == nil {
		return
	}
	r.ResponseAPIFilter.maybeStoreTranslatedResponse(ctx.TraceContext, ctx.ResponseAPICtx, response)
}

func responseAPIStreamingResponseForStore(ctx *RequestContext) *responseapi.ResponseAPIResponse {
	if ctx == nil || ctx.ResponseAPICtx == nil || ctx.ResponseAPICtx.OriginalRequest == nil {
		return nil
	}
	req := ctx.ResponseAPICtx.OriginalRequest
	if ctx.ResponseAPICtx.GeneratedResponseID == "" {
		ctx.ResponseAPICtx.GeneratedResponseID = responseapi.GenerateResponseID()
	}
	if ctx.ResponseAPIStreamCreatedAt == 0 {
		ctx.ResponseAPIStreamCreatedAt = time.Now().Unix()
	}

	output, outputText := responseAPIStreamingStoredOutput(ctx)
	usage := extractStreamingUsage(ctx)
	resp := &responseapi.ResponseAPIResponse{
		ID:                 ctx.ResponseAPICtx.GeneratedResponseID,
		Object:             "response",
		CreatedAt:          ctx.ResponseAPIStreamCreatedAt,
		Model:              responseAPIStreamModel(ctx),
		Status:             responseapi.StatusCompleted,
		Output:             output,
		OutputText:         outputText,
		PreviousResponseID: req.PreviousResponseID,
		ConversationID:     ctx.ResponseAPICtx.ConversationID,
		Usage: &responseapi.Usage{
			InputTokens:  int(usage.PromptTokens),
			OutputTokens: int(usage.CompletionTokens),
			TotalTokens:  int(usage.TotalTokens),
		},
		Instructions:    req.Instructions,
		Metadata:        req.Metadata,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		MaxOutputTokens: req.MaxOutputTokens,
		Tools:           req.Tools,
		ToolChoice:      req.ToolChoice,
	}
	if resp.ConversationID == "" {
		resp.ConversationID = req.ConversationID
	}
	if ctx.StreamingReasoning != "" {
		resp.Reasoning = &responseapi.Reasoning{
			EncryptedContent: ctx.StreamingReasoning,
		}
	}
	return resp
}

func responseAPIStreamingStoredOutput(ctx *RequestContext) ([]responseapi.OutputItem, string) {
	if ctx == nil {
		return nil, ""
	}
	indexed := []responseAPIIndexedStoredOutput{}
	if responseAPIStreamNeedsMessageItem(ctx) {
		item := responseapi.OutputItem{
			Type:   responseapi.ItemTypeMessage,
			ID:     ctx.ResponseAPIStreamItemID,
			Role:   responseapi.RoleAssistant,
			Status: responseapi.StatusCompleted,
		}
		if item.ID == "" {
			item.ID = responseapi.GenerateItemID()
		}
		switch {
		case ctx.StreamingContent != "":
			item.Content = []responseapi.ContentPart{{
				Type:        responseapi.ContentTypeOutputText,
				Text:        ctx.StreamingContent,
				Annotations: []responseapi.Annotation{},
			}}
		case ctx.StreamingRefusal != "":
			item.Content = []responseapi.ContentPart{{
				Type: "refusal",
				Text: ctx.StreamingRefusal,
			}}
		}
		indexed = append(indexed, responseAPIIndexedStoredOutput{
			index: responseAPIMessageOutputIndex(ctx),
			item:  item,
		})
	}
	for _, index := range responseAPISortedToolCallIndexes(ctx) {
		state := ctx.StreamingToolCalls[index]
		indexed = append(indexed, responseAPIIndexedStoredOutput{
			index: responseAPIToolCallOutputIndex(ctx, index),
			item: responseapi.OutputItem{
				Type:      responseapi.ItemTypeFunctionCall,
				ID:        responseAPIStoredToolCallItemID(ctx, index),
				CallID:    responseAPIToolCallID(state),
				Name:      responseAPIToolCallName(state),
				Arguments: responseAPIToolCallArguments(state),
				Status:    responseapi.StatusCompleted,
			},
		})
	}
	sort.Slice(indexed, func(i, j int) bool { return indexed[i].index < indexed[j].index })
	output := make([]responseapi.OutputItem, 0, len(indexed))
	for _, entry := range indexed {
		output = append(output, entry.item)
	}
	return output, ctx.StreamingContent
}

type responseAPIIndexedStoredOutput struct {
	index int
	item  responseapi.OutputItem
}

func responseAPIStoredToolCallItemID(ctx *RequestContext, index int) string {
	if ctx.ResponseAPIStreamToolCallItemIDs != nil && ctx.ResponseAPIStreamToolCallItemIDs[index] != "" {
		return ctx.ResponseAPIStreamToolCallItemIDs[index]
	}
	return responseapi.GenerateItemID()
}
