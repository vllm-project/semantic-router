package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// projectAnthropicRequestForBackend keeps the public request intact while
// deriving the controls that a different backend wire format can honor. The
// same projection is used for selection, final qualification, and encoding.
func (r *OpenAIRouter) projectAnthropicRequestForBackend(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, error) {
	projected, _, err := r.projectAnthropicRequestForBackendWithDiagnostics(request, model, target)
	return projected, err
}

func (r *OpenAIRouter) projectAnthropicRequestForBackendWithDiagnostics(
	request llmprotocol.Request,
	model string,
	target llmprotocol.WireFormat,
) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
	if request.Trusted.SourceFormat != llmprotocol.AnthropicMessagesV1 ||
		(target != llmprotocol.OpenAIChatV1 && target != llmprotocol.OpenAIResponsesV1) {
		return request, nil, nil
	}
	projected := request
	changed := false
	if len(request.ContextManagement) > 0 {
		if !noopAnthropicContextManagement(request.ContextManagement) {
			return request, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "lossy_translation",
				"the selected backend cannot apply Anthropic context_management edits", nil)
		}
		projected.ContextManagement = nil
		changed = true
	}
	switch request.ReasoningMode {
	case llmprotocol.ReasoningModeAdaptive:
		// Adaptive means the model decides. An OpenAI-compatible backend's
		// default is the only portable representation; effort still translates.
		projected.ReasoningMode = ""
		changed = true
	case llmprotocol.ReasoningModeDisabled:
		// A generic Chat or Responses endpoint has no portable off switch.
		// Its configured family must provide a provider-specific one.
		if !anthropicBackendCanDisableReasoning(r.getModelReasoningFamily(model)) {
			return request, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
				fmt.Sprintf("model %q has no configured reasoning-off control", model), nil)
		}
		projected.ReasoningMode = ""
		changed = true
	}
	if request.ReasoningDisplay != "" {
		if request.ReasoningDisplay != "omitted" {
			return request, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_reasoning_display",
				"the selected backend cannot produce summarized reasoning", nil)
		}
		// The Router enforces omitted on the client response, both buffered
		// and streaming, instead of relying on a backend display control.
		projected.ReasoningDisplay = ""
		changed = true
	}
	if changed {
		projected.Generation++ // The source envelope no longer describes this request.
	}
	projected, diagnostics := llmprotocol.ProjectAnthropicCacheDirectives(projected, target)
	return projected, diagnostics, nil
}

func anthropicBackendCanDisableReasoning(family *config.ReasoningFamilyConfig) bool {
	if !reasoningFamilyCanDisable(family) ||
		(len(family.Modes) > 0 && !reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeDisabled))) {
		return false
	}
	// A top-level effort with no disabled sentinel cannot express off even if
	// the family metadata advertises the mode.
	return family.Type != config.ReasoningFamilyTypeTopLevelReasoningEffort || family.Disabled != ""
}

// Only keep:all has no effect. Restrict the accepted shape so a future edit
// option cannot be silently lost by an OpenAI-compatible backend.
func noopAnthropicContextManagement(raw json.RawMessage) bool {
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil || len(object) != 1 {
		return false
	}
	rawEdits, found := object["edits"]
	rawEdits = bytes.TrimSpace(rawEdits)
	if !found || len(rawEdits) == 0 || rawEdits[0] != '[' {
		return false
	}
	var edits []map[string]json.RawMessage
	if json.Unmarshal(rawEdits, &edits) != nil {
		return false
	}
	for _, edit := range edits {
		if len(edit) != 2 {
			return false
		}
		var kind, keep string
		if json.Unmarshal(edit["type"], &kind) != nil ||
			json.Unmarshal(edit["keep"], &keep) != nil ||
			kind != "clear_thinking_20251015" || keep != "all" {
			return false
		}
	}
	return true
}

func omitCrossFormatReasoning(ctx *RequestContext, backend llmprotocol.WireFormat) bool {
	return ctx != nil && ctx.SemanticRequest != nil &&
		ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 &&
		backend != "" && backend != llmprotocol.AnthropicMessagesV1 &&
		ctx.SemanticRequest.ReasoningDisplay == "omitted"
}

func explicitAnthropicReasoningDisabled(ctx *RequestContext, dispatch *providerDispatch) bool {
	return ctx != nil && dispatch != nil && ctx.SemanticRequest != nil &&
		ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 &&
		ctx.SemanticRequest.ReasoningMode == llmprotocol.ReasoningModeDisabled &&
		dispatch.targetFormat != llmprotocol.AnthropicMessagesV1
}

func preserveExplicitAnthropicReasoning(request *llmprotocol.Request, target llmprotocol.WireFormat) bool {
	return request != nil && request.Trusted.SourceFormat == llmprotocol.AnthropicMessagesV1 &&
		target != llmprotocol.AnthropicMessagesV1 &&
		(request.ReasoningMode == llmprotocol.ReasoningModeAdaptive || request.ReasoningMode == llmprotocol.ReasoningModeDisabled)
}

func clientResponseMutation(ctx *RequestContext, backend llmprotocol.WireFormat) protocolcodec.ResponseMutation {
	responseID := responseObjectPublicID(ctx)
	omitReasoning := omitCrossFormatReasoning(ctx, backend)
	if responseID == "" && !omitReasoning {
		return nil
	}
	return func(response *llmprotocol.Response) error {
		if responseID != "" {
			response.ID = responseID
		}
		if omitReasoning {
			omitResponseReasoning(response)
		}
		return nil
	}
}

func clientStreamMutation(ctx *RequestContext, backend llmprotocol.WireFormat) protocolcodec.StreamEventMutation {
	responseID := responseObjectPublicID(ctx)
	omitReasoning := omitCrossFormatReasoning(ctx, backend)
	if responseID == "" && !omitReasoning {
		return nil
	}
	hiddenItems := make(map[int]struct{})
	return func(event *llmprotocol.Event) error {
		if responseID != "" {
			event.ResponseID = responseID
		}
		if !omitReasoning {
			return nil
		}
		switch event.Type {
		case llmprotocol.EventOutputItemStarted:
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
				hiddenItems[event.ItemIndex] = struct{}{}
				return protocolcodec.ErrOmitStreamEvent
			}
		case llmprotocol.EventReasoningDelta:
			return protocolcodec.ErrOmitStreamEvent
		case llmprotocol.EventOutputItemCompleted:
			if _, hidden := hiddenItems[event.ItemIndex]; hidden {
				delete(hiddenItems, event.ItemIndex)
				return protocolcodec.ErrOmitStreamEvent
			}
		}
		return nil
	}
}

func omitResponseReasoning(response *llmprotocol.Response) {
	if response == nil {
		return
	}
	response.Output = withoutReasoningOutputItems(response.Output)
	for alternative := range response.Alternatives {
		response.Alternatives[alternative] = withoutReasoningOutputItems(response.Alternatives[alternative])
	}
}

func withoutReasoningOutputItems(items []llmprotocol.OutputItem) []llmprotocol.OutputItem {
	kept := make([]llmprotocol.OutputItem, 0, len(items))
	for _, item := range items {
		content := withoutReasoningContent(item.Content)
		if len(item.Content) > 0 && len(content) == 0 {
			continue
		}
		item.Content = content
		kept = append(kept, item)
	}
	return kept
}

func withoutReasoningContent(content []llmprotocol.Content) []llmprotocol.Content {
	kept := make([]llmprotocol.Content, 0, len(content))
	for _, block := range content {
		if block.Kind != llmprotocol.ContentReasoning {
			kept = append(kept, block)
		}
	}
	return kept
}
