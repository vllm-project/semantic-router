package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	promptCacheActionInserted  = "inserted"
	promptCacheActionPreserved = "preserved"
	promptCacheActionRejected  = "rejected"
	promptCacheActionSkipped   = "skipped"

	promptCacheReasonCallerMarkers     = "caller_markers"
	promptCacheReasonNoEligibleTarget  = "no_eligible_target"
	promptCacheReasonUnsupportedTarget = "unsupported_target"

	promptCacheErrorTargetUnsupported = "prompt_cache_target_unsupported"
)

func (r *OpenAIRouter) applyPromptCachePolicy(
	request *llmprotocol.Request,
	ctx *RequestContext,
	targetFormat llmprotocol.WireFormat,
) error {
	if request == nil || ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	plugin := ctx.VSRSelectedDecision.GetPromptCacheConfig()
	if plugin == nil || !plugin.Enabled {
		return nil
	}

	start := time.Now()
	preserved := countPromptCacheMarkers(*request)
	if !r.promptCacheTargetSupported(targetFormat) {
		if plugin.EffectiveOnUnsupported() == config.PromptCacheUnsupportedReject {
			recordPromptCacheReceipt(
				ctx,
				promptCacheActionRejected,
				promptCacheReasonUnsupportedTarget,
				0,
				preserved,
				time.Since(start),
			)
			return llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				promptCacheErrorTargetUnsupported,
				"prompt cache marker injection requires an Anthropic target",
				nil,
			)
		}
		recordPromptCacheReceipt(
			ctx,
			promptCacheActionSkipped,
			promptCacheReasonUnsupportedTarget,
			0,
			preserved,
			time.Since(start),
		)
		return nil
	}

	if preserved > 0 {
		recordPromptCacheReceipt(
			ctx,
			promptCacheActionPreserved,
			promptCacheReasonCallerMarkers,
			0,
			preserved,
			time.Since(start),
		)
		return nil
	}

	inserted := injectPromptCacheMarkers(
		request,
		plugin.EffectiveTargets(),
		plugin.EffectiveTTL(),
	)

	if inserted > 0 {
		request.Generation++
		recordPromptCacheReceipt(
			ctx,
			promptCacheActionInserted,
			"",
			inserted,
			preserved,
			time.Since(start),
		)
		return nil
	}

	recordPromptCacheReceipt(
		ctx,
		promptCacheActionSkipped,
		promptCacheReasonNoEligibleTarget,
		0,
		preserved,
		time.Since(start),
	)
	return nil
}

func injectPromptCacheMarkers(
	request *llmprotocol.Request,
	targets []string,
	ttl string,
) int {
	inserted := 0
	for _, target := range targets {
		switch target {
		case config.PromptCacheTargetInstructions:
			if injectInstructionPromptCacheMarker(request, ttl) {
				inserted++
			}
		case config.PromptCacheTargetTools:
			if injectToolPromptCacheMarker(request, ttl) {
				inserted++
			}
		}
	}
	return inserted
}

func (r *OpenAIRouter) promptCacheTargetSupported(format llmprotocol.WireFormat) bool {
	if format != llmprotocol.AnthropicMessagesV1 {
		return false
	}
	capabilities, ok := r.codecCapabilitiesForFormat(format)
	return ok && capabilities.Supports(llmprotocol.CapabilityCacheDirectives)
}

func injectInstructionPromptCacheMarker(
	request *llmprotocol.Request,
	ttl string,
) bool {
	for instructionIndex := len(request.Instructions) - 1; instructionIndex >= 0; instructionIndex-- {
		content := request.Instructions[instructionIndex].Content
		for contentIndex := len(content) - 1; contentIndex >= 0; contentIndex-- {
			if content[contentIndex].Kind != llmprotocol.ContentText ||
				content[contentIndex].Text == "" ||
				content[contentIndex].Cache != nil {
				continue
			}
			instructions := append([]llmprotocol.InstructionBlock(nil), request.Instructions...)
			instructions[instructionIndex].Content = append([]llmprotocol.Content(nil), content...)
			instructions[instructionIndex].Content[contentIndex].Cache = promptCacheDirective(ttl)
			request.Instructions = instructions
			return true
		}
	}
	return false
}

func injectToolPromptCacheMarker(request *llmprotocol.Request, ttl string) bool {
	for index := len(request.Tools) - 1; index >= 0; index-- {
		if request.Tools[index].Cache != nil {
			continue
		}
		tools := append([]llmprotocol.Tool(nil), request.Tools...)
		tools[index].Cache = promptCacheDirective(ttl)
		request.Tools = tools
		return true
	}
	return false
}

func promptCacheDirective(ttl string) *llmprotocol.CacheDirective {
	return &llmprotocol.CacheDirective{Type: "ephemeral", TTL: ttl}
}

func countPromptCacheMarkers(request llmprotocol.Request) int {
	count := 0
	for _, instruction := range request.Instructions {
		count += countContentPromptCacheMarkers(instruction.Content)
	}
	for _, message := range request.Messages {
		count += countContentPromptCacheMarkers(message.Content)
	}
	for _, tool := range request.Tools {
		if tool.Cache != nil {
			count++
		}
	}
	return count
}

func countContentPromptCacheMarkers(contents []llmprotocol.Content) int {
	count := 0
	for _, content := range contents {
		if content.Cache != nil {
			count++
		}
		if content.ToolResult != nil {
			count += countContentPromptCacheMarkers(content.ToolResult.Content)
		}
	}
	return count
}

func recordPromptCacheReceipt(
	ctx *RequestContext,
	action string,
	reason string,
	inserted int,
	preserved int,
	latency time.Duration,
) {
	if ctx.PromptCacheAction != "" {
		return
	}
	ctx.PromptCacheAction = action
	ctx.PromptCacheReason = reason
	ctx.PromptCacheInserted = inserted
	ctx.PromptCachePreserved = preserved
	decision := requestDecisionStateKey(ctx)
	metrics.RecordPluginExecution(
		config.DecisionPluginPromptCache,
		decision,
		action,
		latency.Seconds(),
	)
	logging.ComponentDebugEvent("extproc", "prompt_cache_policy_applied", map[string]interface{}{
		"request_id": ctx.RequestID,
		"decision":   decision,
		"action":     action,
		"reason":     reason,
		"inserted":   inserted,
		"preserved":  preserved,
	})
}

func addPromptCacheReceiptToImmediateResponse(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) {
	if response == nil ||
		ctx == nil ||
		ctx.PromptCacheAction == "" ||
		!debugHeadersRequested(ctx) {
		return
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	builder := newResponseHeaderMutationBuilder()
	addPromptCacheReceiptHeaders(builder, ctx)
	immediate.Headers.SetHeaders = append(immediate.Headers.SetHeaders, builder.setHeaders...)
}
