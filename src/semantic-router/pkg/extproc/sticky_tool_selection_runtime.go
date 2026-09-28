package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type stickyToolDefinitionKey struct {
	name        string
	fingerprint string
}

// applyStickyToolSelectionWithStatus applies the bounded session merge and
// reports whether the resulting identity set was committed and projected
// back to the current request definitions. Callers that are handling an empty
// request turn use the status to preserve the ordinary no-op fallback when
// sticky state cannot be reused.
func (r *OpenAIRouter) applyStickyToolSelectionWithStatus(
	request *llmprotocol.Request,
	authorizedTools []llmprotocol.Tool,
	selectedTools []llmprotocol.Tool,
	selectionConfig *config.ToolSelectionPluginConfig,
	toolsConfig *config.ToolsPluginConfig,
	strategyID string,
	ctx *RequestContext,
) ([]llmprotocol.Tool, bool) {
	return r.applyStickyToolSelectionWithStatusAndRetrieval(
		request,
		authorizedTools,
		selectedTools,
		selectionConfig,
		toolsConfig,
		strategyID,
		ctx,
		"",
	)
}

func (r *OpenAIRouter) applyStickyToolSelectionWithStatusAndRetrieval(
	request *llmprotocol.Request,
	authorizedTools []llmprotocol.Tool,
	selectedTools []llmprotocol.Tool,
	selectionConfig *config.ToolSelectionPluginConfig,
	toolsConfig *config.ToolsPluginConfig,
	strategyID string,
	ctx *RequestContext,
	retrievalFingerprint string,
) ([]llmprotocol.Tool, bool) {
	if !r.shouldApplyStickyToolSelection(request, selectionConfig, ctx) {
		return selectedTools, false
	}

	policyFingerprint := tools.EffectiveToolPolicyFingerprint(selectionConfig, toolsConfig)
	identity := ResolveStickyToolIdentity(ctx, string(ctx.Routing.RecipeName()), policyFingerprint)
	if !identity.Trusted {
		emitStickyToolSelectionReceipt(sessiontools.SelectionReceipt{
			Fallback: true,
			Reason:   sessiontools.SelectionReasonUntrusted,
		})
		return selectedTools, false
	}

	authorized, definitions := stickyToolCandidates(authorizedTools)
	// Manager output is identity-only and must always be resolvable against
	// the current authorized definitions before a request can consume it. A
	// duplicate exact identity would be rejected by the manager; reject it at
	// this adapter seam as well so a projection failure can never leave a
	// committed state that the provider-bound request cannot represent.
	if len(definitions) != len(authorizedTools) {
		emitStickyToolSelectionAdapterFallback("duplicate_authorized_definition")
		return selectedTools, false
	}
	selected, _ := stickyToolCandidates(selectedTools)
	capabilities := r.stickyToolModelCapabilities(ctx)

	result := r.stickyToolSelectionManager.Select(ctx.embeddingContext(), sessiontools.SelectionInput{
		Enabled:               true,
		Trusted:               true,
		Key:                   identity.StorageKey,
		Quota:                 identity.QuotaKey,
		PolicyFingerprint:     policyFingerprint,
		CatalogFingerprint:    tools.EffectiveToolCatalogFingerprint(authorizedTools, retrievalFingerprint),
		CapabilityFingerprint: tools.ToolCapabilityFingerprint(capabilities, string(ctx.TargetFormat)),
		Authorized:            authorized,
		Selected:              selected,
		CalledToolNames:       stickyCalledToolNames(request, selectionConfig.Sticky.EffectiveMaxTools()),
		Turn:                  ctx.TurnIndex,
		MaxTools:              selectionConfig.Sticky.EffectiveMaxTools(),
		MaxNewToolsPerTurn:    selectionConfig.Sticky.EffectiveMaxNewToolsPerTurn(),
		PinCalledTools:        selectionConfig.Sticky.EffectivePinCalledTools(),
		StrategyID:            strategyID,
	})
	emitStickyToolSelectionReceipt(result.Receipt)
	if !result.Receipt.Committed {
		return selectedTools, false
	}

	resolved := make([]llmprotocol.Tool, 0, len(result.Selected))
	for _, candidate := range result.Selected {
		definition, ok := definitions[stickyToolDefinitionKey{
			name:        candidate.Name,
			fingerprint: candidate.DefinitionFingerprint,
		}]
		if !ok {
			// This should be unreachable because MergeToolSet only emits
			// identities from Authorized. Keep the request fail-open if a future
			// manager implementation violates that invariant.
			emitStickyToolSelectionAdapterFallback("definition_projection_failed")
			return selectedTools, false
		}
		resolved = append(resolved, definition)
	}
	if len(resolved) == 0 {
		// An empty request-time selection has no provider-bound mutation to
		// apply. Keep the ordinary empty-turn no-op semantics even when the
		// manager committed an empty envelope (for example, a first turn with
		// no called or relevant tools, or a state invalidated by reauthorization).
		return selectedTools, false
	}
	return resolved, true
}

func (r *OpenAIRouter) stickyToolModelCapabilities(ctx *RequestContext) []string {
	if r == nil || r.Config == nil || ctx == nil {
		return nil
	}
	modelName := strings.TrimSpace(ctx.VSRSelectedModel)
	if modelName == "" {
		return nil
	}
	return r.Config.GetModelCapabilities(modelName)
}

func (r *OpenAIRouter) shouldApplyStickyToolSelection(
	request *llmprotocol.Request,
	selectionConfig *config.ToolSelectionPluginConfig,
	ctx *RequestContext,
) bool {
	return r != nil && r.Config != nil && r.stickyToolSelectionManager != nil &&
		request != nil && request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto &&
		selectionConfig != nil && selectionConfig.Enabled &&
		selectionConfig.Sticky != nil && selectionConfig.Sticky.Enabled &&
		ctx != nil && !ctx.SkipProcessing && !ctx.LooperRequest &&
		strings.TrimSpace(ctx.VSRSelectedModel) != "" &&
		ctx.VSRSelectedDecision != nil && ctx.Routing.SelectedRecipe() != nil && !ctx.Routing.IsPassthrough()
}

func stickyToolCandidates(values []llmprotocol.Tool) ([]sessiontools.ToolCandidate, map[stickyToolDefinitionKey]llmprotocol.Tool) {
	candidates := make([]sessiontools.ToolCandidate, 0, len(values))
	definitions := make(map[stickyToolDefinitionKey]llmprotocol.Tool, len(values))
	for _, tool := range values {
		fingerprint := tools.ToolDefinitionFingerprint(tool)
		candidate := sessiontools.ToolCandidate{
			Name:                  tool.Name,
			DefinitionFingerprint: fingerprint,
		}
		candidates = append(candidates, candidate)
		definitions[stickyToolDefinitionKey{name: tool.Name, fingerprint: fingerprint}] = tool
	}
	return candidates, definitions
}

func filterStickyToolCatalog(
	values []llmprotocol.Tool,
	advanced *config.AdvancedToolFilteringConfig,
) []llmprotocol.Tool {
	if advanced == nil || !advanced.Enabled {
		return values
	}
	allow := stickyToolNameSet(advanced.AllowTools)
	block := stickyToolNameSet(advanced.BlockTools)
	filtered := make([]llmprotocol.Tool, 0, len(values))
	for _, tool := range values {
		name := strings.ToLower(strings.TrimSpace(tool.Name))
		if _, denied := block[name]; denied {
			continue
		}
		if len(allow) > 0 {
			if _, permitted := allow[name]; !permitted {
				continue
			}
		}
		filtered = append(filtered, tool)
	}
	return filtered
}

func stickyToolNameSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value != "" {
			result[value] = struct{}{}
		}
	}
	return result
}

func effectiveStickyAddPolicy(
	selectionConfig *config.ToolSelectionPluginConfig,
	topK int,
	similarityThreshold *float32,
	advanced *config.AdvancedToolFilteringConfig,
	fallbackToEmpty bool,
) *config.ToolSelectionPluginConfig {
	effective := *selectionConfig
	effective.TopK = topK
	effective.SimilarityThreshold = similarityThreshold
	effective.AdvancedFiltering = advanced
	effective.FallbackToEmpty = &fallbackToEmpty
	return &effective
}

func effectiveStickyFilterPolicy(
	selectionConfig *config.ToolSelectionPluginConfig,
	fallbackToEmpty bool,
) *config.ToolSelectionPluginConfig {
	effective := *selectionConfig
	effective.FallbackToEmpty = &fallbackToEmpty
	return &effective
}

func stickyCalledToolNames(request *llmprotocol.Request, limit int) []string {
	if request == nil || limit <= 0 {
		return nil
	}
	names := make([]string, 0, limit)
	seen := make(map[string]struct{}, limit)
	// Scan newest calls first so the bounded result retains the latest distinct
	// names, then restore their original chronological order below.
	for messageIndex := len(request.Messages) - 1; messageIndex >= 0 && len(names) < limit; messageIndex-- {
		message := request.Messages[messageIndex]
		if message.Role != llmprotocol.RoleAssistant {
			continue
		}
		for contentIndex := len(message.Content) - 1; contentIndex >= 0 && len(names) < limit; contentIndex-- {
			content := message.Content[contentIndex]
			if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil {
				continue
			}
			name := strings.TrimSpace(content.ToolCall.Name)
			if name == "" {
				continue
			}
			if _, exists := seen[name]; exists {
				continue
			}
			seen[name] = struct{}{}
			names = append(names, name)
		}
	}
	for left, right := 0, len(names)-1; left < right; left, right = left+1, right-1 {
		names[left], names[right] = names[right], names[left]
	}
	return names
}

func emitStickyToolSelectionReceipt(receipt sessiontools.SelectionReceipt) {
	logging.ComponentDebugEvent("extproc", "sticky_tool_selection", map[string]interface{}{
		"committed":   receipt.Committed,
		"fallback":    receipt.Fallback,
		"invalidated": receipt.Invalidated,
		"reason":      receipt.Reason,
		"reused":      receipt.Merge.Reused,
		"added":       receipt.Merge.Added,
		"replaced":    receipt.Merge.Replaced,
		"evicted":     receipt.Merge.Evicted,
		"pinned":      receipt.Merge.Pinned,
		"cas_retries": receipt.CASRetries,
	})
}

func emitStickyToolSelectionAdapterFallback(reason string) {
	logging.ComponentDebugEvent("extproc", "sticky_tool_selection", map[string]interface{}{
		"committed":   false,
		"fallback":    true,
		"invalidated": false,
		"reason":      reason,
	})
}
