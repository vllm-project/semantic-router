package extproc

import (
	"fmt"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The tool calls the context plugin injects each source's context under.
const (
	ragContextToolName    = "vsr_rag_context"
	memoryContextToolName = "vsr_memory_context"
)

// retrieveContextSources runs the sources of the decision's context plugin
// in order and injects what each kept, as the plugin's injection mode says.
// A decision without a context plugin reads the global memory. Only a rag
// source whose on_failure is block fails the request.
func (r *OpenAIRouter) retrieveContextSources(ctx *RequestContext, userContent string, request *llmprotocol.Request) error {
	if requestBypassesRouting(ctx) || ctx.LooperRequest {
		return nil
	}
	ctx.MemoryBackend = r.Config.Memory.Backend
	if ctx.MemoryBackend == "" {
		ctx.MemoryBackend = "milvus"
	}
	plugin := config.ContextPluginOf(requestPlugins(r.Config, ctx))
	if plugin.Source(config.ContextSourceMemory) == nil {
		recordMemoryOutcome(ctx, "disabled", memoryDisabledReason(ctx), false)
	}
	if plugin == nil || len(plugin.Sources) == 0 {
		return nil
	}
	if applies, reason := r.pluginApplies(ctx, config.DecisionPluginContext); !applies {
		// pluginApplies already counted the skip; only the replay-visible outcome is set here.
		if plugin.Source(config.ContextSourceMemory) != nil {
			ctx.MemoryStatus, ctx.MemoryReason = "skipped", string(reason)
		}
		return nil
	}
	// Memory eligibility reads the request as it arrived: a rag source's
	// documents, injected as a tool result, do not make it a tool query.
	memoryEligible := plugin.Source(config.ContextSourceMemory) != nil && ShouldSearchMemory(ctx, userContent)
	for index := range plugin.Sources {
		source := &plugin.Sources[index]
		switch source.Type {
		case config.ContextSourceRAG:
			if err := r.retrieveRAGSource(ctx, plugin, source, request); err != nil {
				return err
			}
		case config.ContextSourceMemory:
			if err := r.retrieveMemorySource(ctx, plugin, source, userContent, request, memoryEligible); err != nil {
				logging.ComponentWarnEvent("extproc", "memory_retrieval_failed", map[string]interface{}{
					"request_id": ctx.RequestID,
					"error":      err.Error(),
					"fallback":   "continue_without_memory",
				})
			}
		}
	}
	return nil
}

// memoryDisabledReason says why a request retrieves no memories: its
// decision's context plugin lists no memory source, or nothing enables it.
func memoryDisabledReason(ctx *RequestContext) string {
	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.HasPlugin(config.DecisionPluginContext) {
		return "decision_config"
	}
	return "global_config"
}

// injectContext places one source's context as the plugin's injection mode
// says: a tool call and its result after the last user message, behind the
// sources injected before it, or a block of the system prompt after theirs.
func (r *OpenAIRouter) injectContext(ctx *RequestContext, plugin *config.ContextPluginConfig, sourceType, text string) error {
	request := ctx.SemanticRequest
	if request == nil {
		return fmt.Errorf("neutral inference request is unavailable")
	}
	if plugin.InjectionMode() == config.ContextInjectionSystemPrompt {
		injectContextIntoSystemPrompt(ctx, request, sourceType, text)
		return nil
	}
	prefix, name, calls := contextcompression.RAGToolCallPrefix, ragContextToolName, &ctx.RAGToolCallIDs
	if sourceType == config.ContextSourceMemory {
		prefix, name, calls = contextcompression.MemoryToolCallPrefix, memoryContextToolName, &ctx.MemoryToolCallIDs
	}
	at, ok := contextInsertionIndex(ctx, request)
	if !ok {
		return fmt.Errorf("no user message found")
	}
	callID := fmt.Sprintf("%s%d", prefix, time.Now().UnixNano())
	if ctx.RequestID != "" {
		callID = fmt.Sprintf("%s%s_%d", prefix, ctx.RequestID, time.Now().UnixNano())
	}
	if *calls == nil {
		*calls = make(map[string]struct{})
	}
	(*calls)[callID] = struct{}{}
	exchange := []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
			ID: callID, Name: name, Arguments: "{}",
		}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID:  callID,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		}}}},
	}
	request.Messages = slices.Insert(request.Messages, at, exchange...)
	request.Generation++
	if sourceType == config.ContextSourceRAG {
		ctx.HasToolsForFactCheck = true
	}
	logging.Debugf("Injected %s context as a tool result (%d chars)", sourceType, len(text))
	return nil
}

// contextInsertionIndex is where the next source's tool exchange goes: after
// the last user message and the exchanges injected before it.
func contextInsertionIndex(ctx *RequestContext, request *llmprotocol.Request) (int, bool) {
	lastUser := -1
	for index := len(request.Messages) - 1; index >= 0; index-- {
		if request.Messages[index].Role == llmprotocol.RoleUser {
			lastUser = index
			break
		}
	}
	if lastUser < 0 {
		return 0, false
	}
	at := lastUser + 1
	for at < len(request.Messages) && injectedContextMessage(ctx, request.Messages[at]) {
		at++
	}
	return at, true
}

func injectedContextMessage(ctx *RequestContext, message llmprotocol.Message) bool {
	for _, content := range message.Content {
		id := ""
		switch {
		case content.ToolCall != nil:
			id = content.ToolCall.ID
		case content.ToolResult != nil:
			id = content.ToolResult.CallID
		}
		_, rag := ctx.RAGToolCallIDs[id]
		_, memory := ctx.MemoryToolCallIDs[id]
		if id != "" && (rag || memory) {
			return true
		}
	}
	return false
}

// injectContextIntoSystemPrompt prepends a source's context to the first
// system instruction, after the blocks the sources before it prepended, or
// starts one.
func injectContextIntoSystemPrompt(ctx *RequestContext, request *llmprotocol.Request, sourceType, text string) {
	block := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: fmt.Sprintf("Context from knowledge base:\n\n%s\n\n", text)}
	if sourceType == config.ContextSourceMemory {
		block.Text = text + "\n\n"
	}
	injected := false
	for index := range request.Instructions {
		if request.Instructions[index].Role != llmprotocol.RoleSystem {
			continue
		}
		at := min(ctx.contextPrefixBlocks, len(request.Instructions[index].Content))
		request.Instructions[index].Content = slices.Insert(request.Instructions[index].Content, at, block)
		injected = true
		break
	}
	if !injected {
		request.Instructions = append([]llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{block},
		}}, request.Instructions...)
	}
	ctx.contextPrefixBlocks++
	request.Generation++
	logging.Debugf("Injected %s context into the system prompt (%d chars)", sourceType, len(text))
}
