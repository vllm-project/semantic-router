package extproc

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) scheduleSemanticResponseMemoryStore(
	ctx *RequestContext,
	response *llmprotocol.Response,
) {
	r.scheduleResponseMemoryStoreText(ctx, extractSemanticAssistantResponseText(response))
}

func (r *OpenAIRouter) scheduleResponseMemoryStoreText(
	ctx *RequestContext,
	currentAssistantResponse string,
) {
	autoStoreEnabled := extractAutoStore(ctx)
	if requestAutoStore, ok := extractRequestAutoStore(ctx); ok {
		autoStoreEnabled = requestAutoStore
	} else if !autoStoreEnabled && r.Config != nil && r.Config.Memory.AutoStore {
		logging.Infof("extractAutoStore: Falling back to router config, AutoStore=%v", r.Config.Memory.AutoStore)
		autoStoreEnabled = true
	}
	logging.Infof(
		"Memory store check: MemoryExtractor=%v, autoStore=%v, responseJailbreakPassed=%v",
		r.MemoryExtractor != nil,
		autoStoreEnabled,
		!ctx.ResponseJailbreakDetected,
	)
	if r.MemoryExtractor == nil || !autoStoreEnabled || ctx.ResponseJailbreakDetected {
		return
	}

	currentUserMessage := extractCurrentUserMessage(ctx)
	// goSafely wraps the goroutine in a deferred recover so a panic in
	// the memory-store path (e.g. an unexpected payload shape) is
	// logged via observability rather than aborting the router
	// process (#1843).
	goSafely("memory_store", func() {
		bgCtx := context.Background()
		sessionID, userID, history, err := extractMemoryInfo(ctx)
		if err != nil {
			logging.Errorf("Memory store failed: %v", err)
			return
		}
		extractorHistory, err := r.memoryHistoryForExtractor(history)
		if err != nil {
			logging.Warnf("Memory store failed to encode neutral history: %v", err)
			return
		}

		logging.Infof(
			"Memory store: sessionID=%s, userID=%s, userMsg=%d chars, assistantMsg=%d chars, history=%d msgs",
			sessionID,
			userID,
			len(currentUserMessage),
			len(currentAssistantResponse),
			len(history),
		)

		if err := r.MemoryExtractor.ProcessResponseWithHistory(
			bgCtx,
			sessionID,
			userID,
			currentUserMessage,
			currentAssistantResponse,
			extractorHistory,
		); err != nil {
			logging.Warnf("Memory store failed: %v", err)
		}
	})
}

func (r *OpenAIRouter) memoryHistoryForExtractor(
	history []llmprotocol.Message,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(history) == 0 {
		return nil, nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, err
	}
	encoded, err := engine.EncodeRequest(
		llmprotocol.OpenAIChatV1,
		llmprotocol.Request{Generation: 1, Model: "memory-history", Messages: history},
		llmprotocol.Envelope{},
	)
	if err != nil {
		return nil, fmt.Errorf("encode history: %w", err)
	}
	request, err := parseOpenAIRequest(encoded.Body)
	if err != nil {
		return nil, fmt.Errorf("parse encoded history: %w", err)
	}
	return request.Messages, nil
}
