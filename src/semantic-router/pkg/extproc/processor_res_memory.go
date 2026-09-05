package extproc

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
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
	if r.MemoryExtractor == nil {
		r.recordMemoryPersistenceOutcome(ctx, "disabled", "no_extractor", false, nil)
		return
	}
	if !autoStoreEnabled {
		r.recordMemoryPersistenceOutcome(ctx, "disabled", "auto_store_off", false, nil)
		return
	}
	if ctx.ResponseJailbreakDetected {
		reason := "response_jailbreak"
		if ctx.ResponseJailbreakType == classification.JailbreakClassificationErrorType {
			reason = "jailbreak_unverified"
		}
		r.recordMemoryPersistenceOutcome(ctx, "policy_blocked", reason, false, nil)
		return
	}

	// Snapshot request-owned state before dispatch. extractMemoryInfo deep-copies
	// history; protocol encoding and parsing remain in the worker.
	currentUserMessage := extractCurrentUserMessage(ctx)
	sessionID, userID, history, infoErr := extractMemoryInfo(ctx)
	historyCount := len(history)
	extractor := r.MemoryExtractor
	receipt := r.snapshotMemoryPersistenceReceipt(ctx)
	r.memoryPersistence.Submit(ctx.TraceContext, memory.PersistenceJob{
		Run: func(jobCtx context.Context) (memory.PersistenceOutcome, error) {
			if infoErr != nil {
				return memory.PersistenceOutcome{
					Status: "skipped",
					Reason: "memory_info_unavailable",
				}, infoErr
			}
			extractorHistory, historyErr := r.memoryHistoryForExtractor(history)
			if historyErr != nil {
				return memory.PersistenceOutcome{
					Status:   "extraction_failed",
					Reason:   "history_encode_error",
					FailOpen: true,
				}, historyErr
			}

			logging.Infof(
				"Memory store: sessionID=%s, userID=%s, userMsg=%d chars, assistantMsg=%d chars, history=%d msgs",
				sessionID,
				userID,
				len(currentUserMessage),
				len(currentAssistantResponse),
				historyCount,
			)

			storedCount, err := extractor.ProcessResponseWithHistory(
				jobCtx,
				sessionID,
				userID,
				currentUserMessage,
				currentAssistantResponse,
				extractorHistory,
			)
			if err != nil {
				return memory.PersistenceOutcome{}, err
			}
			if storedCount == 0 {
				return memory.PersistenceOutcome{Status: "skipped", Reason: "no_write"}, nil
			}
			return memory.PersistenceOutcome{}, nil
		},
		Report: receipt.record,
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
