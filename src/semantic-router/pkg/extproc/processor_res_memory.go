package extproc

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
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

	// Reject missing identity before inspecting or copying conversation content.
	if ctx.SemanticRequest == nil || len(ctx.SemanticRequest.Messages) == 0 || extractUserID(ctx) == "" {
		r.recordMemoryPersistenceOutcome(ctx, "skipped", "memory_info_unavailable", false, nil)
		return
	}
	receipt := r.snapshotMemoryPersistenceReceipt(ctx)
	if !receipt.reserve() {
		receipt.record("rejected", "receipt_queue_full", true, nil)
		return
	}
	reservation := r.memoryPersistence.TryReserve(ctx.TraceContext, receipt.record)
	if reservation == nil {
		return
	}
	defer reservation.Abort(memory.PersistenceOutcome{Status: "extraction_failed", Reason: "snapshot_failed", FailOpen: true}, nil)
	var retained []*responseapi.StoredResponse
	if ctx.ResponseObjectState != nil {
		retained = ctx.ResponseObjectState.ConversationHistory
	}
	if err := validateMemoryHistoryBudget(reservation.Context(), ctx.SemanticRequest.Messages, retained); err != nil {
		reservation.Abort(memory.PersistenceOutcome{Status: "skipped", Reason: "history_too_large", FailOpen: true}, err)
		return
	}
	// Snapshot while the request still owns its mutable state. Preparation is
	// admitted and bounded; protocol encoding remains in the worker.
	currentUserMessage := extractCurrentUserMessage(ctx)
	sessionID, userID, history, infoErr := extractMemoryInfo(ctx)
	if infoErr != nil {
		reservation.Abort(memory.PersistenceOutcome{Status: "skipped", Reason: "memory_info_unavailable"}, infoErr)
		return
	}
	historyCount := len(history)
	extractor := r.MemoryExtractor
	reservation.Start(func(jobCtx context.Context) (memory.PersistenceOutcome, error) {
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

		storedCount, err := extractor.ProcessResponseWithHistoryCount(
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
