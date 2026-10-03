package extproc

import (
	"context"
	"fmt"
	"runtime/debug"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func (r *OpenAIRouter) scheduleSemanticResponseMemoryStore(
	ctx *RequestContext,
	response *llmprotocol.Response,
) {
	if r == nil || ctx == nil {
		return
	}
	// Snapshot preparation runs on the response path, so an unexpected payload
	// shape must be logged rather than failing a request whose answer is already
	// generated (#1843). The reservation's deferred Abort still publishes its
	// receipt first, because deferred calls unwind in reverse order.
	defer func() {
		if recovered := recover(); recovered != nil {
			logging.ComponentErrorEvent("extproc", "memory_snapshot_panic", map[string]interface{}{
				"request_id": ctx.RequestID,
				"panic":      recovered,
				"stack":      string(debug.Stack()),
			})
		}
	}()
	status, reason, suppressed := r.suppressedResponseMemoryStore(ctx)
	logging.Infof(
		"Memory store check: MemoryExtractor=%v, suppressed=%v, reason=%s",
		r.MemoryExtractor != nil, suppressed, reason,
	)
	if suppressed {
		r.recordMemoryPersistenceOutcome(ctx, status, reason, false, nil)
		return
	}

	// Reject missing identity before inspecting or copying conversation content.
	if ctx.SemanticRequest == nil || len(ctx.SemanticRequest.Messages) == 0 || extractUserID(ctx) == "" {
		r.recordMemoryPersistenceOutcome(ctx, "skipped", "memory_info_unavailable", true, nil)
		return
	}
	var retained []*responseapi.StoredResponse
	if ctx.ResponseObjectState != nil && !ctx.ResponseObjectState.ProviderContextApplied {
		retained = ctx.ResponseObjectState.ConversationHistory
	}
	// Inspect bounded structure and lengths before admission. Rejected snapshots
	// must not occupy capacity while workers are blocked. No text is copied here.
	if err := validateMemorySnapshotBudget(ctx.SemanticRequest.Messages, retained, response); err != nil {
		r.recordMemoryPersistenceOutcome(ctx, "skipped", "history_too_large", true, err)
		return
	}
	receipt := r.snapshotMemoryPersistenceReceipt(ctx)
	if !receipt.reserve() {
		receipt.record("rejected", "receipt_queue_full", true, nil)
		return
	}
	if r.memoryPersistence == nil {
		receipt.record("disabled", "no_runner", false, nil)
		return
	}
	reservation := r.memoryPersistence.TryReserve(ctx.TraceContext, receipt.record)
	if reservation == nil {
		return
	}
	defer reservation.Abort(memory.PersistenceOutcome{Status: "extraction_failed", Reason: "snapshot_failed", FailOpen: true}, nil)
	if reservation.Context().Err() != nil {
		return
	}
	// Snapshot while the request still owns its mutable state. Preparation is
	// admitted and bounded; protocol encoding remains in the worker.
	currentAssistantResponse := extractSemanticAssistantResponseText(response)
	currentUserMessage := extractCurrentUserMessage(ctx)
	sessionID, userID, history, infoErr := extractMemoryInfo(ctx)
	if infoErr != nil {
		reservation.Abort(memory.PersistenceOutcome{Status: "skipped", Reason: "memory_info_unavailable", FailOpen: true}, infoErr)
		return
	}
	job := memoryPersistenceJob{
		extractor:         r.MemoryExtractor,
		codecs:            r.ProtocolCodecs,
		sessionID:         sessionID,
		userID:            userID,
		userMessage:       currentUserMessage,
		assistantResponse: currentAssistantResponse,
		history:           history,
	}
	if r.memoryConsolidation != nil {
		job.consolidate = r.memoryConsolidation
	}
	reservation.Start(job.run)
}

// memoryPersistenceJob owns everything the worker needs, so a queued write
// never keeps a router generation alive through a captured receiver.
type memoryPersistenceJob struct {
	extractor         *memory.MemoryExtractor
	codecs            *protocolcodec.Registry
	sessionID         string
	userID            string
	userMessage       string
	assistantResponse string
	history           []llmprotocol.Message
	consolidate       memoryConsolidator
}

// memoryConsolidator merges one user's memories off the response path.
// A nil implementation is not called.
type memoryConsolidator interface {
	Enqueue(userID string)
}

func (job memoryPersistenceJob) run(jobCtx context.Context) (memory.PersistenceOutcome, error) {
	extractorHistory, historyErr := memoryHistoryForExtractor(job.codecs, job.history)
	if historyErr != nil {
		return memory.PersistenceOutcome{
			Status:   "extraction_failed",
			Reason:   "history_encode_error",
			FailOpen: true,
		}, historyErr
	}

	logging.Infof(
		"Memory store: sessionID=%s, userID=%s, userMsg=%d chars, assistantMsg=%d chars, history=%d msgs",
		job.sessionID,
		job.userID,
		len(job.userMessage),
		len(job.assistantResponse),
		len(job.history),
	)

	storedCount, err := job.extractor.ProcessResponseWithHistoryCount(
		jobCtx,
		job.sessionID,
		job.userID,
		job.userMessage,
		job.assistantResponse,
		extractorHistory,
	)
	if err != nil {
		return memory.PersistenceOutcome{}, err
	}
	if storedCount == 0 {
		return memory.PersistenceOutcome{Status: "skipped", Reason: "no_write"}, nil
	}
	enqueueMemoryConsolidation(job.consolidate, job.userID, storedCount, nil)
	return memory.PersistenceOutcome{}, nil
}

func enqueueMemoryConsolidation(consolidator memoryConsolidator, userID string, storedCount int, err error) {
	if consolidator == nil || err != nil || storedCount <= 0 || userID == "" {
		return
	}
	consolidator.Enqueue(userID)
}

// recordUnscheduledResponseMemoryStore reports the terminal receipt for a
// response that never reaches scheduling. Enablement and response-stage policy
// outrank the caller's reason, so a suppressed write reads the same either way.
func (r *OpenAIRouter) recordUnscheduledResponseMemoryStore(
	ctx *RequestContext,
	status, reason string,
	failOpen bool,
) {
	if r == nil || ctx == nil {
		return
	}
	if suppressedStatus, suppressedReason, suppressed := r.suppressedResponseMemoryStore(ctx); suppressed {
		status, reason, failOpen = suppressedStatus, suppressedReason, false
	}
	r.recordMemoryPersistenceOutcome(ctx, status, reason, failOpen, nil)
}

// suppressedResponseMemoryStore is the single enablement and response-stage
// policy ladder: every path that owes a receipt reports the same verdict.
func (r *OpenAIRouter) suppressedResponseMemoryStore(ctx *RequestContext) (string, string, bool) {
	switch {
	case r.MemoryExtractor == nil:
		return "disabled", "no_extractor", true
	case !r.responseMemoryAutoStoreEnabled(ctx):
		return "disabled", "auto_store_off", true
	case ctx.ResponseJailbreakDetected:
		if ctx.ResponseJailbreakType == classification.JailbreakClassificationErrorType {
			return "policy_blocked", "jailbreak_unverified", true
		}
		return "policy_blocked", "response_jailbreak", true
	}
	return "", "", false
}

// responseMemoryAutoStoreEnabled applies server policy before client choices.
// A client may opt out but cannot undo an explicit decision-level prohibition.
func (r *OpenAIRouter) responseMemoryAutoStoreEnabled(ctx *RequestContext) bool {
	if r == nil || r.Config == nil || ctx == nil || retentionDropsResponseContent(ctx) {
		return false
	}
	memoryConfig, enabled := r.resolveMemoryPluginConfig(ctx)
	if !enabled || (memoryConfig != nil && memoryConfig.AutoStore != nil && !*memoryConfig.AutoStore) {
		return false
	}
	if state := ctx.ResponseObjectState; state != nil && state.AutoStore != nil {
		return *state.AutoStore
	}
	if requestAutoStore, ok := extractRequestAutoStore(ctx); ok {
		return requestAutoStore
	}
	if memoryConfig != nil && memoryConfig.AutoStore != nil {
		return *memoryConfig.AutoStore
	}
	return r.Config.Memory.AutoStore
}

func memoryHistoryForExtractor(
	codecs *protocolcodec.Registry,
	history []llmprotocol.Message,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(history) == 0 {
		return nil, nil
	}
	engine, err := protocolEngineFor(codecs)
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
