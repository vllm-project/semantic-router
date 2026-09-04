package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
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

	currentUserMessage := extractCurrentUserMessage(ctx)
	r.memoryPersistence.Submit(ctx.TraceContext, memory.PersistenceJob{
		Run: func(jobCtx context.Context) (memory.PersistenceOutcome, error) {
			sessionID, userID, history, err := extractMemoryInfo(ctx)
			if err != nil {
				return memory.PersistenceOutcome{
					Status: "skipped",
					Reason: "memory_info_unavailable",
				}, err
			}
			extractorHistory, err := r.memoryHistoryForExtractor(history)
			if err != nil {
				return memory.PersistenceOutcome{
					Status:   "extraction_failed",
					Reason:   "history_encode_error",
					FailOpen: true,
				}, err
			}

			logging.Infof(
				"Memory store: sessionID=%s, userID=%s, userMsg=%d chars, assistantMsg=%d chars, history=%d msgs",
				sessionID,
				userID,
				len(currentUserMessage),
				len(currentAssistantResponse),
				len(history),
			)

			storedCount, err := r.MemoryExtractor.ProcessResponseWithHistory(
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
		Report: func(status, reason string, failOpen bool, cause error) {
			r.recordMemoryPersistenceOutcome(ctx, status, reason, failOpen, cause)
		},
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

func (r *OpenAIRouter) recordMemoryPersistenceOutcome(
	ctx *RequestContext, status,
	reason string,
	failOpen bool,
	cause error,
) {
	if status != "scheduled" {
		metrics.RecordPluginExecution("memory_persistence", requestDecisionStateKey(ctx), status, 0)
	}
	r.appendMemoryPersistenceReplayOutcome(ctx, status, reason, failOpen)
	if status == "scheduled" {
		return
	}

	if !failOpen && cause == nil && status != "rejected" {
		return
	}

	fields := map[string]interface{}{
		"request_id": ctx.RequestID,
		"status":     status,
		"reason":     reason,
		"fail_open":  failOpen,
	}
	if cause != nil {
		fields["error_class"] = fmt.Sprintf("%T", cause)
	}
	logging.ComponentWarnEvent("extproc", "memory_persistence_outcome", fields)
}

func (r *OpenAIRouter) appendMemoryPersistenceReplayOutcome(
	ctx *RequestContext,
	status string,
	reason string,
	failOpen bool,
) {
	if ctx.RouterReplayID == "" {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}
	phase := "terminal"
	if status == "scheduled" {
		phase = "scheduled"
	}
	if err := recorder.AppendOutcome(ctx.RouterReplayID, routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "router",
		Target:    "router",
		TargetRef: "memory_persistence",
		Verdict:   status,
		Reason:    reason,
		Metadata: map[string]string{
			"kind":      "memory_persistence_receipt",
			"phase":     phase,
			"fail_open": fmt.Sprintf("%t", failOpen),
		},
	}); err != nil {
		logging.ComponentWarnEvent("extproc", "memory_persistence_replay_outcome_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"status":     status,
			"reason":     reason,
			"phase":      phase,
			"error":      err.Error(),
		})
	}
}
