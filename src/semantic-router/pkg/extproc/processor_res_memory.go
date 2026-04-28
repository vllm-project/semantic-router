package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) scheduleResponseMemoryStore(ctx *RequestContext, responseBody []byte) {
	autoStoreEnabled := extractAutoStore(ctx)
	if !autoStoreEnabled && r.Config != nil && r.Config.Memory.AutoStore {
		logging.Infof("extractAutoStore: Falling back to global config, AutoStore=%v", r.Config.Memory.AutoStore)
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
	currentAssistantResponse := extractAssistantResponseText(responseBody)
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
			history,
		); err != nil {
			logging.Warnf("Memory store failed: %v", err)
		}
	})
}
