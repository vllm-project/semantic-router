package looper

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// workflowStateRestoreTimeout bounds the post-Take restore Put so a canceled or
// deadline-exceeded resume cannot hang, while still remaining independent of the
// request lifecycle that may already be done after Redis GETDEL.
const workflowStateRestoreTimeout = 5 * time.Second

func workflowStateRestoreContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), workflowStateRestoreTimeout)
}

func (l *WorkflowsLooper) restoreWorkflowToolState(state *workflowPendingToolState, restore *bool) error {
	if restore == nil || !*restore || l == nil || l.toolStates == nil || state == nil {
		return nil
	}
	ctx, cancel := workflowStateRestoreContext()
	defer cancel()
	if _, err := l.toolStates.Put(ctx, state); err != nil {
		logging.ComponentErrorEvent("looper", "workflow_tool_state_restore_failed", map[string]interface{}{
			"state_id": state.ID,
			"error":    err.Error(),
		})
		return fmt.Errorf("restore workflow tool state %q: %w", state.ID, err)
	}
	return nil
}
