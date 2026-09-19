package looper

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// workflowStateRestoreTimeout bounds claim commit/release so a canceled or
// deadline-exceeded resume cannot hang, while still remaining independent of
// the request lifecycle after the durable claim is taken.
const workflowStateRestoreTimeout = 5 * time.Second

func workflowStateRestoreContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), workflowStateRestoreTimeout)
}

func (l *WorkflowsLooper) releaseWorkflowToolState(claim *workflowStateClaim, restore *bool) error {
	if restore == nil || !*restore || l == nil || l.toolStates == nil || claim == nil {
		return nil
	}
	ctx, cancel := workflowStateRestoreContext()
	defer cancel()
	if err := l.toolStates.Release(ctx, claim.Recipe, claim.ID, claim.Token); err != nil {
		logging.ComponentErrorEvent("looper", "workflow_tool_state_release_failed", map[string]interface{}{
			"state_id": claim.ID,
			"error":    err.Error(),
		})
		return fmt.Errorf("release workflow tool state %q: %w", claim.ID, err)
	}
	return nil
}

func (l *WorkflowsLooper) watchWorkflowStateClaim(ctx context.Context, claim *workflowStateClaim) (context.Context, context.CancelFunc) {
	holdCtx, cancelHold := context.WithCancel(ctx)
	if l == nil || l.toolStates == nil || claim == nil {
		return holdCtx, cancelHold
	}
	renewCtx, cancelRenew := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(workflowStateClaimRenewInterval())
		defer ticker.Stop()
		for {
			select {
			case <-renewCtx.Done():
				return
			case <-holdCtx.Done():
				return
			case <-ticker.C:
				renewTimeout, stop := context.WithTimeout(renewCtx, workflowStateRestoreTimeout)
				err := l.toolStates.Renew(renewTimeout, claim.Recipe, claim.ID, claim.Token)
				stop()
				if err != nil {
					cancelHold()
					return
				}
			}
		}
	}()
	return holdCtx, func() {
		cancelRenew()
		wg.Wait()
		cancelHold()
	}
}

func (l *WorkflowsLooper) commitWorkflowToolState(claim *workflowStateClaim) error {
	if l == nil || l.toolStates == nil || claim == nil {
		return nil
	}
	ctx, cancel := workflowStateRestoreContext()
	defer cancel()
	if err := l.toolStates.Commit(ctx, claim.Recipe, claim.ID, claim.Token); err != nil {
		logging.ComponentErrorEvent("looper", "workflow_tool_state_commit_failed", map[string]interface{}{
			"state_id": claim.ID,
			"error":    err.Error(),
		})
		return fmt.Errorf("commit workflow tool state %q: %w", claim.ID, err)
	}
	return nil
}
