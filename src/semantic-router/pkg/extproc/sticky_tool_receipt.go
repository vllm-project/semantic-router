package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func logStickyToolReceipt(ctx *RequestContext, receipt routerreplay.StickyToolSelectionReceipt) {
	decision := ""
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	logging.Infof("[tool_selection] Decision %q sticky outcome=%s reason=%s selected=%d reused=%d added=%d pinned=%d removed=%d",
		decision, receipt.Outcome, receipt.Reason, receipt.Selected, receipt.Reused, receipt.Added, receipt.Pinned, receipt.Removed)
}

// appendPendingStickyToolOutcomes attaches sticky receipts recorded before
// the Replay record existed. Each is appended once.
func appendPendingStickyToolOutcomes(ctx *RequestContext, recorder *routerreplay.Recorder) {
	if ctx == nil || ctx.RouterReplayID == "" || len(ctx.pendingStickyToolOutcomes) == 0 {
		return
	}
	for _, outcome := range ctx.pendingStickyToolOutcomes {
		appendStickyToolReplayOutcome(recorder, ctx.RouterReplayID, outcome)
	}
	ctx.pendingStickyToolOutcomes = nil
}

func appendStickyToolReplayOutcome(recorder *routerreplay.Recorder, replayID string, outcome routerreplay.Outcome) {
	if recorder == nil {
		return
	}
	if err := recorder.AppendOutcome(replayID, outcome); err != nil {
		logging.Warnf("[tool_selection] sticky receipt replay append failed: %v", err)
	}
}
