package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/codexreview"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// observeApprovalReview recognises a Codex approval review at ingress, while
// the wire body still has the client metadata the codec drops. A request is a
// review when it names another entrypoint's review_model or, for reviews that
// use the session model, when it carries Codex's review envelope.
func (r *OpenAIRouter) observeApprovalReview(ctx *RequestContext, body []byte, request *llmprotocol.Request) {
	source := ""
	switch {
	case r.Config.IsReviewEntrypoint(config.ChatAPI, request.Model):
		source = store.ReviewIdentifiedByEntrypoint
	case codexreview.HasEnvelope(request):
		source = store.ReviewIdentifiedByContent
	default:
		return
	}
	ctx.ApprovalReview = codexreview.Observe(request, body, source)
}

// replayApprovalReview is the review a new record starts with. The planned
// action repeats the request's tool arguments, so its command, directory and
// arguments are kept as Replay keeps the request's tools: under raw capture
// with request text captured, and never on a masking route.
func replayApprovalReview(ctx *RequestContext) (string, *routerreplay.Review) {
	if ctx == nil || ctx.ApprovalReview == nil {
		return "", nil
	}
	review := store.CloneReview(ctx.ApprovalReview)
	policy := ctx.RouterReplayPluginConfig
	keepsAction := policy.CapturesRawContent() && policy.CapturesRequestText() && !replaySuppressesBody(ctx)
	if review.Action != nil && !keepsAction {
		review.Action = &routerreplay.ReviewAction{Tool: review.Action.Tool}
	}
	return store.RequestKindCodexApprovalReview, review
}

// applyReplayReviewSession files a review under the session it judged unless
// a header named the request's session. A session id derived from the
// request would describe the reviewer's prompt, not the agent's session.
func applyReplayReviewSession(record *routerreplay.RoutingRecord, headerSession bool) {
	if !headerSession && record.Review != nil && record.Review.ReviewedSessionID != "" {
		record.SessionID = record.Review.ReviewedSessionID
	}
}

// observeApprovalReviewVerdict reads the reviewer's answer from the decoded
// response, before any response-stage rewrite or body truncation.
func observeApprovalReviewVerdict(ctx *RequestContext, response *llmprotocol.Response) {
	if ctx == nil || ctx.ApprovalReview == nil || response == nil {
		return
	}
	ctx.ApprovalReview.Status, ctx.ApprovalReview.Assessment = codexreview.Verdict(response)
}

// observeStreamedApprovalReviewFailure marks a review whose stream carried an
// upstream error event, which otherwise ends like a client going away.
func observeStreamedApprovalReviewFailure(ctx *RequestContext) {
	if ctx != nil && ctx.ApprovalReview != nil && ctx.SemanticStreamState != nil && ctx.SemanticStreamState.failed != nil {
		ctx.ApprovalReview.Status, ctx.ApprovalReview.Assessment = store.ReviewStatusFailed, nil
	}
}

// settleReplayApprovalReview writes the review once, when its record ends.
func settleReplayApprovalReview(ctx *RequestContext, recorder *routerreplay.Recorder, state string) {
	if ctx.ApprovalReview == nil || ctx.approvalReviewSettled {
		return
	}
	ctx.approvalReviewSettled = true
	_, review := replayApprovalReview(ctx)
	review.Status = settledReviewStatus(review.Status, state)
	if review.Status != store.ReviewStatusCompleted {
		review.Assessment = nil
	} else if review.Assessment != nil {
		// The rationale is response text: kept only where Replay keeps it.
		// Redacted whole before it is cut, so a cut never leaves part of a
		// credential that redaction would no longer recognize.
		review.Assessment.Rationale, review.Assessment.RationaleTruncated = codexreview.BoundRationale(
			string(redactCallerCredential(ctx, []byte(review.Assessment.Rationale))))
		if !replayResponseContentAllowed(ctx) || !ctx.RouterReplayPluginConfig.CapturesResponseText() {
			review.Assessment.Rationale, review.Assessment.RationaleTruncated = "", false
		}
	}
	if err := recorder.SettleReview(ctx.RouterReplayID, *review); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_review_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

// settledReviewStatus is a review's final status. A failed call is a failed
// review whatever its body said, and a call that ended without an answer the
// Router read is cancelled when the client left and invalid otherwise.
func settledReviewStatus(observed, lifecycle string) string {
	switch {
	case lifecycle == routerreplay.LifecycleFailed:
		return store.ReviewStatusFailed
	case observed != store.ReviewStatusPending:
		return observed
	case lifecycle == routerreplay.LifecycleAborted:
		return store.ReviewStatusCancelled
	default:
		return store.ReviewStatusInvalid
	}
}
