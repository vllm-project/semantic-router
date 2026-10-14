package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// responseStageCheck is one response-stage signal's work for an answer: run
// reads the model (nil when there is nothing to read) and publish records the
// observation on the request context (nil when there is nothing to record).
type responseStageCheck struct {
	run     func(context.Context)
	publish func()
}

// scoreResponseStageSignals scores the selected recipe's response-stage rules
// against the answer: the response-direction jailbreak rules and the
// hallucination rules. The stage opens its own bundle, so the model calls of
// both run concurrently and reach each runtime process as one /v1/bundle call.
// Only this goroutine writes the request context: every observation is
// published after the calls have returned, jailbreak first.
func (r *OpenAIRouter) scoreResponseStageSignals(ctx *RequestContext, assistantContent string) {
	runResponseStage(selectionRequestContext(ctx),
		r.responseJailbreakCheck(ctx, assistantContent),
		r.hallucinationCheck(ctx, assistantContent),
	)
}

// runResponseStage runs the checks' model calls concurrently in one bundle
// opened on parent, then publishes every check in order.
func runResponseStage(parent context.Context, checks ...responseStageCheck) {
	runs := make([]func(context.Context), 0, len(checks))
	for _, check := range checks {
		if check.run != nil {
			runs = append(runs, check.run)
		}
	}
	if len(runs) > 0 {
		stage, bundle := modelservice.WithBundle(parent, 0)
		leave := bundle.Join()
		modelservice.Fan(stage, len(runs), func(i int) { runs[i](stage) })
		leave()
	}
	for _, check := range checks {
		if check.publish != nil {
			check.publish()
		}
	}
}
