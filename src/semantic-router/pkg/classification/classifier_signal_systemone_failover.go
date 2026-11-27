package classification

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// sendSystemOneFailover tries the deployments in order under one caller
// budget. Each deployment reads its own view of the same source state.
// Attempt failures are not signal failures until the chain gives up.
func (c *Classifier) sendSystemOneFailover(ctx context.Context, call *systemOneCall, request modelservice.Request) (modelservice.Response, error) {
	ctx, cancel := decisionFailoverContext(ctx, call.rules)
	defer cancel()
	var response modelservice.Response
	var err error
	for i, deployment := range call.fallbackChain {
		if stopped := ctx.Err(); stopped != nil {
			return modelservice.Response{}, stopped
		}
		response, err = c.sendFailoverAttempt(ctx, deployment, call, request, i > 0)
		if !modelservice.CanFailOverDecision(err) {
			return response, err
		}
	}
	return response, err
}

// A rule timeout bounds the whole chain; the enclosing stage or request may
// end it sooner. No rule timeout leaves the caller's deadline in effect.
func decisionFailoverContext(ctx context.Context, rules []askedRule) (context.Context, context.CancelFunc) {
	var timeout time.Duration
	for _, asked := range rules {
		if t := time.Duration(asked.rule.TimeoutMs) * time.Millisecond; t > 0 && (timeout == 0 || t < timeout) {
			timeout = t
		}
	}
	if timeout == 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func (c *Classifier) sendFailoverAttempt(ctx context.Context, deployment string, call *systemOneCall, request modelservice.Request, backup bool) (modelservice.Response, error) {
	resource := c.Config.ModelDeployments[deployment]
	if backup {
		// The stage expects this participant to ask the primary only. Mark it
		// as waiting while sending to a different deployment, so that neither
		// the backup nor other participants hold up the stage's primary calls.
		done := modelservice.Await(ctx)
		defer done()
		ctx = modelservice.Unbundled(ctx)
	}
	if !call.rules[0].rule.FullCoverage() {
		limit, headShare := c.routingViewBudget(deployment)
		request = boundedState(request, limit, headShare)
	}
	request.MaxTokens = resource.ScanBudget()
	if err := c.checkFailoverMedia(deployment, request); err != nil {
		return modelservice.Response{}, err
	}
	timeout := resource.CallTimeout()
	if timeout == 0 {
		timeout = config.DefaultSystemOneTimeoutMs * time.Millisecond
	}
	attempt, cancel := context.WithTimeout(modelservice.WithOwnDecisionDeadline(ctx), timeout)
	defer cancel()
	response, err := c.askDecisionDeployment(attempt, deployment, request)
	if err == nil && attempt.Err() != nil {
		return modelservice.Response{}, attempt.Err()
	}
	return response, err
}

// An attached card can arrive after preparation. Do not silently drop media
// when an observed deployment cannot read it.
func (c *Classifier) checkFailoverMedia(deployment string, request modelservice.Request) error {
	if len(request.Images)+len(request.Videos) == 0 {
		return nil
	}
	card, ok := c.deploymentCard(deployment)
	if !ok {
		return nil
	}
	modalities := card.ReadModalities()
	if len(request.Images) != 0 && !slices.Contains(modalities, config.SystemOneModalityImage) {
		return fmt.Errorf("%w: deployment %q does not read image input", modelservice.ErrRejected, deployment)
	}
	if len(request.Videos) != 0 && !slices.Contains(modalities, config.SystemOneModalityVideo) {
		return fmt.Errorf("%w: deployment %q does not read video input", modelservice.ErrRejected, deployment)
	}
	return nil
}
