package modelservice

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

const decisionMediaBodyLimit = 48 << 20

func (c *Client) mediaBodyLimit() int {
	limit := int(c.requestBytes.Load())
	if limit <= 0 || limit > decisionMediaBodyLimit {
		return decisionMediaBodyLimit
	}
	return limit
}

func hasDecisionMedia(body *api.DecisionRequest) bool {
	if body == nil {
		return false
	}
	if body.Images != nil || body.Videos != nil {
		return true
	}
	if body.States != nil {
		for _, state := range *body.States {
			if state.Images != nil || state.Videos != nil {
				return true
			}
		}
	}
	return false
}

func bundleHasMedia(tasks []api.BundleTask) bool {
	for _, task := range tasks {
		if hasDecisionMedia(task.Decisions) {
			return true
		}
	}
	return false
}

func (c *Client) checkMediaBody(body any, media bool) error {
	if !media {
		return nil
	}
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("%w: cannot encode decision media request", ErrRejected)
	}
	if len(data) > c.mediaBodyLimit() {
		return fmt.Errorf("%w: decision media body exceeds runtime byte limit", ErrRejected)
	}
	return nil
}

// A bundle's envelopes also consume bytes. Split complete tasks rather than
// stripping media after their cache identities have been computed.
func (c *Client) splitMediaBundle(ctx context.Context, tasks []api.BundleTask) ([]api.BundleResult, exchangeTiming, error) {
	mid := len(tasks) / 2
	left, a, err := c.sendBundle(ctx, tasks[:mid])
	if err != nil {
		return nil, a, err
	}
	right, b, err := c.sendBundle(ctx, tasks[mid:])
	if err != nil {
		return nil, b, err
	}
	a.elapsed += b.elapsed
	a.total += b.total
	a.reported = a.reported && b.reported
	for i := range a.phases {
		a.phases[i] += b.phases[i]
	}
	return append(left, right...), a, nil
}

// Preserve per-state media when deciding whether another call can join a task.
// A task that cannot fit stays separate; sendBundle also bounds its envelope.
func mediaFusionFits(client *Client, task *bundleTask, call *bundleCall, state string) bool {
	if !hasDecisionMedia(call.task.Decisions) && !fusedTaskHasMedia(task) {
		return true
	}
	candidate := &bundleTask{decision: &fusedDecision{}}
	for _, existing := range task.decision.states {
		for _, member := range existing.calls {
			candidate.decision.add(member, existing.text)
			candidate.calls = append(candidate.calls, member)
		}
	}
	candidate.decision.add(call, state)
	candidate.calls = append(candidate.calls, call)
	candidate.prepareDecision()
	return client.checkMediaBody(api.BundleRequest{Tasks: []api.BundleTask{candidate.task}}, true) == nil
}

func fusedTaskHasMedia(task *bundleTask) bool {
	for _, state := range task.decision.states {
		if len(state.request.Images)+len(state.request.Videos) > 0 {
			return true
		}
	}
	return false
}
