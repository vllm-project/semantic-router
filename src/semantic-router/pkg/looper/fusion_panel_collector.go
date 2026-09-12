/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// unusableFusionPanelReason explains an attempt that completed without content
// the judge can deliberate over. It is recorded as evidence and, under
// on_error: fail, becomes the panel's abort reason.
const unusableFusionPanelReason = "response contained no usable assistant content or reasoning"

// fusionPanelCollector accumulates panel results into quorum state and bounded
// per-attempt evidence.
//
// Two paths reach it: handleResult, for results that can still satisfy the
// quorum, and drainUncountedResults, for work that arrived after the panel was
// already decided. Both record through recordAttempt so evidence, usage, and
// dispatch tracking cannot drift apart.
type fusionPanelCollector struct {
	cfg        fusionExecutionConfig
	cancel     context.CancelFunc
	ordered    []*ModelResponse
	attempts   []FusionPanelAttemptEvidence
	dispatched []bool
	usable     int
	usage      TokenUsage
}

func newFusionPanelCollector(cfg fusionExecutionConfig, cancel context.CancelFunc) *fusionPanelCollector {
	return newFusionPanelCollectorForModels(cfg.AnalysisModels, cfg, cancel)
}

func newFusionPanelCollectorForModels(
	models []string,
	cfg fusionExecutionConfig,
	cancel context.CancelFunc,
) *fusionPanelCollector {
	attempts := make([]FusionPanelAttemptEvidence, len(models))
	for index, model := range models {
		attempts[index].Model = model
	}
	return &fusionPanelCollector{
		cfg:        cfg,
		cancel:     cancel,
		ordered:    make([]*ModelResponse, len(models)),
		attempts:   attempts,
		dispatched: make([]bool, len(models)),
	}
}

// recordAttempt writes terminal evidence, usage, and dispatch tracking for one
// result and classifies it. Every result goes through here, so the counted and
// uncounted paths cannot classify the same result differently.
//
// One result per configured slot is a producer invariant: each worker closes
// over its own index and sends on exactly one of its three exit paths. The
// boundary path joins every worker and drains again before reporting, so no
// attempt can still be outstanding by then and none has to be repaired.
func (c *fusionPanelCollector) recordAttempt(
	result fusionPanelResult,
) *FusionPanelAttemptEvidence {
	c.noteDispatch(result)

	evidence := &c.attempts[result.index]
	if strings.TrimSpace(result.model) != "" {
		evidence.Model = result.model
	}
	if result.resp != nil {
		evidence.Usage = result.resp.Usage
		c.usage = c.usage.Add(result.resp)
	}

	switch {
	case result.err != nil:
		evidence.State = fusionPanelErrorState(result.err)
		evidence.Error = result.err.Error()
	case !isUsableFusionPanelResponse(result.resp):
		evidence.State = FusionPanelAttemptUnusable
		evidence.Error = unusableFusionPanelReason
	default:
		evidence.State = FusionPanelAttemptUsable
	}
	return evidence
}

// handleResult records a result that can still count toward the quorum. It
// reports whether the panel is decided, and the error that decided it.
func (c *fusionPanelCollector) handleResult(result fusionPanelResult) (bool, error) {
	evidence := c.recordAttempt(result)
	if evidence.State != FusionPanelAttemptUsable {
		// on_error: fail aborts the panel on the first bad attempt rather than
		// waiting to see whether the remaining models reach quorum.
		if c.cfg.OnError == config.FusionOnErrorFail {
			c.cancel()
			return true, fusionPanelAttemptError(evidence.Model, result.err)
		}
		return false, nil
	}

	c.ordered[result.index] = result.resp
	c.usable++
	if c.usable < c.cfg.MinSuccessfulResponses {
		return false, nil
	}
	c.logQuorum()
	c.cancel()
	return true, nil
}

// fusionPanelAttemptError describes why on_error: fail aborted the panel. A nil
// err means the attempt completed but carried nothing usable.
func fusionPanelAttemptError(model string, err error) error {
	if err != nil {
		return fmt.Errorf("fusion panel model %q failed: %w", model, err)
	}
	return fmt.Errorf("fusion panel model %q returned no usable assistant content or reasoning", model)
}

// drainUncountedResults records terminal evidence and token usage for results
// that arrived after the panel had already reached its own decision. They are
// accounted so no attempt is dropped from evidence or usage, but none of them
// can change a decision that was already made.
//
// Boundary-stopped panels do not come here: their results are arbitrated against
// the completion timestamps instead, because the boundary firing is not a
// decision about any particular result.
func (c *fusionPanelCollector) drainUncountedResults(results <-chan fusionPanelResult) {
	for {
		select {
		case result := <-results:
			// Recorded truthfully, but a late usable result cannot
			// retroactively satisfy a quorum that was already decided.
			c.recordAttempt(result)
		default:
			return
		}
	}
}

func (c *fusionPanelCollector) outcome() fusionPanelOutcome {
	attempts := append([]FusionPanelAttemptEvidence(nil), c.attempts...)
	return fusionPanelOutcome{
		responses:        compactUsableFusionPanelResponses(c.ordered),
		failedModels:     failedFusionPanelModels(attempts),
		attempts:         attempts,
		usage:            c.usage,
		dispatchedModels: c.dispatchedPanelModels(),
	}
}

// noteDispatch marks a panel slot as having attempted a backend call. It is
// keyed off the worker's own flag rather than the attempt state, because a
// cancelled attempt may be either queued or already in flight.
func (c *fusionPanelCollector) noteDispatch(result fusionPanelResult) {
	if !result.dispatched {
		return
	}
	c.dispatched[result.index] = true
}

// dispatchedPanelModels returns the dispatched models in configured-panel order,
// so response accounting stays deterministic regardless of completion order.
func (c *fusionPanelCollector) dispatchedPanelModels() []string {
	models := make([]string, 0, len(c.dispatched))
	for index, dispatched := range c.dispatched {
		if dispatched {
			models = append(models, c.attempts[index].Model)
		}
	}
	return models
}

func (c *fusionPanelCollector) logQuorum() {
	if c.usable >= len(c.attempts) {
		return
	}
	logging.ComponentEvent("looper", "fusion_panel_quorum_reached", map[string]interface{}{
		"responses": c.usable,
		"panel":     len(c.attempts),
	})
}
