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
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FusionPanelAttemptState classifies one panel model's terminal outcome.
type FusionPanelAttemptState string

const (
	// FusionPanelAttemptUsable means the response supplied non-empty assistant content or reasoning.
	FusionPanelAttemptUsable FusionPanelAttemptState = "usable"
	// FusionPanelAttemptUnusable means a successful call supplied no usable assistant text.
	FusionPanelAttemptUnusable FusionPanelAttemptState = "unusable"
	// FusionPanelAttemptFailed means the call failed before producing a parsed response.
	FusionPanelAttemptFailed FusionPanelAttemptState = "failed"
	// FusionPanelAttemptTimedOut means the attempt ended at the panel deadline.
	FusionPanelAttemptTimedOut FusionPanelAttemptState = "timed_out"
	// FusionPanelAttemptCancelled means the parent context cancelled the attempt.
	FusionPanelAttemptCancelled FusionPanelAttemptState = "cancelled"
)

// FusionPanelAttemptEvidence records content-free outcome and accounting data
// for one panel model. It intentionally excludes response bodies.
type FusionPanelAttemptEvidence struct {
	Model string
	State FusionPanelAttemptState
	Error string
	Usage TokenUsage
}

// FusionQuorumEvidence records the content-free panel evidence available when
// Fusion cannot meet its configured usable-response quorum.
type FusionQuorumEvidence struct {
	RequiredCount int
	UsableCount   int
	Attempts      []FusionPanelAttemptEvidence
	Usage         TokenUsage
}

// FusionQuorumError reports an unmet usable-response quorum while retaining
// content-free per-attempt evidence for internal callers.
type FusionQuorumError struct {
	cause    error
	evidence FusionQuorumEvidence
}

func (e *FusionQuorumError) Error() string {
	if e == nil {
		return "fusion panel quorum not met"
	}
	message := fmt.Sprintf(
		"fusion panel quorum not met: got %d usable response%s, require %d",
		e.evidence.UsableCount,
		pluralSuffix(e.evidence.UsableCount),
		e.evidence.RequiredCount,
	)
	if e.cause != nil {
		return message + ": " + e.cause.Error()
	}
	return message
}

func (e *FusionQuorumError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Evidence returns a defensive copy of the content-free panel evidence.
func (e *FusionQuorumError) Evidence() FusionQuorumEvidence {
	if e == nil {
		return FusionQuorumEvidence{}
	}
	evidence := e.evidence
	evidence.Attempts = append([]FusionPanelAttemptEvidence(nil), evidence.Attempts...)
	return evidence
}

// FusionQuorumEvidenceFromError extracts Fusion panel accounting from a typed
// quorum error, including when another error wraps it.
func FusionQuorumEvidenceFromError(err error) (FusionQuorumEvidence, bool) {
	var quorumErr *FusionQuorumError
	if !errors.As(err, &quorumErr) {
		return FusionQuorumEvidence{}, false
	}
	return quorumErr.Evidence(), true
}

type fusionPanelResult struct {
	index int
	model string
	resp  *ModelResponse
	err   error
}

type fusionPanelOutcome struct {
	responses    []*ModelResponse
	failedModels []FusionFailedModel
	attempts     []FusionPanelAttemptEvidence
	usage        TokenUsage
}

func (l *FusionLooper) executeFusionPanel(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
) (fusionPanelOutcome, error) {
	// Paired multi-arm evaluation supplies the panel verbatim so every arm
	// synthesizes from a byte-identical panel (see bench/grounded_fusion). The
	// cached values still pass through the same usability and quorum checks, but
	// need not contain the live client's Parsed representation.
	if req.CachedPanel != nil {
		return collectCachedFusionPanel(req.CachedPanel, cfg)
	}

	panelCtx := ctx
	cancel := func() {}
	if cfg.RoundTimeoutSeconds > 0 {
		panelCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
	}
	defer cancel()

	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	sem := make(chan struct{}, cfg.MaxConcurrent)
	for i, model := range cfg.AnalysisModels {
		go func(index int, modelName string) {
			select {
			case sem <- struct{}{}:
			case <-panelCtx.Done():
				results <- fusionPanelResult{index: index, model: modelName, err: panelCtx.Err()}
				return
			}
			defer func() { <-sem }()
			resp, err := l.callFusionModel(panelCtx, req, req.OriginalRequest, cfg, modelName, false, false, index+1, cfg.AnalysisOverrides[modelName])
			results <- fusionPanelResult{index: index, model: modelName, resp: resp, err: err}
		}(i, model)
	}

	collector := newFusionPanelCollector(cfg, cancel)
	for range cfg.AnalysisModels {
		select {
		case result := <-results:
			done, err := collector.handleResult(result)
			if done {
				return collector.outcome(), err
			}
		case <-panelCtx.Done():
			collector.handleContextDone(panelCtx.Err())
			outcome := collector.outcome()
			return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, panelCtx.Err())
		}
	}

	outcome := collector.outcome()
	if len(outcome.responses) < cfg.MinSuccessfulResponses {
		return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, nil)
	}
	return outcome, nil
}

func collectCachedFusionPanel(
	panel []*ModelResponse,
	cfg fusionExecutionConfig,
) (fusionPanelOutcome, error) {
	collector := newFusionPanelCollectorForModels(cachedFusionPanelModels(panel, cfg), cfg, func() {})
	for index, response := range panel {
		model := collector.attempts[index].Model
		_, err := collector.handleResult(fusionPanelResult{
			index: index,
			model: model,
			resp:  response,
		})
		if err != nil {
			return collector.outcome(), err
		}
	}
	outcome := collector.outcome()
	if len(outcome.responses) < cfg.MinSuccessfulResponses {
		return outcome, newFusionQuorumError(cfg.MinSuccessfulResponses, outcome, nil)
	}
	return outcome, nil
}

func cachedFusionPanelModels(panel []*ModelResponse, cfg fusionExecutionConfig) []string {
	models := make([]string, len(panel))
	for index, response := range panel {
		if response != nil && strings.TrimSpace(response.Model) != "" {
			models[index] = response.Model
			continue
		}
		if index < len(cfg.AnalysisModels) {
			models[index] = cfg.AnalysisModels[index]
			continue
		}
		models[index] = fmt.Sprintf("cached-panel-%d", index+1)
	}
	return models
}

type fusionPanelCollector struct {
	cfg       fusionExecutionConfig
	cancel    context.CancelFunc
	ordered   []*ModelResponse
	attempts  []FusionPanelAttemptEvidence
	completed []bool
	usable    int
	usage     TokenUsage
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
		cfg:       cfg,
		cancel:    cancel,
		ordered:   make([]*ModelResponse, len(models)),
		attempts:  attempts,
		completed: make([]bool, len(models)),
	}
}

func (c *fusionPanelCollector) handleResult(result fusionPanelResult) (bool, error) {
	if result.index < 0 || result.index >= len(c.attempts) || c.completed[result.index] {
		return false, nil
	}

	evidence := &c.attempts[result.index]
	if strings.TrimSpace(result.model) != "" {
		evidence.Model = result.model
	}
	c.completed[result.index] = true
	if result.resp != nil {
		evidence.Usage = result.resp.Usage
		c.usage = c.usage.Add(result.resp)
	}

	if result.err != nil {
		evidence.State = fusionPanelErrorState(result.err)
		evidence.Error = result.err.Error()
		if c.cfg.OnError == config.FusionOnErrorFail {
			c.cancel()
			return true, fmt.Errorf("fusion panel model %q failed: %w", evidence.Model, result.err)
		}
		return false, nil
	}
	if !isUsableFusionPanelResponse(result.resp) {
		evidence.State = FusionPanelAttemptUnusable
		evidence.Error = "response contained no usable assistant content or reasoning"
		if c.cfg.OnError == config.FusionOnErrorFail {
			c.cancel()
			return true, fmt.Errorf("fusion panel model %q returned no usable assistant content or reasoning", evidence.Model)
		}
		return false, nil
	}

	evidence.State = FusionPanelAttemptUsable
	c.ordered[result.index] = result.resp
	c.usable++
	if c.usable < c.cfg.MinSuccessfulResponses {
		return false, nil
	}
	c.logQuorum()
	c.cancel()
	return true, nil
}

func (c *fusionPanelCollector) handleContextDone(err error) {
	state := FusionPanelAttemptCancelled
	if errors.Is(err, context.DeadlineExceeded) {
		state = FusionPanelAttemptTimedOut
	}
	for index := range c.attempts {
		if c.completed[index] {
			continue
		}
		c.completed[index] = true
		c.attempts[index].State = state
		c.attempts[index].Error = err.Error()
	}
}

func (c *fusionPanelCollector) outcome() fusionPanelOutcome {
	attempts := append([]FusionPanelAttemptEvidence(nil), c.attempts...)
	return fusionPanelOutcome{
		responses:    compactUsableFusionPanelResponses(c.ordered),
		failedModels: failedFusionPanelModels(attempts),
		attempts:     attempts,
		usage:        c.usage,
	}
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

func isUsableFusionPanelResponse(response *ModelResponse) bool {
	if response == nil {
		return false
	}
	return strings.TrimSpace(response.Content) != "" || strings.TrimSpace(response.ReasoningContent) != ""
}

func compactUsableFusionPanelResponses(ordered []*ModelResponse) []*ModelResponse {
	responses := make([]*ModelResponse, 0, len(ordered))
	for _, response := range ordered {
		if !isUsableFusionPanelResponse(response) {
			continue
		}
		responses = append(responses, response)
	}
	return responses
}

func failedFusionPanelModels(attempts []FusionPanelAttemptEvidence) []FusionFailedModel {
	failed := make([]FusionFailedModel, 0, len(attempts))
	for _, attempt := range attempts {
		if attempt.State == "" || attempt.State == FusionPanelAttemptUsable {
			continue
		}
		failed = append(failed, FusionFailedModel{Model: attempt.Model, Error: attempt.Error})
	}
	return failed
}

func fusionPanelErrorState(err error) FusionPanelAttemptState {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return FusionPanelAttemptTimedOut
	case errors.Is(err, context.Canceled):
		return FusionPanelAttemptCancelled
	default:
		return FusionPanelAttemptFailed
	}
}

func newFusionQuorumError(
	required int,
	outcome fusionPanelOutcome,
	cause error,
) error {
	return &FusionQuorumError{
		cause: cause,
		evidence: FusionQuorumEvidence{
			RequiredCount: required,
			UsableCount:   len(outcome.responses),
			Attempts:      append([]FusionPanelAttemptEvidence(nil), outcome.attempts...),
			Usage:         outcome.usage,
		},
	}
}

func pluralSuffix(count int) string {
	if count == 1 {
		return ""
	}
	return "s"
}
