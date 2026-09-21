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
	"sync"
	"sync/atomic"
	"time"
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

// FusionPanelAttemptEvidence records the internal outcome and accounting data
// for one panel model.
//
// It excludes response bodies, but Error deliberately carries raw provider error
// text so failures stay diagnosable inside the router. It must therefore be
// projected -- onto FusionQuorumAttemptOutcome or the Replay diagnostics --
// before crossing any telemetry or response boundary.
type FusionPanelAttemptEvidence struct {
	Model string
	State FusionPanelAttemptState
	Error string
	Usage TokenUsage
}

// FusionQuorumEvidence records the internal panel evidence available when Fusion
// cannot meet its configured usable-response quorum. Its attempts carry provider
// error text, so it is projected before crossing a boundary.
type FusionQuorumEvidence struct {
	RequiredCount int
	UsableCount   int
	Attempts      []FusionPanelAttemptEvidence
	Usage         TokenUsage

	// SelectedPolicy, FallbackTarget, and Disposition are filled in by the
	// quorum-failure policy layer once it decides what to do with the panel.
	// They stay empty when the panel error never reaches that layer.
	SelectedPolicy string
	FallbackTarget string
	Disposition    FusionQuorumDisposition
}

// FusionQuorumError reports an unmet usable-response quorum while retaining
// per-attempt evidence for internal callers.
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

// Evidence returns a defensive copy of the internal panel evidence.
func (e *FusionQuorumError) Evidence() FusionQuorumEvidence {
	if e == nil {
		return FusionQuorumEvidence{}
	}
	evidence := e.evidence
	evidence.Attempts = append([]FusionPanelAttemptEvidence(nil), evidence.Attempts...)
	return evidence
}

// WithPolicyOutcome returns a copy of the error carrying the below-quorum
// policy decision, so Replay and metrics can explain why the request ended the
// way it did without inspecting logs.
// The totalUsage argument replaces the panel-only usage so a fallback attempt's
// tokens are accounted even when that attempt failed.
func (e *FusionQuorumError) WithPolicyOutcome(
	policy string,
	fallbackTarget string,
	disposition FusionQuorumDisposition,
	totalUsage TokenUsage,
) *FusionQuorumError {
	updated := &FusionQuorumError{cause: e.cause, evidence: e.Evidence()}
	updated.evidence.SelectedPolicy = policy
	updated.evidence.FallbackTarget = fallbackTarget
	updated.evidence.Disposition = disposition
	updated.evidence.Usage = totalUsage
	return updated
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
	// completedAt is stamped by the worker the moment its call returns. Channel
	// readiness cannot prove ordering: a worker can enqueue between the executor
	// observing the deadline and draining the buffer, so the collector needs
	// explicit evidence to tell a pre-deadline result from a late one.
	completedAt time.Time

	// dispatched records that this worker passed the dispatch gate and invoked
	// the backend client. A worker still waiting on the concurrency semaphore
	// when the round ends returns without calling anything, so the configured
	// panel size overstates the attempts made. Attempt state cannot substitute:
	// both queued and in-flight workers end up cancelled.
	//
	// It marks an attempt, not a receipt. Cancellation can land while the call
	// is in flight, so a dispatched attempt may never be observed by the
	// backend, and is still work the request initiated.
	dispatched bool
}

type fusionPanelOutcome struct {
	responses    []*ModelResponse
	failedModels []FusionFailedModel
	attempts     []FusionPanelAttemptEvidence
	usage        TokenUsage
	// dispatchedModels lists, in configured-panel order, the panel models whose
	// worker crossed the dispatch gate. Callers report calls actually made from
	// this rather than from the configured panel.
	dispatchedModels []string
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

	// Fix both boundaries before any worker can finish, so arbitration never has
	// to infer one from a context that has already fired.
	bounds := newFusionPanelBoundaries(ctx, cfg, time.Now())
	panelCtx, cancel := newFusionPanelContext(ctx, bounds)
	defer cancel()

	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	var panelWork sync.WaitGroup
	l.startFusionPanelWorkers(panelCtx, req, cfg, results, &panelWork)

	arbiter := newFusionPanelArbiter(bounds, newFusionPanelCollector(cfg, cancel))
	return awaitFusionPanel(ctx, panelCtx, cfg, arbiter, results, cancel, &panelWork)
}

// startFusionPanelWorkers dispatches one goroutine per analysis model. Each
// stamps completedAt the moment its call returns, because channel readiness
// cannot establish when the work actually finished.
func (l *FusionLooper) startFusionPanelWorkers(
	panelCtx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	results chan<- fusionPanelResult,
	panelWork *sync.WaitGroup,
) {
	sem := make(chan struct{}, cfg.MaxConcurrent)
	// Call ordinals are allocated at the dispatch gate rather than from the
	// configured panel index, so they count actual calls. A slot index would
	// disagree with the fallback's ordinal whenever the dispatched subset is not
	// a prefix of the panel, letting the fallback appear to run before a panel
	// call that preceded it, or reuse its number.
	var dispatchOrdinal atomic.Int64
	for i, model := range cfg.AnalysisModels {
		panelWork.Add(1)
		go func(index int, modelName string) {
			defer panelWork.Done()
			select {
			case sem <- struct{}{}:
			case <-panelCtx.Done():
				results <- fusionPanelResult{
					index: index, model: modelName, err: panelCtx.Err(),
					completedAt: time.Now(),
				}
				return
			}
			defer func() { <-sem }()
			// Winning the semaphore does not mean the panel is still running. Once
			// cancel fires, both this send and panelCtx.Done() are ready and select
			// picks at random, so a queued worker can arrive here after the panel
			// already stopped. Re-check before taking an ordinal: numbering a call
			// that was never initiated would overstate the work the request paid
			// for.
			if err := panelCtx.Err(); err != nil {
				results <- fusionPanelResult{
					index: index, model: modelName, err: err,
					completedAt: time.Now(),
				}
				return
			}
			// Past the semaphore with the panel still live, so this worker is about
			// to call a backend and takes the next call ordinal.
			iteration := int(dispatchOrdinal.Add(1))
			resp, err := l.callFusionModel(panelCtx, req, req.OriginalRequest, cfg, modelName, false, false, iteration, cfg.AnalysisOverrides[modelName])
			results <- fusionPanelResult{
				index: index, model: modelName, resp: resp, err: err,
				completedAt: time.Now(), dispatched: true,
			}
		}(i, model)
	}
}

// awaitFusionPanel collects panel results until the panel decides, a boundary
// stops it, or every attempt has reported.
//
// ctx is the caller's context and panelCtx the panel's derived one. Both are
// needed: the arbiter must tell the caller giving up apart from the round budget
// running out, and panelCtx alone cannot distinguish them.
func awaitFusionPanel(
	ctx context.Context,
	panelCtx context.Context,
	cfg fusionExecutionConfig,
	arbiter *fusionPanelArbiter,
	results <-chan fusionPanelResult,
	cancel context.CancelFunc,
	panelWork *sync.WaitGroup,
) (fusionPanelOutcome, error) {
	for range cfg.AnalysisModels {
		select {
		case result := <-results:
			if !arbiter.admit(ctx, result) {
				continue
			}
			arbiter.settle(results, cancel, panelWork)
			return arbiter.decided()
		case <-panelCtx.Done():
			// The boundary firing is not itself a decision about any particular
			// result, so everything still outstanding is arbitrated: first what is
			// already queued, then whatever lands while the workers are joined.
			//
			// Both drains go through admit so the completion timestamps decide
			// countability. A worker can stamp a result before the boundary and
			// then be descheduled before its channel send; treating that result as
			// late would make the outcome depend on which side of the join the send
			// happened to fall, which is the nondeterminism the timestamps exist to
			// remove. A cancelled caller still rejects everything, because
			// uncountable checks the caller's context first.
			terminal := arbiter.drainQueued(ctx, results)
			arbiter.stopAndJoin(cancel, panelWork)
			if arbiter.drainQueued(ctx, results) {
				terminal = true
			}
			if terminal {
				return arbiter.decided()
			}
			return arbiter.halted(cfg, panelCtx.Err())
		}
	}

	// Every attempt reported back, so nothing can be waiting to arrive: there is
	// no drain to choose between, only the join.
	arbiter.stopAndJoin(cancel, panelWork)
	return arbiter.exhausted(ctx, cfg)
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
			// A replayed panel entry stands in for an attempt that was made, so
			// it counts toward dispatched models. Leaving it unset would report
			// a cached panel as zero calls and drop its members from
			// ModelsUsed and Iterations.
			dispatched: true,
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
