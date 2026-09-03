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

// applyQuorumFailurePolicy decides what a Fusion panel does when it ends below
// the configured usable-response quorum. It only interprets *FusionQuorumError:
// any other error is a genuine execution failure and is returned untouched, so
// this policy never converts an unrelated failure into a served response.
func (l *FusionLooper) applyQuorumFailurePolicy(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panel fusionPanelOutcome,
	panelErr error,
) (*Response, error) {
	var quorumErr *FusionQuorumError
	if !errors.As(panelErr, &quorumErr) {
		return nil, panelErr
	}
	evidence := quorumErr.Evidence()
	policy := string(resolvedQuorumFailurePolicy(cfg))

	// A cancelled or timed-out parent context is not a quality decision. Serving
	// a fallback would spend more budget after the caller already gave up.
	if ctxErr := fusionCallerBudgetErr(ctx); ctxErr != nil {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumCancelled, "")
		return nil, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumCancelled, evidence.Usage)
	}

	if cfg.QuorumFailurePolicy != config.FusionQuorumFailurePolicyFallback {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage)
	}

	target := cfg.QuorumFallbackTarget

	if target == "" {
		// Validation rejects this at config load; treat it as a hard failure
		// rather than silently degrading to the conservative default.
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, fmt.Errorf("fusion quorum fallback requested without a target: %w",
			quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage))
	}

	fallbackResp, err := l.callFusionFallback(ctx, req, cfg, target, len(panel.dispatchedModels))
	if err != nil {
		// The fallback may have produced billable tokens before failing, so the
		// evidence must carry panel plus fallback usage even on this path.
		evidence.Usage = evidence.Usage.Add(fallbackResp)
		disposition := fusionQuorumFallbackDisposition(ctx, err)
		l.logQuorumDecision(req, cfg, evidence, disposition, target)
		// The panel failure is the root cause; the fallback failure explains why
		// recovery did not happen. Keep both so callers can classify either.
		return nil, fmt.Errorf("fusion quorum fallback to %q failed: %w: %w",
			target, err, quorumErr.WithPolicyOutcome(policy, target, disposition, evidence.Usage))
	}

	return l.serveFusionFallbackResponse(
		req, cfg, panel, evidence, quorumErr, policy, target, fallbackResp)
}

// serveFusionFallbackResponse formats the recovered answer and attaches the
// bounded operator outcome. It is split from the policy decision so neither the
// policy branches nor the response construction hides inside the other.
func (l *FusionLooper) serveFusionFallbackResponse(
	req *Request,
	cfg fusionExecutionConfig,
	panel fusionPanelOutcome,
	evidence FusionQuorumEvidence,
	quorumErr *FusionQuorumError,
	policy string,
	target string,
	fallbackResp *ModelResponse,
) (*Response, error) {
	var err error
	// Accounting must include the panel work already paid for plus the fallback
	// call, each exactly once. panel.usage already covers every panel attempt,
	// including the unusable ones.
	usage := panel.usage.Add(fallbackResp)
	evidence.Usage = usage

	// ModelsUsed and Iterations describe the calls this request initiated, so
	// they are built from the panel models that passed the dispatch gate. The
	// configured panel overstates both when max_concurrent is smaller than the
	// panel and a round boundary fires: workers still waiting on the semaphore
	// return without calling anything.
	modelsUsed := append(append([]string{}, panel.dispatchedModels...), target)
	iterations := len(panel.dispatchedModels) + 1

	// The fallback bypassed the judge, so the panel did not contribute to this
	// answer: no Fusion trace is attached and IntermediateResponses stays nil.
	// The quorum evidence travels on the internal outcome instead.
	var response *Response
	if req.IsStreaming {
		response, err = l.formatFusionStreamingResponse(fallbackResp, modelsUsed, iterations, cfg, nil, usage)
	} else {
		response, err = l.formatFusionJSONResponse(fallbackResp, modelsUsed, iterations, cfg, nil, usage)
	}
	if err != nil {
		// Formatting failed after the fallback was paid for, so the outcome is not
		// a served fallback and the evidence must still carry the spend.
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFallbackResponseFailed, target)
		return nil, fmt.Errorf("fusion quorum fallback to %q could not be formatted: %w: %w",
			target, err, quorumErr.WithPolicyOutcome(
				policy, target, FusionQuorumFallbackResponseFailed, usage))
	}

	// The fallback call and its formatting both succeeded, but nothing has been
	// encoded for the wire yet: protocol translation still runs at the ExtProc
	// boundary and can fail. This layer therefore reports readiness, and ExtProc
	// promotes it to a terminal disposition once encoding has succeeded or
	// failed, so a single metric always agrees with the status returned.
	response.QuorumOutcome = newFusionQuorumOutcome(
		evidence, policy, target, FusionQuorumFallbackReady)
	return response, nil
}

// callFusionFallback issues the single recovery call.
//
// The fallback answers the client directly, so it is a final-serving stage
// rather than a panel member: tools stay enabled as they are for judge
// synthesis, and a tool-only reply counts as usable. It still routes through
// callFusionModel to inherit sampling bounds, access-key resolution, and the
// stage context-window gate. The response is returned alongside any error so
// the caller can account for tokens the fallback already spent.
func (l *FusionLooper) callFusionFallback(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	target string,
	dispatchedPanelCalls int,
) (*ModelResponse, error) {
	// The fallback is the final-serving stage, so it must honour the same
	// recipe-owned output contract as judge synthesis: the contract is merged
	// into the prompt and the response is post-processed the same way.
	fallbackReq := req.OriginalRequest
	if outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract); outputContract != "" {
		fallbackReq = appendFusionStageMessage(req.OriginalRequest, outputContract)
	}

	// The fallback runs after the panel calls that were actually dispatched, so
	// its ordinal is allocated from that count rather than the configured panel
	// length. Per-call telemetry then agrees with the aggregate Iterations
	// reported on the response.
	iteration := dispatchedPanelCalls + 1
	// No per-model override: analysis overrides are filtered to panel members,
	// and a target that is a panel member is rejected, so a lookup here could
	// never hit. A named fallback override would need its own config surface.
	resp, err := l.callFusionModel(
		ctx, req, fallbackReq, cfg, target,
		true, false, iteration, config.FusionModelOverride{},
	)
	if err != nil {
		return resp, err
	}
	if !isUsableFusionFallbackResponse(resp) {
		return resp, fmt.Errorf("fallback model %q returned no usable assistant content", target)
	}
	// Run the same post-processing as judge synthesis. The panel is not passed as
	// a candidate source: those responses were below quorum and are exactly what
	// the fallback exists to avoid serving, so extraction must come from the
	// fallback's own reply.
	applyJSONActionOutputContract(req.OutputContractSpec, resp, nil)
	applyFinalOutputContract(req.OutputContractSpec, resp)
	return resp, nil
}

// isUsableFusionFallbackResponse decides whether a fallback reply is an answer.
//
// It diverges from the panel rule in both directions, because the two serve
// different purposes. A tool-only reply gives the judge nothing to deliberate
// over but is a complete answer from a final-serving stage, so it counts here.
// Reasoning alone is the opposite: useful panel evidence, but not an answer.
// Every outbound codec emits assistant content and tool calls only, so
// accepting a reasoning-only reply would return 200 with an empty answer while
// telemetry recorded a served fallback. Reasoning is also private by contract
// and must not be promoted into assistant content to fill the gap.
func isUsableFusionFallbackResponse(response *ModelResponse) bool {
	if response == nil {
		return false
	}
	return strings.TrimSpace(response.Content) != "" || response.HasToolCalls
}

// logQuorumDecision records the algorithm layer's view of the panel.
//
// It deliberately emits no metric: a below-quorum panel is not terminal until
// ExtProc has encoded a response or failed to, so every quorum metric comes from
// finalizeLooperQuorumOutcome. A second producer here would either double count
// or make exactly-once a convention maintained only by tests.
func (l *FusionLooper) logQuorumDecision(
	req *Request,
	cfg fusionExecutionConfig,
	evidence FusionQuorumEvidence,
	disposition FusionQuorumDisposition,
	target string,
) {
	policy := string(resolvedQuorumFailurePolicy(cfg))
	fields := map[string]interface{}{
		"decision":        req.DecisionName,
		"required_count":  evidence.RequiredCount,
		"usable_count":    evidence.UsableCount,
		"panel":           len(cfg.AnalysisModels),
		"selected_policy": policy,
		"disposition":     string(disposition),
	}
	if target != "" {
		fields["fallback_target"] = target
	}
	logging.ComponentWarnEvent("looper", "fusion_panel_quorum_failed", fields)
}

func resolvedQuorumFailurePolicy(cfg fusionExecutionConfig) config.FusionQuorumFailurePolicy {
	if cfg.QuorumFailurePolicy == "" {
		return config.FusionQuorumFailurePolicyFail
	}
	return cfg.QuorumFailurePolicy
}

// fusionQuorumFallbackDisposition preserves stage-window exhaustion and uses
// the caller context, rather than timeout-shaped transport errors, to identify
// cancellation. The connector applies its own attempt timeout on a derived
// context, so a hung target fails with a deadline error while the caller is
// still waiting.
func fusionQuorumFallbackDisposition(ctx context.Context, err error) FusionQuorumDisposition {
	var windowErr *StageContextWindowError
	switch {
	case errors.As(err, &windowErr):
		return FusionQuorumBudgetExhausted
	case fusionCallerBudgetErr(ctx) != nil:
		return FusionQuorumCancelled
	default:
		return FusionQuorumFallbackFailed
	}
}

// fusionCallerBudgetErr reports whether the caller's budget is spent.
//
// It reads the elapsed deadline as well as Err because a context publishes Done
// from a timer callback: between the deadline instant and that callback running,
// Err still returns nil. Guarding on Err alone would let a fallback dispatch in
// that window and spend budget on a request whose deadline has already passed.
// Only the caller's context is consulted, so an internal round timeout still
// leaves a live caller eligible for the fallback.
func fusionCallerBudgetErr(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if deadline, ok := ctx.Deadline(); ok && !time.Now().Before(deadline) {
		return context.DeadlineExceeded
	}
	return nil
}
