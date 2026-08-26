package extproc

import (
	"fmt"
	"math"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
)

// consumeBackendDispatchOutcome authenticates the private handler's bounded
// outcome, binds its attempted prefix to the request-local dispatch plan, and
// returns a mutation that always removes the internal header from the public
// response.
func (r *OpenAIRouter) consumeBackendDispatchOutcome(
	request *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) (*ext_proc.HeaderMutation, error) {
	mutation := &ext_proc.HeaderMutation{RemoveHeaders: []string{
		strings.ToLower(backendinvoker.DispatchOutcomeHeader),
	}}
	token, present, duplicate := takeDispatchOutcomeHeader(responseHeaderMap(request))
	if duplicate {
		return mutation, fmt.Errorf("dispatch outcome is duplicated")
	}
	expected := dispatchOutcomeExpected(ctx)
	if !present {
		if expected {
			return mutation, fmt.Errorf("dispatch outcome is missing")
		}
		return mutation, nil
	}
	if !expected || r == nil || r.DispatchCapabilities == nil {
		return mutation, fmt.Errorf("unexpected dispatch outcome")
	}
	generation, ok := routingcontext.GenerationFrom(ctx.TraceContext)
	if !ok {
		return mutation, fmt.Errorf("durable routing generation is unavailable")
	}
	outcome, err := r.DispatchCapabilities.VerifyDispatchOutcome(
		ctx.TraceContext,
		token,
		dispatchauthority.OutcomeVerificationRequest{Generation: generation, RequestID: ctx.RequestID},
	)
	if err != nil {
		return mutation, err
	}
	if err := applyDispatchOutcome(ctx, outcome); err != nil {
		return mutation, err
	}
	return mutation, nil
}

func responseHeaderMap(request *ext_proc.ProcessingRequest_ResponseHeaders) *core.HeaderMap {
	if request == nil || request.ResponseHeaders == nil {
		return nil
	}
	return request.ResponseHeaders.Headers
}

func takeDispatchOutcomeHeader(headerMap *core.HeaderMap) (string, bool, bool) {
	if headerMap == nil {
		return "", false, false
	}
	value := ""
	count := 0
	filtered := headerMap.Headers[:0]
	for _, header := range headerMap.Headers {
		if header == nil || !strings.EqualFold(strings.TrimSpace(header.Key), backendinvoker.DispatchOutcomeHeader) {
			filtered = append(filtered, header)
			continue
		}
		count++
		if count == 1 {
			value = strings.TrimSpace(extractHeaderValue(header))
		}
		header.Value = ""
		header.RawValue = nil
	}
	headerMap.Headers = filtered
	if count == 0 {
		return "", false, false
	}
	if count != 1 || value == "" {
		return "", true, true
	}
	return value, true, false
}

func dispatchOutcomeExpected(ctx *RequestContext) bool {
	if ctx == nil || ctx.DispatchState == nil {
		return false
	}
	ctx.DispatchState.mu.Lock()
	defer ctx.DispatchState.mu.Unlock()
	return ctx.DispatchState.capabilityIssued && !ctx.DispatchState.outcomeConsumed
}

func applyDispatchOutcome(ctx *RequestContext, outcome backendinvoker.DispatchOutcome) error {
	if ctx == nil || ctx.DispatchState == nil {
		return fmt.Errorf("request dispatch state is unavailable")
	}
	var admissionID, admissionDigest string
	if ctx.InferenceAccess != nil {
		ctx.InferenceAccess.mu.Lock()
		if ctx.InferenceAccess.admission != nil {
			admissionID = ctx.InferenceAccess.admission.Tenant.AdmissionID
			admissionDigest = ctx.InferenceAccess.admission.RequestDigest
		}
		ctx.InferenceAccess.mu.Unlock()
		if admissionID == "" || outcome.AdmissionID != admissionID ||
			outcome.AdmissionDigest != admissionDigest {
			return fmt.Errorf("dispatch outcome admission mismatch")
		}
	}
	state := ctx.DispatchState
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.capabilityIssued || state.outcomeConsumed || state.requestDigest == "" ||
		outcome.RequestDigest != state.requestDigest {
		return fmt.Errorf("dispatch outcome request identity mismatch")
	}
	if state.primaryCandidateCount == 0 {
		// Nested Looper requests are authorized by an already verified grant.
		// They own no outer settlement journal, but still authenticate and strip
		// the private response proof before it can reach the caller.
		state.outcomeConsumed = true
		return nil
	}
	if err := validateDispatchOutcomeCandidates(state, outcome); err != nil {
		return err
	}
	if err := applyDispatchOutcomeAttempts(state, outcome); err != nil {
		return err
	}
	state.selectedDispatchID = outcome.SelectedDispatchID
	state.outcomeConsumed = true
	if outcome.SelectedDispatchID != "" {
		selected := state.dispatches[len(outcome.Attempted)-1]
		ctx.VSRSelectedModel = selected.model
	}
	return nil
}

func validateDispatchOutcomeCandidates(
	state *requestDispatchState, outcome backendinvoker.DispatchOutcome,
) error {
	if state.primaryCandidateCount > len(state.dispatches) ||
		len(outcome.Attempted) > state.primaryCandidateCount {
		return fmt.Errorf("dispatch outcome candidate count mismatch")
	}
	for index, attempted := range outcome.Attempted {
		dispatch := state.dispatches[index]
		if !sameDispatchOutcomeCandidate(dispatch, attempted) {
			return fmt.Errorf("dispatch outcome candidate %d mismatch", index)
		}
	}
	if outcome.SelectedDispatchID != "" {
		last := len(outcome.Attempted) - 1
		if last < 0 || outcome.Attempted[last].DispatchID != outcome.SelectedDispatchID ||
			outcome.Attempted[last].State != backendinvoker.AttemptResponseStarted {
			return fmt.Errorf("dispatch outcome selected candidate mismatch")
		}
	} else {
		for index, attempted := range outcome.Attempted {
			if attempted.State == backendinvoker.AttemptResponseStarted {
				return fmt.Errorf("dispatch outcome candidate %d started a response without selection", index)
			}
		}
	}
	return nil
}

func applyDispatchOutcomeAttempts(
	state *requestDispatchState, outcome backendinvoker.DispatchOutcome,
) error {
	completedAt := time.Now().UTC()
	if len(outcome.Attempted) == 0 {
		// The private handler failed before a physical attempt began. Preserve a
		// single local known-zero record for request settlement without turning
		// any planned fallback Model into attempted usage.
		state.noDispatchProven = true
		dispatch := state.dispatches[0]
		dispatch.settlementEligible = true
		dispatch.attempted = false
		dispatch.attemptEvidenceRequired = false
		dispatch.dispatchType = "local"
		dispatch.state = usageaccounting.EvidenceKnownZero
		dispatch.usage = usageaccounting.ActualUsage{}
		dispatch.reason = ""
		dispatch.completedAt = completedAt
	}
	for index, attempted := range outcome.Attempted {
		dispatch := state.dispatches[index]
		dispatch.attempted = true
		dispatch.settlementEligible = true
		dispatch.attemptEvidenceRequired = true
		dispatch.dispatchType = attempted.DispatchType
		switch attempted.State {
		case backendinvoker.AttemptKnownZero:
			dispatch.state = usageaccounting.EvidenceKnownZero
			dispatch.usage = usageaccounting.ActualUsage{}
			dispatch.reason = ""
			dispatch.completedAt = completedAt
		case backendinvoker.AttemptUnknown:
			dispatch.state = usageaccounting.EvidenceUnknown
			dispatch.usage = usageaccounting.ActualUsage{}
			dispatch.reason = "backend_attempt_outcome_unknown"
			dispatch.completedAt = completedAt
		case backendinvoker.AttemptResponseStarted:
			dispatch.state = usageaccounting.EvidenceUnknown
			dispatch.reason = "authoritative_usage_missing"
		default:
			return fmt.Errorf("dispatch outcome candidate %d has invalid evidence", index)
		}
	}
	return nil
}

func sameDispatchOutcomeCandidate(
	dispatch *inferenceDispatch,
	candidate backendinvoker.DispatchOutcomeCandidate,
) bool {
	if candidate.Ordinal < 0 || int64(candidate.Ordinal) > math.MaxUint32 {
		return false
	}
	// #nosec G115 -- the candidate ordinal is bounded to MaxUint32 above.
	ordinal := uint32(candidate.Ordinal)
	return dispatch != nil && dispatch.planned &&
		dispatch.dispatchType == candidate.DispatchType &&
		dispatch.id == candidate.DispatchID && dispatch.ordinal == ordinal &&
		dispatch.planDigest == candidate.DispatchPlanDigest &&
		dispatch.modelID == candidate.ModelID && dispatch.modelRevision == candidate.ModelRevision &&
		dispatch.priority == candidate.Priority
}

// ensureSettlementDispatchLocked guarantees that terminal request settlement
// contains at most the first planned Model when no signed attempt evidence is
// available. Later planned candidates remain excluded.
func ensureSettlementDispatchLocked(state *requestDispatchState, unknown bool) *inferenceDispatch {
	for _, dispatch := range state.dispatches {
		if dispatch != nil && dispatch.settlementEligible {
			return dispatch
		}
	}
	if len(state.dispatches) == 0 || state.dispatches[0] == nil {
		return nil
	}
	dispatch := state.dispatches[0]
	dispatch.settlementEligible = true
	dispatch.dispatchType = "local"
	dispatch.completedAt = time.Now().UTC()
	dispatch.usage = usageaccounting.ActualUsage{}
	if unknown && state.capabilityIssued {
		dispatch.attempted = true
		dispatch.attemptEvidenceRequired = true
		dispatch.state = usageaccounting.EvidenceUnknown
		dispatch.reason = "dispatch_outcome_missing"
	} else {
		dispatch.attempted = false
		dispatch.attemptEvidenceRequired = false
		dispatch.state = usageaccounting.EvidenceKnownZero
		dispatch.reason = ""
	}
	return dispatch
}
