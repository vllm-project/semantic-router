package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func attemptEvidenceRequest(
	admission accessruntime.Admission,
	dispatches []*inferenceDispatch,
) accessruntime.AttemptEvidenceRequest {
	request := accessruntime.AttemptEvidenceRequest{
		Admission:  admission,
		Dispatches: make([]accessruntime.AttemptEvidenceDispatch, 0, len(dispatches)),
	}
	for _, dispatch := range dispatches {
		if dispatch == nil {
			continue
		}
		request.Dispatches = append(request.Dispatches, accessruntime.AttemptEvidenceDispatch{
			DispatchID: dispatch.id, Ordinal: dispatch.ordinal,
			DispatchPlanDigest: dispatch.planDigest, ModelID: dispatch.modelID,
			ModelRevision: dispatch.modelRevision,
		})
	}
	return request
}

func reconcileAttemptEvidence(
	dispatches []*inferenceDispatch,
	snapshot accessruntime.AttemptEvidenceSnapshot,
) error {
	observations := snapshot.Observations()
	if len(observations) != len(dispatches) {
		return fmt.Errorf("attempt evidence does not cover the complete dispatch journal")
	}
	for index, observation := range observations {
		dispatch := dispatches[index]
		if dispatch == nil || observation.DispatchID != dispatch.id {
			return fmt.Errorf("attempt evidence dispatch order differs at ordinal %d", index)
		}
		reconcileDispatchAttemptEvidence(dispatch, observation)
	}
	return nil
}

func reconcileDispatchAttemptEvidence(
	dispatch *inferenceDispatch,
	observation accessruntime.AttemptEvidenceObservation,
) {
	dispatch.attempts = nil
	if !observation.Present {
		if dispatch.attemptEvidenceRequired {
			markDispatchUsageUnknown(dispatch, "attempt_evidence_missing")
		}
		return
	}
	evidence := observation.Evidence
	if evidence.DispatchID != dispatch.id || evidence.Ordinal != dispatch.ordinal ||
		evidence.DispatchPlanDigest != dispatch.planDigest || evidence.ModelID != dispatch.modelID ||
		evidence.ModelRevision != dispatch.modelRevision {
		markDispatchUsageUnknown(dispatch, "attempt_evidence_mismatch")
		return
	}
	dispatch.dispatchType = canonicalUsageReason(evidence.DispatchType)
	if dispatch.startedAt.IsZero() || (!evidence.StartedAt.IsZero() && evidence.StartedAt.Before(dispatch.startedAt)) {
		dispatch.startedAt = evidence.StartedAt
	}
	if len(evidence.Attempts) == 0 {
		markDispatchUsageUnknown(dispatch, "attempt_not_started")
		return
	}
	for index := 0; index < len(evidence.Attempts)-1; index++ {
		if evidence.Attempts[index].State != quotaruntime.AttemptEvidenceKnownZero {
			markDispatchUsageUnknown(dispatch, "attempt_sequence_invalid")
			return
		}
	}

	terminal := evidence.Attempts[len(evidence.Attempts)-1]
	terminalState := usageledger.UsageUnknown
	var terminalReason string
	switch terminal.State {
	case quotaruntime.AttemptEvidenceKnownZero:
		if dispatch.state == usageaccounting.EvidenceKnownActual {
			terminalReason = "attempt_usage_mismatch"
		} else {
			dispatch.state = usageaccounting.EvidenceKnownZero
			dispatch.usage = usageaccounting.ActualUsage{}
			dispatch.reason = ""
			terminalState = usageledger.UsageKnownZero
			terminalReason = ""
		}
	case quotaruntime.AttemptEvidenceResponseStarted:
		if dispatch.state == usageaccounting.EvidenceKnownActual {
			terminalState = usageledger.UsageKnownActual
			terminalReason = ""
		} else {
			terminalReason = canonicalUsageReason(dispatch.reason)
			if terminalReason == "request_terminated" || terminalReason == "dispatch_not_terminal" {
				terminalReason = "authoritative_usage_missing"
			}
		}
	case quotaruntime.AttemptEvidenceUnknown:
		terminalReason = canonicalUsageReason(terminal.ErrorCode)
	default:
		terminalReason = "attempt_evidence_invalid"
	}
	if terminalState == usageledger.UsageUnknown {
		markDispatchUsageUnknown(dispatch, terminalReason)
	}

	completedAt := dispatch.completedAt.UTC()
	for _, attempt := range evidence.Attempts {
		if attempt.CompletedAt.After(completedAt) {
			completedAt = attempt.CompletedAt.UTC()
		}
	}
	if completedAt.IsZero() || completedAt.Before(dispatch.startedAt) {
		completedAt = time.Now().UTC()
	}
	dispatch.completedAt = completedAt
	dispatch.attempts = make([]usageledger.Attempt, 0, len(evidence.Attempts))
	for index, attempt := range evidence.Attempts {
		state := usageledger.UsageUnknown
		errorCode := canonicalUsageReason(attempt.ErrorCode)
		switch attempt.State {
		case quotaruntime.AttemptEvidenceKnownZero:
			if index == len(evidence.Attempts)-1 {
				state = terminalState
				errorCode = terminalReason
			} else {
				state = usageledger.UsageKnownZero
			}
		case quotaruntime.AttemptEvidenceResponseStarted:
			if index == len(evidence.Attempts)-1 {
				state = terminalState
				errorCode = terminalReason
			}
		case quotaruntime.AttemptEvidenceUnknown:
		default:
			errorCode = "attempt_evidence_invalid"
		}
		attemptCompletedAt := attempt.CompletedAt.UTC()
		if attemptCompletedAt.IsZero() || attemptCompletedAt.Before(attempt.StartedAt) {
			attemptCompletedAt = completedAt
		}
		if state != usageledger.UsageUnknown {
			errorCode = ""
		}
		dispatch.attempts = append(dispatch.attempts, usageledger.Attempt{
			AttemptID: attempt.AttemptID, Ordinal: index,
			BackendID:  canonicalOptionalUUID(attempt.BackendID),
			ProviderID: canonicalUsageReason(attempt.ProviderID),
			State:      state, StatusCode: attempt.StatusCode, ErrorCode: errorCode,
			StartedAt: attempt.StartedAt.UTC(), CompletedAt: attemptCompletedAt,
		})
	}
}

func markDispatchUsageUnknown(dispatch *inferenceDispatch, reason string) {
	dispatch.state = usageaccounting.EvidenceUnknown
	dispatch.usage = usageaccounting.ActualUsage{}
	dispatch.reason = canonicalUsageReason(reason)
	if dispatch.completedAt.IsZero() {
		dispatch.completedAt = time.Now().UTC()
	}
}
