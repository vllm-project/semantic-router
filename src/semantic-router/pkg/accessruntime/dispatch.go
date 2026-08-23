package accessruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

const maximumSettlementDispatches = 4096

// JournalDispatch pins one physical backend dispatch to its admitted request
// before inference starts. Replays with the same identity are idempotent;
// identity reuse with different facts fails closed in the quota runtime.
func (r *Runtime) JournalDispatch(ctx context.Context, request DispatchJournalRequest) (quotaruntime.MutationResult, error) {
	state, err := r.validateAdmitted(request.Admission)
	if err != nil {
		return quotaruntime.MutationResult{}, err
	}
	result, err := r.engine.JournalDispatch(ctx, quotaruntime.DispatchJournalRequest{
		Partition:       state.tenant.QuotaPartition,
		AdmissionID:     state.tenant.AdmissionID,
		AdmissionDigest: state.requestDigest,
		DispatchID:      request.DispatchID,
		Ordinal:         request.Ordinal,
		Digest:          request.Digest,
	})
	if err != nil {
		return quotaruntime.MutationResult{}, fmt.Errorf("journal inference dispatch: %w", err)
	}
	return result, nil
}

// PrepareDispatch validates an unmodified allowed Admission and returns only
// the immutable non-secret authority facts needed to issue a request-bound
// dispatch capability. The returned value has no serialization surface and
// cannot be constructed with a different admission identity.
func (r *Runtime) PrepareDispatch(admission Admission, facts DispatchFacts) (PreparedDispatch, error) {
	state, err := r.validateAdmitted(admission)
	if err != nil {
		return PreparedDispatch{}, err
	}
	if strings.TrimSpace(facts.DispatchID) == "" ||
		strings.TrimSpace(facts.DispatchID) != facts.DispatchID ||
		strings.ContainsRune(facts.DispatchID, '\x00') || len(facts.DispatchID) > 256 {
		return PreparedDispatch{}, fmt.Errorf("a bounded dispatch ID is required")
	}
	if !validRuntimeDigest(facts.DispatchPlanDigest) {
		return PreparedDispatch{}, fmt.Errorf("dispatch plan digest must be 32-byte lowercase hex")
	}
	if !validRuntimeDigest(state.requestDigest) {
		return PreparedDispatch{}, fmt.Errorf("admission digest must be 32-byte lowercase hex")
	}
	prepared := &preparedDispatchState{
		owner:              r.identity,
		namespaceID:        state.tenant.NamespaceID,
		quotaPartition:     state.tenant.QuotaPartition,
		publicationID:      state.tenant.PublicationID,
		runtimeEpoch:       state.tenant.RuntimeEpoch,
		routingRevision:    state.tenant.RoutingRevision,
		routingDigest:      state.tenant.RoutingDigest,
		admissionID:        state.tenant.AdmissionID,
		admissionDigest:    state.requestDigest,
		dispatchID:         facts.DispatchID,
		ordinal:            facts.Ordinal,
		dispatchPlanDigest: facts.DispatchPlanDigest,
	}
	return PreparedDispatch{state: prepared}, nil
}

func validRuntimeDigest(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

// ReadAttemptEvidence observes every journaled physical dispatch through the
// same quota partition as its opaque Admission. All observations must report
// one stable attempt-journal revision; otherwise the caller rebuilds the
// settlement from a fresh snapshot.
func (r *Runtime) ReadAttemptEvidence(
	ctx context.Context,
	request AttemptEvidenceRequest,
) (AttemptEvidenceSnapshot, error) {
	state, err := r.validateAdmitted(request.Admission)
	if err != nil {
		return AttemptEvidenceSnapshot{}, err
	}
	if len(request.Dispatches) == 0 || len(request.Dispatches) > maximumSettlementDispatches {
		return AttemptEvidenceSnapshot{}, fmt.Errorf("attempt evidence requires a bounded dispatch journal")
	}
	observations := make([]AttemptEvidenceObservation, 0, len(request.Dispatches))
	seen := make(map[string]struct{}, len(request.Dispatches))
	var revision uint64
	for index, dispatch := range request.Dispatches {
		if strings.TrimSpace(dispatch.DispatchID) == "" ||
			strings.TrimSpace(dispatch.DispatchID) != dispatch.DispatchID ||
			strings.ContainsRune(dispatch.DispatchID, '\x00') || len(dispatch.DispatchID) > 256 {
			return AttemptEvidenceSnapshot{}, fmt.Errorf("dispatch %d has an invalid ID", index)
		}
		if _, exists := seen[dispatch.DispatchID]; exists {
			return AttemptEvidenceSnapshot{}, fmt.Errorf("dispatch %q is duplicated", dispatch.DispatchID)
		}
		seen[dispatch.DispatchID] = struct{}{}
		if dispatch.Ordinal != uint32(index) || !validRuntimeDigest(dispatch.DispatchPlanDigest) ||
			strings.TrimSpace(dispatch.ModelID) == "" || strings.TrimSpace(dispatch.ModelID) != dispatch.ModelID ||
			strings.ContainsRune(dispatch.ModelID, '\x00') || len(dispatch.ModelID) > 256 ||
			dispatch.ModelRevision <= 0 {
			return AttemptEvidenceSnapshot{}, fmt.Errorf("dispatch %q has invalid immutable evidence facts", dispatch.DispatchID)
		}
		result, readErr := r.engine.ReadAttemptEvidence(ctx, quotaruntime.ReadAttemptEvidenceRequest{
			AttemptEvidenceReference: quotaruntime.AttemptEvidenceReference{
				Partition: state.tenant.QuotaPartition, AdmissionID: state.tenant.AdmissionID,
				AdmissionDigest: state.requestDigest, DispatchID: dispatch.DispatchID,
				Ordinal: dispatch.Ordinal, DispatchPlanDigest: dispatch.DispatchPlanDigest,
				ModelID: dispatch.ModelID, ModelRevision: dispatch.ModelRevision,
			},
		})
		if readErr != nil {
			return AttemptEvidenceSnapshot{}, fmt.Errorf("read dispatch %q attempt evidence: %w", dispatch.DispatchID, readErr)
		}
		if index == 0 {
			revision = result.Revision
		} else if result.Revision != revision {
			return AttemptEvidenceSnapshot{}, fmt.Errorf(
				"%w: dispatch attempt evidence changed while it was read",
				quotaruntime.ErrEvidenceChanged,
			)
		}
		observations = append(observations, AttemptEvidenceObservation{
			DispatchID: dispatch.DispatchID, Present: result.Present, Evidence: result.Evidence,
		})
	}
	return AttemptEvidenceSnapshot{
		Dispatches: observations,
		state: &attemptEvidenceSnapshotState{
			owner: r.identity, admissionID: state.tenant.AdmissionID,
			admissionDigest: state.requestDigest, revision: revision,
			dispatchCount: uint32(len(observations)),
		},
	}, nil
}
