package accessruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Settle atomically applies authoritative response-actual quantities, releases
// concurrency, opens any required unknown-usage fence, and appends the durable
// terminal event. A repeated identical settlement is idempotent.
func (r *Runtime) Settle(ctx context.Context, request SettlementRequest) (quotaruntime.FinalizationResult, error) {
	state, settleErr := r.validateAdmitted(request.Admission)
	if settleErr != nil {
		return quotaruntime.FinalizationResult{}, settleErr
	}
	snapshot, settleErr := r.validateAttemptEvidenceSnapshot(request.AttemptEvidence, state)
	if settleErr != nil {
		return quotaruntime.FinalizationResult{}, settleErr
	}
	evidence := make(map[quota.CounterIdentity]quotaruntime.ActualEvidence)
	unknown := false
	for _, binding := range state.rules {
		if binding.Rule.Accounting != quota.AccountingResponseActual {
			continue
		}
		identity, err := binding.Counter()
		if err != nil {
			return quotaruntime.FinalizationResult{}, fmt.Errorf("settlement counter: %w", err)
		}
		metric := request.Aggregate.Metric(binding.Rule.Metric)
		if metric.Complete {
			evidence[identity] = quotaruntime.ActualEvidence{
				State:  quotaruntime.ActualEvidenceKnown,
				Amount: metric.Value,
			}
			continue
		}
		unknown = true
		reason := strings.TrimSpace(metric.Reason)
		if reason == "" {
			reason = "authoritative_usage_missing"
		}
		evidence[identity] = quotaruntime.ActualEvidence{
			State:  quotaruntime.ActualEvidenceUnknown,
			Reason: reason,
		}
	}
	if unknown && strings.TrimSpace(request.FenceID) == "" {
		return quotaruntime.FinalizationResult{}, fmt.Errorf("unknown settlement requires a fence ID")
	}
	if !unknown && request.FenceID != "" {
		return quotaruntime.FinalizationResult{}, fmt.Errorf("known settlement cannot carry a fence ID")
	}
	result, settleErr := r.engine.Finalize(ctx, quotaruntime.FinalizationRequest{
		Partition:          state.tenant.QuotaPartition,
		AdmissionID:        state.tenant.AdmissionID,
		AdmissionDigest:    state.requestDigest,
		FinalizationDigest: request.FinalizationDigest,
		DispatchCount:      snapshot.dispatchCount,
		EvidenceRevision:   snapshot.revision,
		Event:              request.Event,
		FenceID:            request.FenceID,
		Rules:              cloneRuleBindings(state.rules),
		Evidence:           evidence,
	})
	if settleErr != nil {
		return quotaruntime.FinalizationResult{}, fmt.Errorf("settle inference usage: %w", settleErr)
	}
	return result, nil
}

func (r *Runtime) validateAttemptEvidenceSnapshot(
	snapshot AttemptEvidenceSnapshot,
	admission *admissionState,
) (*attemptEvidenceSnapshotState, error) {
	state := snapshot.state
	if state == nil || state.owner != r.identity || state.dispatchCount == 0 ||
		state.admissionID != admission.tenant.AdmissionID ||
		state.admissionDigest != admission.requestDigest {
		return nil, fmt.Errorf("a complete attempt-evidence snapshot from this runtime is required")
	}
	return state, nil
}

func (r *Runtime) validateAdmitted(admission Admission) (*admissionState, error) {
	state := admission.state
	if !admission.Result.Allowed() || state == nil || state.owner != r.identity ||
		strings.TrimSpace(state.tenant.AdmissionID) == "" ||
		strings.TrimSpace(state.tenant.QuotaPartition) == "" ||
		strings.TrimSpace(state.requestDigest) == "" {
		return nil, fmt.Errorf("a complete allowed admission from this runtime is required")
	}
	if admission.Tenant.AdmissionID != state.tenant.AdmissionID ||
		admission.Tenant.QuotaPartition != state.tenant.QuotaPartition ||
		admission.RequestDigest != state.requestDigest || admission.Target != state.target {
		return nil, fmt.Errorf("admission identity was modified after authorization")
	}
	return state, nil
}
