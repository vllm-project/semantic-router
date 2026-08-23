package accessruntime

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Admit reuses an authenticated Session, evaluates the target locally, then
// performs one atomic access-precondition and quota decision. The raw
// credential is neither accepted nor reverified here.
func (r *Runtime) Admit(ctx context.Context, request AdmissionRequest) (Admission, error) {
	if strings.TrimSpace(request.AdmissionID) == "" || strings.TrimSpace(request.RequestDigest) == "" || request.LeaseDuration <= 0 {
		return Admission{}, fmt.Errorf("admission ID, request digest, and positive lease duration are required")
	}
	state, preconditions, result, err := r.prepareTarget(ctx, request.Session, request.Target)
	if err != nil || result.Disposition != quotaruntime.AdmissionAllowed {
		return Admission{Result: result, Target: request.Target}, err
	}
	preparedAt := time.Now().UTC()

	result, err = r.engine.Admit(ctx, quotaruntime.AdmissionRequest{
		Partition:     state.tenant.QuotaPartition,
		AdmissionID:   request.AdmissionID,
		Digest:        request.RequestDigest,
		LeaseDuration: request.LeaseDuration,
		Preconditions: preconditions,
		Rules:         cloneRuleBindings(state.rules),
	})
	if err != nil {
		return Admission{Result: unavailable("atomic_admission_failed"), Target: request.Target}, err
	}
	if !result.Allowed() {
		return Admission{
			Result: result, Rules: cloneRuleBindings(state.rules), Target: request.Target,
			PreparedAt: preparedAt,
		}, nil
	}
	tenant := cloneTenantContext(state.tenant)
	tenant.AdmissionID = request.AdmissionID
	internal := &admissionState{
		owner: r.identity, tenant: cloneTenantContext(tenant), rules: cloneRuleBindings(state.rules),
		target: request.Target, requestDigest: request.RequestDigest,
	}
	return Admission{
		Result: result, Tenant: tenant, Rules: cloneRuleBindings(state.rules), Target: request.Target,
		RequestDigest: request.RequestDigest, PreparedAt: preparedAt, state: internal,
	}, nil
}
