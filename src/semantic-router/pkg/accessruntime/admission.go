package accessruntime

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
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
	recovery, err := authoritativeAdmissionRecovery(state.tenant, request.Target, request.Recovery)
	if err != nil {
		return Admission{Result: unavailable("invalid_admission_recovery"), Target: request.Target}, err
	}

	result, err = r.engine.Admit(ctx, quotaruntime.AdmissionRequest{
		Partition:     state.tenant.QuotaPartition,
		AdmissionID:   request.AdmissionID,
		Digest:        request.RequestDigest,
		LeaseDuration: request.LeaseDuration,
		Preconditions: preconditions,
		Rules:         cloneRuleBindings(state.rules),
		Recovery:      recovery,
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
	if strings.TrimSpace(result.PlanDigest) == "" {
		return Admission{Result: unavailable("atomic_admission_plan_missing"), Target: request.Target},
			fmt.Errorf("atomic admission did not return its immutable plan digest")
	}
	tenant := cloneTenantContext(state.tenant)
	tenant.AdmissionID = request.AdmissionID
	internal := &admissionState{
		owner: r.identity, tenant: cloneTenantContext(tenant), rules: cloneRuleBindings(state.rules),
		target: request.Target, requestDigest: request.RequestDigest,
		planDigest: result.PlanDigest, leaseDuration: request.LeaseDuration,
	}
	return Admission{
		Result: result, Tenant: tenant, Rules: cloneRuleBindings(state.rules), Target: request.Target,
		RequestDigest: request.RequestDigest, PreparedAt: preparedAt, state: internal,
	}, nil
}

func authoritativeAdmissionRecovery(
	tenant TenantContext,
	target Target,
	provided *quotaruntime.AdmissionRecoveryContext,
) (*quotaruntime.AdmissionRecoveryContext, error) {
	if provided == nil {
		return nil, nil
	}
	if tenant.PolicyRevision > math.MaxInt64 {
		return nil, fmt.Errorf("access policy revision exceeds recovery range")
	}
	recovery := *provided
	if recovery.NamespaceID != tenant.NamespaceID ||
		recovery.Principal.APIKeyID != tenant.APIKeyID ||
		recovery.Principal.UserID != tenant.UserID ||
		recovery.Principal.TeamID != tenant.TeamID ||
		recovery.Routing.AccessRevision != int64(tenant.PolicyRevision) ||
		recovery.FallbackDispatch.Currency != tenant.BillingCurrency {
		return nil, fmt.Errorf("admission recovery identity differs from authenticated tenant")
	}
	switch target.ResourceType {
	case accesscontrol.GrantResourceEntrypoint:
		if recovery.Routing.EntrypointID != string(target.ResourceID) {
			return nil, fmt.Errorf("admission recovery entrypoint differs from authorized target")
		}
	case accesscontrol.GrantResourceModel:
		if recovery.FallbackDispatch.ModelID != string(target.ResourceID) {
			return nil, fmt.Errorf("admission recovery Model differs from authorized target")
		}
	}
	return &recovery, nil
}
