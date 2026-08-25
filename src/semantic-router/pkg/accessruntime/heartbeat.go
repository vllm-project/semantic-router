package accessruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Heartbeat renews only the liveness of an opaque admission created by this
// Runtime. Callers cannot supply a new lease, rule plan, or counter scope.
func (r *Runtime) Heartbeat(
	ctx context.Context,
	admission Admission,
) (quotaruntime.AdmissionHeartbeatResult, error) {
	state, err := r.validateAdmitted(admission)
	if err != nil {
		return quotaruntime.AdmissionHeartbeatResult{}, err
	}
	if strings.TrimSpace(state.planDigest) == "" || state.leaseDuration <= 0 {
		return quotaruntime.AdmissionHeartbeatResult{}, fmt.Errorf(
			"admission heartbeat identity is unavailable",
		)
	}
	result, err := r.engine.Heartbeat(ctx, quotaruntime.AdmissionHeartbeatRequest{
		Partition:       state.tenant.QuotaPartition,
		AdmissionID:     state.tenant.AdmissionID,
		AdmissionDigest: state.requestDigest,
		PlanDigest:      state.planDigest,
		LeaseDuration:   state.leaseDuration,
		Rules:           cloneRuleBindings(state.rules),
	})
	if err != nil {
		return quotaruntime.AdmissionHeartbeatResult{}, fmt.Errorf("heartbeat inference admission: %w", err)
	}
	return result, nil
}
