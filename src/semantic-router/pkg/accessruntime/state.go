package accessruntime

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// runtimeIdentity is deliberately non-zero-sized so distinct Runtime values
// cannot share an address through zero-sized allocation coalescing.
type runtimeIdentity struct {
	marker byte
}

type sessionState struct {
	owner         *runtimeIdentity
	preconditions []quotaruntime.AdmissionPrecondition
	grants        []accessprojection.Grant
	rules         []quotaruntime.RuleBinding
	tenant        TenantContext
	delegation    *delegationIdentity
}

type delegationIdentity struct {
	managementSessionID string
	principalID         string
}

type admissionState struct {
	owner         *runtimeIdentity
	tenant        TenantContext
	rules         []quotaruntime.RuleBinding
	target        Target
	requestDigest string
}

type preparedDispatchState struct {
	owner              *runtimeIdentity
	namespaceID        string
	quotaPartition     string
	publicationID      string
	runtimeEpoch       uint64
	routingRevision    int64
	routingDigest      string
	admissionID        string
	admissionDigest    string
	dispatchID         string
	ordinal            uint32
	dispatchPlanDigest string
}

func cloneDelegationIdentity(source *delegationIdentity) *delegationIdentity {
	if source == nil {
		return nil
	}
	copyOfSource := *source
	return &copyOfSource
}

func (r *Runtime) sessionState(session Session) (*sessionState, error) {
	if session.state == nil || session.state.owner == nil || session.state.owner != r.identity {
		return nil, ErrInvalidSession
	}
	return session.state, nil
}

func sessionPreconditions(source []quotaruntime.AdmissionPrecondition) []quotaruntime.AdmissionPrecondition {
	result := make([]quotaruntime.AdmissionPrecondition, 0, len(source))
	for _, precondition := range source {
		// Authenticate atomically checks the exact verifier that was used for
		// constant-time credential verification. The verifier is deliberately
		// not retained in the session. Every later operation pins the immutable
		// publication, credential identity/status, and policy pointers; replacing
		// a credential therefore requires a publication switch and invalidates
		// the session through those guards.
		if precondition.Field == "secret_hmac" {
			continue
		}
		result = append(result, precondition)
	}
	return result
}

func cloneRuleBindings(source []quotaruntime.RuleBinding) []quotaruntime.RuleBinding {
	if len(source) == 0 {
		return nil
	}
	result := make([]quotaruntime.RuleBinding, len(source))
	for index, binding := range source {
		result[index] = binding
		result[index].CalendarSchedule = append([]quotaruntime.CalendarInterval(nil), binding.CalendarSchedule...)
		rule := binding.Rule
		if rule.WholeLimit != nil {
			value := *rule.WholeLimit
			rule.WholeLimit = &value
		}
		if rule.CostLimit != nil {
			value := *rule.CostLimit
			rule.CostLimit = &value
		}
		if rule.BucketCapacity != nil {
			value := *rule.BucketCapacity
			rule.BucketCapacity = &value
		}
		if rule.RefillAmount != nil {
			value := *rule.RefillAmount
			rule.RefillAmount = &value
		}
		if rule.GCRABurstTolerance != nil {
			value := *rule.GCRABurstTolerance
			rule.GCRABurstTolerance = &value
		}
		result[index].Rule = rule
	}
	return result
}

func cloneTenantContext(source TenantContext) TenantContext {
	result := source
	result.RoutingClaims = make(map[string]routingsnapshot.ClaimValue, len(source.RoutingClaims))
	for key, value := range source.RoutingClaims {
		result.RoutingClaims[key] = value
	}
	return result
}

func classifyReadFailure(err error, unavailableReason string, missing quotaruntime.AdmissionDisposition) quotaruntime.AdmissionResult {
	if errors.Is(err, ErrProjectionNotFound) {
		return quotaruntime.AdmissionResult{Disposition: missing, BlockingReason: "credential_not_found"}
	}
	return unavailable(unavailableReason)
}

func readFailure(err error) error {
	if errors.Is(err, ErrProjectionNotFound) {
		return nil
	}
	return err
}

func unauthenticated(reason string) quotaruntime.AdmissionResult {
	return quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionUnauthenticated, BlockingReason: reason}
}

func forbidden(reason string) quotaruntime.AdmissionResult {
	return quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionForbidden, BlockingReason: reason}
}

func unavailable(reason string) quotaruntime.AdmissionResult {
	return quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionUnavailable, BlockingReason: reason}
}

func accessResult(result quotaruntime.AdmissionResult) quotaruntime.AccessCheckResult {
	return quotaruntime.AccessCheckResult{
		Disposition: result.Disposition,
		ServerTime:  result.ServerTime,
		Reason:      result.BlockingReason,
	}
}
