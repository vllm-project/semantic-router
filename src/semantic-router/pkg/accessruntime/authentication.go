package accessruntime

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Authenticate performs the only raw-credential operation in the inference
// access runtime. It verifies the presented secret, reads and verifies one
// immutable projection, then atomically pins the live publication and policy
// before returning a process-local Session.
func (r *Runtime) Authenticate(ctx context.Context, request AuthenticationRequest) (Authentication, error) {
	prepared, result, err := r.prepareCredential(ctx, request.Credential)
	if err != nil || result.Disposition != quotaruntime.AdmissionAllowed {
		return Authentication{Result: accessResult(result)}, err
	}
	checked, err := r.engine.CheckAccess(ctx, quotaruntime.AccessCheckRequest{
		Partition: prepared.tenant.QuotaPartition, Preconditions: prepared.preconditions,
	})
	if err != nil {
		return Authentication{
			Result: quotaruntime.AccessCheckResult{
				Disposition: quotaruntime.AdmissionUnavailable,
				Reason:      "atomic_authentication_failed",
			},
		}, err
	}
	if !checked.Allowed() {
		return Authentication{Result: checked}, nil
	}

	state := &sessionState{
		owner:         r.identity,
		preconditions: sessionPreconditions(prepared.preconditions),
		grants:        append([]accessprojection.Grant(nil), prepared.grants...),
		rules:         cloneRuleBindings(prepared.rules),
		tenant:        cloneTenantContext(prepared.tenant),
		delegation:    cloneDelegationIdentity(prepared.delegation),
	}
	source := AuthenticationSourceAPIKey
	if state.delegation != nil {
		source = AuthenticationSourceDelegated
	}
	return Authentication{
		Result:  checked,
		Tenant:  cloneTenantContext(state.tenant),
		Session: Session{state: state},
		Source:  source,
	}, nil
}

type preparedAuthentication struct {
	preconditions []quotaruntime.AdmissionPrecondition
	grants        []accessprojection.Grant
	rules         []quotaruntime.RuleBinding
	tenant        TenantContext
	delegation    *delegationIdentity
}

func (r *Runtime) prepareCredential(ctx context.Context, presented string) (preparedAuthentication, quotaruntime.AdmissionResult, error) {
	kind, publicID, prepareCredentialErr := accesscredential.PublicID(presented)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, unauthenticated("invalid_credential"), nil
	}
	location, prepareCredentialErr := r.reader.LocateCredential(ctx, kind, publicID)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, classifyReadFailure(prepareCredentialErr, "credential_directory_unavailable", quotaruntime.AdmissionUnauthenticated), readFailure(prepareCredentialErr)
	}
	credential, prepareCredentialErr := r.reader.ReadCredential(ctx, location, kind, publicID)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, classifyReadFailure(prepareCredentialErr, "credential_projection_unavailable", quotaruntime.AdmissionUnauthenticated), readFailure(prepareCredentialErr)
	}
	if credential.Kind != string(kind) {
		return preparedAuthentication{}, unavailable("credential_kind_mismatch"), ErrRuntimeCorrupt
	}
	verification := accesscredential.Digest{
		Kind: kind, PublicID: publicID, PepperVersion: credential.PepperVersion,
		HMAC: append([]byte(nil), credential.SecretHMAC...),
	}
	defer clear(verification.HMAC)
	peppers, found := r.peppers[kind]
	if !found {
		return preparedAuthentication{}, unavailable("credential_pepper_unavailable"), ErrRuntimeCorrupt
	}
	if err := peppers.Verify(presented, verification); err != nil {
		if errors.Is(err, accesscredential.ErrPepperUnavailable) {
			return preparedAuthentication{}, unavailable("credential_pepper_unavailable"), err
		}
		return preparedAuthentication{}, unauthenticated("invalid_credential"), nil
	}
	active, prepareCredentialErr := r.reader.ReadActivePolicy(ctx, location, credential.KeyID)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, classifyReadFailure(prepareCredentialErr, "active_policy_unavailable", quotaruntime.AdmissionUnauthenticated), readFailure(prepareCredentialErr)
	}
	projection, prepareCredentialErr := r.reader.ReadPolicy(ctx, location, active)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, classifyReadFailure(prepareCredentialErr, "policy_projection_unavailable", quotaruntime.AdmissionUnauthenticated), readFailure(prepareCredentialErr)
	}
	if active.KeyID != credential.KeyID || projection.KeyID != credential.KeyID ||
		projection.NamespaceID != location.NamespaceID || projection.QuotaPartition != location.QuotaPartition {
		return preparedAuthentication{}, unavailable("projection_identity_mismatch"), ErrRuntimeCorrupt
	}
	if kind == accesscredential.KindDelegation {
		if credential.Audience != r.delegationAudience || credential.DelegationEpoch != projection.DelegationEpoch ||
			credential.UserID == "" || credential.ManagementSessionID == "" || credential.PrincipalID == "" {
			return preparedAuthentication{}, unauthenticated("delegation_binding_invalid"), nil
		}
		if projection.UserID != "" {
			if credential.UserID != projection.UserID || credential.TeamID != projection.TeamID {
				return preparedAuthentication{}, forbidden("delegation_subject_changed"), nil
			}
		} else if projection.TeamID == "" || credential.TeamID != projection.TeamID {
			return preparedAuthentication{}, forbidden("delegation_team_changed"), nil
		}
	}
	if projection.Revision != active.Revision {
		return preparedAuthentication{}, unavailable("projection_revision_mismatch"), ErrRuntimeCorrupt
	}
	if err := projection.VerifyDigest(active.Digest); err != nil {
		return preparedAuthentication{}, unavailable("projection_digest_mismatch"), fmt.Errorf("%w: %w", ErrRuntimeCorrupt, err)
	}
	preconditions, prepareCredentialErr := compilePreconditions(r.keyPrefix, location, kind, publicID, credential, active, projection)
	if prepareCredentialErr != nil {
		return preparedAuthentication{}, unavailable("projection_precondition_invalid"), fmt.Errorf("%w: %w", ErrRuntimeCorrupt, prepareCredentialErr)
	}
	rules := CompileRuleBindings(projection)
	var delegation *delegationIdentity
	if kind == accesscredential.KindDelegation {
		delegation = &delegationIdentity{
			managementSessionID: credential.ManagementSessionID,
			principalID:         credential.PrincipalID,
		}
		barrierResult, barrierErr := r.checkDelegationBarriers(ctx, delegation)
		if barrierErr != nil || barrierResult.Disposition != quotaruntime.AdmissionAllowed {
			return preparedAuthentication{}, barrierResult, barrierErr
		}
	}
	prepared := preparedAuthentication{
		preconditions: preconditions,
		grants:        append([]accessprojection.Grant(nil), projection.Grants...),
		rules:         cloneRuleBindings(rules),
		tenant:        tenantContext(projection, credential, active),
		delegation:    delegation,
	}
	return prepared, quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed}, nil
}

func (r *Runtime) checkDelegationBarriers(
	ctx context.Context,
	delegation *delegationIdentity,
) (quotaruntime.AdmissionResult, error) {
	if delegation == nil {
		return quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed}, nil
	}
	state, err := r.delegationBarriers.CheckDelegation(ctx, managementauth.DelegationBarrierCheck{
		SessionID: delegation.managementSessionID, PrincipalID: delegation.principalID,
	})
	if err != nil || !state.Ready {
		return unavailable("management_revocation_barriers_unavailable"), fmt.Errorf("%w: delegated Management barriers are unavailable", ErrRuntimeUnavailable)
	}
	if state.SessionDenied {
		return forbidden("management_session_denied"), nil
	}
	if state.PrincipalDenied {
		return forbidden("management_principal_denied"), nil
	}
	return quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed}, nil
}
