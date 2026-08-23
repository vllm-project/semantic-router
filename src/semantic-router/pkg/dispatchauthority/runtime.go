package dispatchauthority

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

// Runtime is the mode-exact process authority borrowed by every immutable
// Router generation. Exactly one named authority exists for its lifetime.
type Runtime struct {
	metered     *MeteredAuthority
	routingOnly *RoutingOnlyAuthority
}

func NewMeteredRuntime(options MeteredAuthorityOptions) (*Runtime, error) {
	authority, err := NewMeteredAuthority(options)
	if err != nil {
		return nil, err
	}
	return &Runtime{metered: authority}, nil
}

func NewRoutingOnlyRuntime(options RoutingOnlyAuthorityOptions) (*Runtime, error) {
	authority, err := NewRoutingOnlyAuthority(options)
	if err != nil {
		return nil, err
	}
	return &Runtime{routingOnly: authority}, nil
}

func (runtime *Runtime) Metered() bool {
	return runtime != nil && runtime.metered != nil && runtime.routingOnly == nil
}

func (runtime *Runtime) IssueMeteredPrimary(request PrimaryIssueRequest) (string, error) {
	if runtime == nil || runtime.metered == nil || runtime.routingOnly != nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	return runtime.metered.IssuePrimary(request)
}

func (runtime *Runtime) IssueMeteredChain(request MeteredChainIssueRequest) (string, error) {
	if runtime == nil || runtime.metered == nil || runtime.routingOnly != nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	return runtime.metered.IssueChain(request)
}

func (runtime *Runtime) IssueMeteredGrant(request GrantIssueRequest) (string, error) {
	if runtime == nil || runtime.metered == nil || runtime.routingOnly != nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	return runtime.metered.IssueGrant(request)
}

func (runtime *Runtime) IssueRoutingOnlyPrimary(
	ctx context.Context,
	request RoutingOnlyIssueRequest,
) (string, error) {
	if runtime == nil || runtime.routingOnly == nil || runtime.metered != nil {
		return "", fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	return runtime.routingOnly.IssuePrimary(ctx, request)
}

func (runtime *Runtime) IssueRoutingOnlyChain(
	ctx context.Context,
	request RoutingOnlyChainIssueRequest,
) (string, error) {
	if runtime == nil || runtime.routingOnly == nil || runtime.metered != nil {
		return "", fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	return runtime.routingOnly.IssueChain(ctx, request)
}

func (runtime *Runtime) IssueRoutingOnlyGrant(
	ctx context.Context,
	request RoutingOnlyGrantIssueRequest,
) (string, error) {
	if runtime == nil || runtime.routingOnly == nil || runtime.metered != nil {
		return "", fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	return runtime.routingOnly.IssueGrant(ctx, request)
}

func (runtime *Runtime) VerifyGrant(
	ctx context.Context,
	token string,
	expected GrantVerificationRequest,
) (VerifiedGrant, error) {
	if runtime == nil {
		return VerifiedGrant{}, fmt.Errorf("dispatch authority is unavailable")
	}
	if runtime.metered != nil && runtime.routingOnly == nil {
		return runtime.metered.VerifyGrant(token, expected)
	}
	if runtime.routingOnly != nil && runtime.metered == nil {
		return runtime.routingOnly.VerifyGrant(ctx, token, expected)
	}
	return VerifiedGrant{}, fmt.Errorf("dispatch authority mode is invalid")
}

func (runtime *Runtime) IssueFromGrant(
	ctx context.Context,
	verified VerifiedGrant,
	request FinalRequest,
) (string, error) {
	if runtime == nil {
		return "", fmt.Errorf("dispatch authority is unavailable")
	}
	if runtime.metered != nil && runtime.routingOnly == nil {
		return runtime.metered.IssueFromGrant(verified, request)
	}
	if runtime.routingOnly != nil && runtime.metered == nil {
		return runtime.routingOnly.IssueFromGrant(ctx, verified, request)
	}
	return "", fmt.Errorf("dispatch authority mode is invalid")
}

func (runtime *Runtime) VerifyDispatchOutcome(
	ctx context.Context,
	token string,
	expected OutcomeVerificationRequest,
) (backendinvoker.DispatchOutcome, error) {
	if runtime == nil || expected.Generation.Validate() != nil || !boundedIdentity(expected.RequestID) {
		return backendinvoker.DispatchOutcome{}, fmt.Errorf("dispatch outcome request context is invalid")
	}
	var outcome backendinvoker.DispatchOutcome
	var err error
	switch {
	case runtime.metered != nil && runtime.routingOnly == nil:
		authority := runtime.metered
		authority.mu.RLock()
		defer authority.mu.RUnlock()
		if authority.closed || authority.issuer == nil {
			return backendinvoker.DispatchOutcome{}, fmt.Errorf("metered dispatch authority is closed")
		}
		outcome, err = authority.issuer.VerifyOutcome(token)
	case runtime.routingOnly != nil && runtime.metered == nil:
		authority := runtime.routingOnly
		authority.mu.RLock()
		defer authority.mu.RUnlock()
		if authority.closed || authority.issuer == nil {
			return backendinvoker.DispatchOutcome{}, fmt.Errorf("routing-only dispatch authority is closed")
		}
		if err = authority.validateGenerationLocked(ctx, expected.Generation, ModelIdentity{}); err == nil {
			outcome, err = authority.issuer.VerifyOutcome(token)
		}
	default:
		return backendinvoker.DispatchOutcome{}, fmt.Errorf("dispatch authority mode is invalid")
	}
	if err != nil {
		return backendinvoker.DispatchOutcome{}, err
	}
	if outcome.NamespaceID != expected.Generation.NamespaceID ||
		outcome.QuotaPartition != expected.Generation.QuotaPartition ||
		outcome.PublicationID != expected.Generation.PublicationID ||
		outcome.RuntimeEpoch != expected.Generation.RuntimeEpoch ||
		outcome.RoutingRevision != expected.Generation.SnapshotRevision ||
		outcome.RoutingDigest != expected.Generation.RoutingDigest ||
		outcome.RequestID != expected.RequestID {
		return backendinvoker.DispatchOutcome{}, fmt.Errorf("dispatch outcome request context mismatch")
	}
	return outcome, nil
}

func (runtime *Runtime) AttachRoutingSnapshots(source backendinvoker.RoutingSnapshotSource) error {
	if runtime == nil {
		return fmt.Errorf("dispatch authority is unavailable")
	}
	if runtime.metered != nil && runtime.routingOnly == nil {
		return nil
	}
	if runtime.routingOnly != nil && runtime.metered == nil {
		return runtime.routingOnly.AttachRoutingSnapshots(source)
	}
	return fmt.Errorf("dispatch authority mode is invalid")
}

func (runtime *Runtime) Close() error {
	if runtime == nil {
		return nil
	}
	if runtime.metered != nil && runtime.routingOnly == nil {
		return runtime.metered.Close()
	}
	if runtime.routingOnly != nil && runtime.metered == nil {
		return runtime.routingOnly.Close()
	}
	return nil
}

var (
	_ CapabilityRuntime         = (*Runtime)(nil)
	_ FallbackCapabilityRuntime = (*Runtime)(nil)
	_ OutcomeRuntime            = (*Runtime)(nil)
	_ RoutingSnapshotAttacher   = (*Runtime)(nil)
)
