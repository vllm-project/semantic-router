package extproc

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

// RoutingPublicationReader exposes only healthy process-local publication
// leases. It must not perform a request-time database read.
type RoutingPublicationReader interface {
	CurrentRoutingPublication(string) (accesspublisher.RuntimePublicationIdentity, bool)
}

// DurableRoutingRequestRuntime resolves a caller credential to one authenticated
// session and one exactly pinned immutable Router generation. It owns neither
// dependency and performs no default-namespace fallback.
type DurableRoutingRequestRuntime struct {
	access            InferenceAccessRuntime
	publications      RoutingPublicationReader
	routers           *DurableRoutingRegistry
	dispatch          DispatchCapabilityRuntime
	publicNamespaceID string
}

type DurableRoutingRequestRuntimeOptions struct {
	Access            InferenceAccessRuntime
	PublicNamespaceID string
	Publications      RoutingPublicationReader
	Routers           *DurableRoutingRegistry
	Dispatch          DispatchCapabilityRuntime
}

// NewDurableRoutingRequestRuntime composes the request-time durable routing seam.
// Exactly one subject source exists: metered deployments authenticate through
// AccessRuntime, while routing-only deployments expose one operator-selected
// public namespace and never infer a default from mutable store contents.
func NewDurableRoutingRequestRuntime(options DurableRoutingRequestRuntimeOptions) (*DurableRoutingRequestRuntime, error) {
	if options.Publications == nil || options.Routers == nil || options.Dispatch == nil {
		return nil, errors.New("durable routing publications, routers, and backend dispatch are required")
	}
	if options.Access != nil && options.PublicNamespaceID != "" {
		return nil, errors.New("native access and a public namespace are mutually exclusive")
	}
	if options.Access == nil {
		parsed, err := uuid.Parse(options.PublicNamespaceID)
		if err != nil || parsed == uuid.Nil || parsed.String() != options.PublicNamespaceID {
			return nil, errors.New("durable routing-only requires one canonical public namespace")
		}
	}
	return &DurableRoutingRequestRuntime{
		access: options.Access, publicNamespaceID: options.PublicNamespaceID,
		publications: options.Publications, routers: options.Routers, dispatch: options.Dispatch,
	}, nil
}

type durableRoutingExternalResolution struct {
	authentication accessruntime.Authentication
	generation     routingcontext.Generation
	lease          *DurableRoutingLease
}

func (runtime *DurableRoutingRequestRuntime) accessEnabled() bool {
	return runtime != nil && runtime.access != nil
}

func (runtime *DurableRoutingRequestRuntime) resolvePublic() (durableRoutingExternalResolution, error) {
	if runtime == nil || runtime.access != nil || runtime.publications == nil || runtime.routers == nil ||
		runtime.publicNamespaceID == "" {
		return durableRoutingExternalResolution{}, errors.New("durable public routing runtime is unavailable")
	}
	publication, ok := runtime.publications.CurrentRoutingPublication(runtime.publicNamespaceID)
	if !ok || !publication.Activated() {
		return durableRoutingExternalResolution{}, ErrDurableRoutingUnavailable
	}
	generation, err := generationForPublication(publication)
	if err != nil {
		return durableRoutingExternalResolution{}, err
	}
	lease, err := runtime.routers.Acquire(generationRegistryPin(generation))
	if err != nil {
		return durableRoutingExternalResolution{}, err
	}
	return durableRoutingExternalResolution{generation: generation, lease: lease}, nil
}

func (runtime *DurableRoutingRequestRuntime) resolveExternal(
	ctx context.Context,
	credential string,
) (durableRoutingExternalResolution, quotaruntime.AccessCheckResult, error) {
	if runtime == nil || runtime.access == nil || runtime.publications == nil || runtime.routers == nil {
		return durableRoutingExternalResolution{}, unavailableAccessResult("routing_request_runtime_unavailable"), nil
	}
	authentication, err := runtime.access.Authenticate(ctx, accessruntime.AuthenticationRequest{Credential: credential})
	if err != nil || !authentication.Result.Allowed() {
		return durableRoutingExternalResolution{}, authentication.Result, err
	}
	generation, err := generationForTenant(authentication.Tenant)
	if err != nil {
		return durableRoutingExternalResolution{}, unavailableAccessResult("authenticated_routing_pin_invalid"), err
	}
	if !runtime.publicationMatches(generation) {
		return durableRoutingExternalResolution{}, unavailableAccessResult("routing_publication_unavailable"), nil
	}
	lease, err := runtime.routers.Acquire(generationRegistryPin(generation))
	if err != nil {
		return durableRoutingExternalResolution{}, unavailableAccessResult("routing_generation_unavailable"), err
	}
	return durableRoutingExternalResolution{
		authentication: authentication, generation: generation, lease: lease,
	}, authentication.Result, nil
}

func (runtime *DurableRoutingRequestRuntime) resolveInternal(
	generation routingcontext.Generation,
) (*DurableRoutingLease, error) {
	if runtime == nil || runtime.publications == nil || runtime.routers == nil {
		return nil, errors.New("durable routing request runtime is unavailable")
	}
	if err := generation.Validate(); err != nil {
		return nil, err
	}
	if !runtime.publicationMatches(generation) {
		return nil, ErrDurableRoutingPinMismatch
	}
	return runtime.routers.Acquire(generationRegistryPin(generation))
}

func (runtime *DurableRoutingRequestRuntime) publicationMatches(generation routingcontext.Generation) bool {
	current, ok := runtime.publications.CurrentRoutingPublication(generation.NamespaceID)
	return ok && current.Activated() &&
		current.NamespaceID == generation.NamespaceID &&
		current.QuotaPartition == generation.QuotaPartition &&
		current.PublicationID == generation.PublicationID &&
		current.RuntimeEpoch == generation.RuntimeEpoch &&
		publicationRevisionMatches(current.DesiredRevision, generation.SnapshotRevision) &&
		current.RoutingDigest == generation.RoutingDigest
}

func publicationRevisionMatches(desired uint64, snapshot int64) bool {
	if snapshot <= 0 {
		return false
	}
	// #nosec G115 -- a positive int64 is fully representable by uint64.
	return desired == uint64(snapshot)
}

func generationForTenant(tenant accessruntime.TenantContext) (routingcontext.Generation, error) {
	generation := routingcontext.Generation{
		NamespaceID: tenant.NamespaceID, QuotaPartition: tenant.QuotaPartition,
		PublicationID: tenant.PublicationID, RuntimeEpoch: tenant.RuntimeEpoch,
		SnapshotRevision: tenant.RoutingRevision, RoutingDigest: tenant.RoutingDigest,
	}
	if err := generation.Validate(); err != nil {
		return routingcontext.Generation{}, fmt.Errorf("authenticated routing generation: %w", err)
	}
	return generation, nil
}

func generationForPublication(publication accesspublisher.RuntimePublicationIdentity) (routingcontext.Generation, error) {
	if err := publication.Validate(); err != nil || !publication.Activated() || publication.DesiredRevision > math.MaxInt64 {
		return routingcontext.Generation{}, fmt.Errorf("public routing publication is unavailable")
	}
	// #nosec G115 -- the publication revision is bounded to MaxInt64 above.
	snapshotRevision := int64(publication.DesiredRevision)
	generation := routingcontext.Generation{
		NamespaceID: publication.NamespaceID, QuotaPartition: publication.QuotaPartition,
		PublicationID: publication.PublicationID, RuntimeEpoch: publication.RuntimeEpoch,
		SnapshotRevision: snapshotRevision, RoutingDigest: publication.RoutingDigest,
	}
	if err := generation.Validate(); err != nil {
		return routingcontext.Generation{}, fmt.Errorf("public routing generation: %w", err)
	}
	return generation, nil
}

func generationRegistryPin(generation routingcontext.Generation) DurableRoutingGenerationPin {
	return DurableRoutingGenerationPin{
		NamespaceID: generation.NamespaceID, QuotaPartition: generation.QuotaPartition,
		PublicationID: generation.PublicationID, RuntimeEpoch: generation.RuntimeEpoch,
		SnapshotRevision: generation.SnapshotRevision, RoutingDigest: generation.RoutingDigest,
	}
}

func unavailableAccessResult(reason string) quotaruntime.AccessCheckResult {
	return quotaruntime.AccessCheckResult{
		Disposition: quotaruntime.AdmissionUnavailable,
		Reason:      reason,
	}
}
