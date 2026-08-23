package extproc

import (
	"context"
	"errors"
	"fmt"

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

// ManagedRequestRuntime resolves a caller credential to one authenticated
// session and one exactly pinned immutable Router generation. It owns neither
// dependency and performs no default-namespace fallback.
type ManagedRequestRuntime struct {
	access            InferenceAccessRuntime
	publications      RoutingPublicationReader
	routers           *ManagedRouterRegistry
	dispatch          DispatchCapabilityRuntime
	publicNamespaceID string
}

type ManagedRequestRuntimeOptions struct {
	Access            InferenceAccessRuntime
	PublicNamespaceID string
	Publications      RoutingPublicationReader
	Routers           *ManagedRouterRegistry
	Dispatch          DispatchCapabilityRuntime
}

// NewManagedRequestRuntime composes the request-time managed routing seam.
// Exactly one subject source exists: metered deployments authenticate through
// AccessRuntime, while routing-only deployments expose one operator-selected
// public namespace and never infer a default from mutable store contents.
func NewManagedRequestRuntime(options ManagedRequestRuntimeOptions) (*ManagedRequestRuntime, error) {
	if options.Publications == nil || options.Routers == nil || options.Dispatch == nil {
		return nil, errors.New("managed publications, routers, and backend dispatch are required")
	}
	if options.Access != nil && options.PublicNamespaceID != "" {
		return nil, errors.New("managed access and a public namespace are mutually exclusive")
	}
	if options.Access == nil {
		parsed, err := uuid.Parse(options.PublicNamespaceID)
		if err != nil || parsed == uuid.Nil || parsed.String() != options.PublicNamespaceID {
			return nil, errors.New("managed routing-only requires one canonical public namespace")
		}
	}
	return &ManagedRequestRuntime{
		access: options.Access, publicNamespaceID: options.PublicNamespaceID,
		publications: options.Publications, routers: options.Routers, dispatch: options.Dispatch,
	}, nil
}

type managedExternalResolution struct {
	authentication accessruntime.Authentication
	generation     routingcontext.Generation
	lease          *ManagedRouterLease
}

func (runtime *ManagedRequestRuntime) accessEnabled() bool {
	return runtime != nil && runtime.access != nil
}

func (runtime *ManagedRequestRuntime) resolvePublic() (managedExternalResolution, error) {
	if runtime == nil || runtime.access != nil || runtime.publications == nil || runtime.routers == nil ||
		runtime.publicNamespaceID == "" {
		return managedExternalResolution{}, errors.New("managed public routing runtime is unavailable")
	}
	publication, ok := runtime.publications.CurrentRoutingPublication(runtime.publicNamespaceID)
	if !ok || !publication.Activated() {
		return managedExternalResolution{}, ErrManagedRouterUnavailable
	}
	generation, err := generationForPublication(publication)
	if err != nil {
		return managedExternalResolution{}, err
	}
	lease, err := runtime.routers.Acquire(generationRegistryPin(generation))
	if err != nil {
		return managedExternalResolution{}, err
	}
	return managedExternalResolution{generation: generation, lease: lease}, nil
}

func (runtime *ManagedRequestRuntime) resolveExternal(
	ctx context.Context,
	credential string,
) (managedExternalResolution, quotaruntime.AccessCheckResult, error) {
	if runtime == nil || runtime.access == nil || runtime.publications == nil || runtime.routers == nil {
		return managedExternalResolution{}, unavailableAccessResult("managed_request_runtime_unavailable"), nil
	}
	authentication, err := runtime.access.Authenticate(ctx, accessruntime.AuthenticationRequest{Credential: credential})
	if err != nil || !authentication.Result.Allowed() {
		return managedExternalResolution{}, authentication.Result, err
	}
	generation, err := generationForTenant(authentication.Tenant)
	if err != nil {
		return managedExternalResolution{}, unavailableAccessResult("authenticated_routing_pin_invalid"), err
	}
	if !runtime.publicationMatches(generation) {
		return managedExternalResolution{}, unavailableAccessResult("routing_publication_unavailable"), nil
	}
	lease, err := runtime.routers.Acquire(generationRegistryPin(generation))
	if err != nil {
		return managedExternalResolution{}, unavailableAccessResult("routing_generation_unavailable"), err
	}
	return managedExternalResolution{
		authentication: authentication, generation: generation, lease: lease,
	}, authentication.Result, nil
}

func (runtime *ManagedRequestRuntime) resolveInternal(
	generation routingcontext.Generation,
) (*ManagedRouterLease, error) {
	if runtime == nil || runtime.publications == nil || runtime.routers == nil {
		return nil, errors.New("managed request runtime is unavailable")
	}
	if err := generation.Validate(); err != nil {
		return nil, err
	}
	if !runtime.publicationMatches(generation) {
		return nil, ErrManagedRouterPinMismatch
	}
	return runtime.routers.Acquire(generationRegistryPin(generation))
}

func (runtime *ManagedRequestRuntime) publicationMatches(generation routingcontext.Generation) bool {
	current, ok := runtime.publications.CurrentRoutingPublication(generation.NamespaceID)
	return ok && current.Activated() &&
		current.NamespaceID == generation.NamespaceID &&
		current.QuotaPartition == generation.QuotaPartition &&
		current.PublicationID == generation.PublicationID &&
		current.RuntimeEpoch == generation.RuntimeEpoch &&
		current.DesiredRevision == uint64(generation.SnapshotRevision) &&
		current.RoutingDigest == generation.RoutingDigest
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
	if err := publication.Validate(); err != nil || !publication.Activated() {
		return routingcontext.Generation{}, fmt.Errorf("public routing publication is unavailable")
	}
	generation := routingcontext.Generation{
		NamespaceID: publication.NamespaceID, QuotaPartition: publication.QuotaPartition,
		PublicationID: publication.PublicationID, RuntimeEpoch: publication.RuntimeEpoch,
		SnapshotRevision: int64(publication.DesiredRevision), RoutingDigest: publication.RoutingDigest,
	}
	if err := generation.Validate(); err != nil {
		return routingcontext.Generation{}, fmt.Errorf("public routing generation: %w", err)
	}
	return generation, nil
}

func generationRegistryPin(generation routingcontext.Generation) ManagedRouterGenerationPin {
	return ManagedRouterGenerationPin{
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
