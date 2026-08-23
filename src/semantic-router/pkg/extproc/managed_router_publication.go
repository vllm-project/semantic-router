package extproc

import (
	"context"
	"fmt"
	"math"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (registry *ManagedRouterRegistry) compilePublication(
	publication accesspublisher.LoadedRoutingPublication,
	requireActive bool,
) (ManagedRouterGenerationPin, *config.RouterConfig, *routingsnapshot.Snapshot, error) {
	identity := publication.Identity
	if err := identity.Validate(); err != nil {
		return ManagedRouterGenerationPin{}, nil, nil, fmt.Errorf("%w: %w", ErrManagedRouterPublicationCorrupt, err)
	}
	if !identity.Loadable() || (requireActive && !identity.Activated()) {
		return ManagedRouterGenerationPin{}, nil, nil, fmt.Errorf(
			"%w: publication state %q is not loadable", ErrManagedRouterPinMismatch, identity.State,
		)
	}
	if identity.DesiredRevision > math.MaxInt64 || publication.Snapshot.Revision != int64(identity.DesiredRevision) ||
		publication.Snapshot.NamespaceID != identity.NamespaceID {
		return ManagedRouterGenerationPin{}, nil, nil, fmt.Errorf(
			"%w: snapshot identity differs from publication", ErrManagedRouterPublicationCorrupt,
		)
	}
	verified, err := validateManagedLoadedDocuments(publication)
	if err != nil {
		return ManagedRouterGenerationPin{}, nil, nil, err
	}
	compiled, err := config.CompileManagedRoutingSnapshot(registry.bootstrap, verified)
	if err != nil {
		return ManagedRouterGenerationPin{}, nil, nil, fmt.Errorf("%w: %w", ErrManagedRouterPublicationCorrupt, err)
	}
	pin := ManagedRouterGenerationPin{
		NamespaceID:      identity.NamespaceID,
		QuotaPartition:   identity.QuotaPartition,
		PublicationID:    identity.PublicationID,
		RuntimeEpoch:     identity.RuntimeEpoch,
		SnapshotRevision: publication.Snapshot.Revision,
		RoutingDigest:    identity.RoutingDigest,
	}
	if err := pin.validate(); err != nil {
		return ManagedRouterGenerationPin{}, nil, nil, err
	}
	return pin, compiled, verified, nil
}

// validateManagedLoadedDocuments defends the lifecycle boundary even though
// the publication store already promises a verified LoadedRoutingPublication.
// Snapshot bundles are rebuilt here because they are the only values compiled
// into an executable router generation.
func validateManagedLoadedDocuments(
	publication accesspublisher.LoadedRoutingPublication,
) (*routingsnapshot.Snapshot, error) {
	identity, manifest, routing := publication.Identity, publication.Manifest, publication.Routing
	if manifest.PublicationID != identity.PublicationID || manifest.NamespaceID != identity.NamespaceID ||
		manifest.QuotaPartition != identity.QuotaPartition || manifest.DesiredRevision != identity.DesiredRevision ||
		manifest.RuntimeEpoch != identity.RuntimeEpoch || manifest.Digest != identity.ManifestDigest ||
		manifest.RoutingDigest != identity.RoutingDigest || routing.NamespaceID != identity.NamespaceID ||
		routing.DesiredRevision != identity.DesiredRevision || routing.Digest != identity.RoutingDigest ||
		!reflect.DeepEqual(manifest.RoutingResources, routing.ResourceDigests) {
		return nil, fmt.Errorf(
			"%w: manifest, routing document, and publication disagree", ErrManagedRouterPublicationCorrupt,
		)
	}
	verifiedRouting, err := routingsnapshot.Compile(routing.Snapshot.Bundle)
	if err != nil || verifiedRouting.Digest != routing.Snapshot.Digest ||
		routing.Snapshot.Digest != publication.Snapshot.Digest {
		return nil, fmt.Errorf("%w: routing snapshot cannot be rebuilt", ErrManagedRouterPublicationCorrupt)
	}
	verifiedLoaded, err := routingsnapshot.Compile(publication.Snapshot.Bundle)
	if err != nil || verifiedLoaded.Digest != publication.Snapshot.Digest {
		return nil, fmt.Errorf("%w: loaded snapshot cannot be rebuilt", ErrManagedRouterPublicationCorrupt)
	}
	return verifiedLoaded, nil
}

func matchingManagedGeneration(
	generation *managedRouterGeneration,
	identity accesspublisher.RuntimePublicationIdentity,
	pin ManagedRouterGenerationPin,
) error {
	if generation == nil || !generation.identity.SameGeneration(identity) || generation.pin != pin {
		return fmt.Errorf(
			"%w: publication id already names another immutable generation",
			ErrManagedRouterPublicationCorrupt,
		)
	}
	return nil
}

func newestManagedGeneration(set *managedRouterNamespace) *managedRouterGeneration {
	newest := set.active
	for _, generation := range set.generations {
		if newest == nil || compareManagedGenerationOrder(newest.identity, generation.identity) < 0 {
			newest = generation
		}
	}
	return newest
}

func compareManagedGenerationOrder(left, right accesspublisher.RuntimePublicationIdentity) int {
	if left.RuntimeEpoch != right.RuntimeEpoch {
		if left.RuntimeEpoch < right.RuntimeEpoch {
			return -1
		}
		return 1
	}
	if left.DesiredRevision < right.DesiredRevision {
		return -1
	}
	if left.DesiredRevision > right.DesiredRevision {
		return 1
	}
	return 0
}

func managedRouterDigestValid(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("context is required")
	}
	return ctx.Err()
}
