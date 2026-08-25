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

func (registry *DurableRoutingRegistry) compilePublication(
	publication accesspublisher.LoadedRoutingPublication,
	requireActive bool,
) (DurableRoutingGenerationPin, *config.RouterConfig, *routingsnapshot.Snapshot, error) {
	identity := publication.Identity
	if err := identity.Validate(); err != nil {
		return DurableRoutingGenerationPin{}, nil, nil, fmt.Errorf("%w: %w", ErrDurableRoutingPublicationCorrupt, err)
	}
	if !identity.Loadable() || (requireActive && !identity.Activated()) {
		return DurableRoutingGenerationPin{}, nil, nil, fmt.Errorf(
			"%w: publication state %q is not loadable", ErrDurableRoutingPinMismatch, identity.State,
		)
	}
	if identity.DesiredRevision > math.MaxInt64 {
		return DurableRoutingGenerationPin{}, nil, nil, fmt.Errorf(
			"%w: snapshot identity differs from publication", ErrDurableRoutingPublicationCorrupt,
		)
	}
	// #nosec G115 -- the publication revision is bounded to MaxInt64 above.
	desiredRevision := int64(identity.DesiredRevision)
	if publication.Snapshot.Revision != desiredRevision ||
		publication.Snapshot.NamespaceID != identity.NamespaceID {
		return DurableRoutingGenerationPin{}, nil, nil, fmt.Errorf(
			"%w: snapshot identity differs from publication", ErrDurableRoutingPublicationCorrupt,
		)
	}
	verified, err := validateDurableRoutingLoadedDocuments(publication)
	if err != nil {
		return DurableRoutingGenerationPin{}, nil, nil, err
	}
	compiled, err := config.CompileDurableRoutingSnapshot(registry.bootstrap, verified)
	if err != nil {
		return DurableRoutingGenerationPin{}, nil, nil, fmt.Errorf("%w: %w", ErrDurableRoutingPublicationCorrupt, err)
	}
	pin := DurableRoutingGenerationPin{
		NamespaceID:      identity.NamespaceID,
		QuotaPartition:   identity.QuotaPartition,
		PublicationID:    identity.PublicationID,
		RuntimeEpoch:     identity.RuntimeEpoch,
		SnapshotRevision: publication.Snapshot.Revision,
		RoutingDigest:    identity.RoutingDigest,
	}
	if err := pin.validate(); err != nil {
		return DurableRoutingGenerationPin{}, nil, nil, err
	}
	return pin, compiled, verified, nil
}

// validateDurableRoutingLoadedDocuments defends the lifecycle boundary even though
// the publication store already promises a verified LoadedRoutingPublication.
// Snapshot bundles are rebuilt here because they are the only values compiled
// into an executable router generation.
func validateDurableRoutingLoadedDocuments(
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
			"%w: manifest, routing document, and publication disagree", ErrDurableRoutingPublicationCorrupt,
		)
	}
	verifiedRouting, err := routingsnapshot.Compile(routing.Snapshot.Bundle)
	if err != nil || verifiedRouting.Digest != routing.Snapshot.Digest ||
		verifiedRouting.SemanticDigest != routing.Snapshot.SemanticDigest ||
		routing.Snapshot.Digest != publication.Snapshot.Digest ||
		routing.Snapshot.SemanticDigest != publication.Snapshot.SemanticDigest {
		return nil, fmt.Errorf("%w: routing snapshot cannot be rebuilt", ErrDurableRoutingPublicationCorrupt)
	}
	verifiedLoaded, err := routingsnapshot.Compile(publication.Snapshot.Bundle)
	if err != nil || verifiedLoaded.Digest != publication.Snapshot.Digest ||
		verifiedLoaded.SemanticDigest != publication.Snapshot.SemanticDigest {
		return nil, fmt.Errorf("%w: loaded snapshot cannot be rebuilt", ErrDurableRoutingPublicationCorrupt)
	}
	return verifiedLoaded, nil
}

func matchingDurableRoutingGeneration(
	generation *durableRoutingGeneration,
	identity accesspublisher.RuntimePublicationIdentity,
	pin DurableRoutingGenerationPin,
) error {
	if generation == nil || !generation.identity.SameGeneration(identity) || generation.pin != pin {
		return fmt.Errorf(
			"%w: publication id already names another immutable generation",
			ErrDurableRoutingPublicationCorrupt,
		)
	}
	return nil
}

func newestDurableRoutingGeneration(set *durableRoutingNamespace) *durableRoutingGeneration {
	newest := set.active
	for _, generation := range set.generations {
		if newest == nil || compareDurableRoutingGenerationOrder(newest.identity, generation.identity) < 0 {
			newest = generation
		}
	}
	return newest
}

func reusableDurableRoutingRuntime(
	set *durableRoutingNamespace,
	semanticDigest string,
	runtimeEpoch uint64,
) *durableRoutingRuntime {
	if set == nil {
		return nil
	}
	for _, generation := range set.generations {
		if generation != nil && generation.runtime.matches(semanticDigest, runtimeEpoch) {
			return generation.runtime
		}
	}
	return nil
}

func compareDurableRoutingGenerationOrder(left, right accesspublisher.RuntimePublicationIdentity) int {
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

func durableRoutingDigestValid(value string) bool {
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
