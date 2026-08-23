package extproc

import (
	"errors"
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// StandaloneRequestRuntime pins every request to the one immutable snapshot
// compiled at process startup. It has no Management, identity, quota, SQL, or
// Valkey dependency; its only mutable state is process-local capability use.
type StandaloneRequestRuntime struct {
	namespaceID  string
	publications RoutingPublicationReader
	dispatch     DispatchCapabilityRuntime
}

type StandaloneRequestRuntimeOptions struct {
	NamespaceID  string
	Publications RoutingPublicationReader
	Dispatch     DispatchCapabilityRuntime
}

func NewStandaloneRequestRuntime(options StandaloneRequestRuntimeOptions) (*StandaloneRequestRuntime, error) {
	if options.NamespaceID == "" || options.Publications == nil || options.Dispatch == nil {
		return nil, errors.New("standalone publication and backend dispatch are required")
	}
	if options.Dispatch.Metered() {
		return nil, errors.New("standalone backend dispatch must be routing-only")
	}
	return &StandaloneRequestRuntime{
		namespaceID:  options.NamespaceID,
		publications: options.Publications,
		dispatch:     options.Dispatch,
	}, nil
}

func (runtime *StandaloneRequestRuntime) generation() (routingcontext.Generation, error) {
	if runtime == nil || runtime.publications == nil {
		return routingcontext.Generation{}, errors.New("standalone request runtime is unavailable")
	}
	publication, ok := runtime.publications.CurrentRoutingPublication(runtime.namespaceID)
	if !ok || !publication.Activated() {
		return routingcontext.Generation{}, errors.New("standalone routing publication is unavailable")
	}
	return generationForPublication(publication)
}

func (runtime *StandaloneRequestRuntime) matches(snapshot *routingsnapshot.Snapshot) bool {
	generation, err := runtime.generation()
	return err == nil && snapshot != nil && generation.NamespaceID == snapshot.NamespaceID &&
		generation.SnapshotRevision == snapshot.Revision && generation.RoutingDigest == snapshot.Digest
}

func (service *RouterService) processStandalone(
	stream ext_proc.ExternalProcessor_ProcessServer,
) error {
	if service == nil || service.standalone == nil {
		return errors.New("standalone request runtime is unavailable")
	}
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	headerMap, err := managedRequestHeaders(first)
	if err != nil {
		return fmt.Errorf("standalone request headers: %w", err)
	}
	generation, err := service.standalone.generation()
	if err != nil {
		return err
	}
	requestContext, err := routingcontext.WithGeneration(stream.Context(), generation)
	if err != nil {
		return err
	}
	if internalGeneration, internal := managedInternalGeneration(headerMap); internal {
		if internalGeneration != generation {
			return errors.New("standalone internal routing generation does not match the active snapshot")
		}
		requestID := headerMapValue(headerMap, headers.RequestID)
		grant, grantErr := consumeDispatchGrant(
			headerMap, service.standalone.dispatch, requestContext, generation, requestID,
		)
		if grantErr != nil {
			return fmt.Errorf("verify standalone dispatch grant: %w", grantErr)
		}
		requestContext = withVerifiedDispatchGrant(requestContext, grant)
	} else {
		// Standalone is explicitly public. Caller bearer material has no identity
		// meaning here and must never survive toward a physical backend.
		_, _ = consumeBearerCredential(headerMap)
	}
	lease, err := service.acquireCurrent()
	if err != nil {
		return err
	}
	defer lease.refs.Done()
	return lease.router.Process(&replayProcessStream{
		ExternalProcessor_ProcessServer: stream,
		ctx:                             requestContext,
		first:                           first,
	})
}
