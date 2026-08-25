package extproc

import (
	"errors"
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// FileRequestRuntime pins every request to the one immutable snapshot
// compiled at process startup. It has no Management, identity, quota, SQL, or
// Valkey dependency; its only mutable state is process-local capability use.
type FileRequestRuntime struct {
	namespaceID  string
	publications RoutingPublicationReader
	dispatch     DispatchCapabilityRuntime
}

type FileRequestRuntimeOptions struct {
	NamespaceID  string
	Publications RoutingPublicationReader
	Dispatch     DispatchCapabilityRuntime
}

func NewFileRequestRuntime(options FileRequestRuntimeOptions) (*FileRequestRuntime, error) {
	if options.NamespaceID == "" || options.Publications == nil || options.Dispatch == nil {
		return nil, errors.New("file routing publication and backend dispatch are required")
	}
	if options.Dispatch.Metered() {
		return nil, errors.New("file routing backend dispatch must be routing-only")
	}
	return &FileRequestRuntime{
		namespaceID:  options.NamespaceID,
		publications: options.Publications,
		dispatch:     options.Dispatch,
	}, nil
}

func (runtime *FileRequestRuntime) generation() (routingcontext.Generation, error) {
	if runtime == nil || runtime.publications == nil {
		return routingcontext.Generation{}, errors.New("file request runtime is unavailable")
	}
	publication, ok := runtime.publications.CurrentRoutingPublication(runtime.namespaceID)
	if !ok || !publication.Activated() {
		return routingcontext.Generation{}, errors.New("file routing publication is unavailable")
	}
	return generationForPublication(publication)
}

func (runtime *FileRequestRuntime) matches(snapshot *routingsnapshot.Snapshot) bool {
	generation, err := runtime.generation()
	return err == nil && snapshot != nil && generation.NamespaceID == snapshot.NamespaceID &&
		generation.SnapshotRevision == snapshot.Revision && generation.RoutingDigest == snapshot.Digest
}

func (service *RouterService) processFileRequest(
	stream ext_proc.ExternalProcessor_ProcessServer,
) error {
	if service == nil || service.fileRequests == nil {
		return errors.New("file request runtime is unavailable")
	}
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	headerMap, err := durableRoutingRequestHeaders(first)
	if err != nil {
		return fmt.Errorf("file request headers: %w", err)
	}
	generation, err := service.fileRequests.generation()
	if err != nil {
		return err
	}
	requestContext, err := routingcontext.WithGeneration(stream.Context(), generation)
	if err != nil {
		return err
	}
	if internalGeneration, internal := durableRoutingInternalGeneration(headerMap); internal {
		if internalGeneration != generation {
			return errors.New("file routing generation does not match the active snapshot")
		}
		requestID := headerMapValue(headerMap, headers.RequestID)
		grant, grantErr := consumeDispatchGrant(
			headerMap, service.fileRequests.dispatch, requestContext, generation, requestID,
		)
		if grantErr != nil {
			return fmt.Errorf("verify file routing dispatch grant: %w", grantErr)
		}
		requestContext = withVerifiedDispatchGrant(requestContext, grant)
	} else {
		// File routing is explicitly public. Caller bearer material has no identity
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
