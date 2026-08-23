package extproc

import (
	"context"
	"errors"
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

type replayProcessStream struct {
	ext_proc.ExternalProcessor_ProcessServer
	ctx   context.Context
	first *ext_proc.ProcessingRequest
}

func (stream *replayProcessStream) Context() context.Context {
	return stream.ctx
}

func (stream *replayProcessStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if stream.first != nil {
		first := stream.first
		stream.first = nil
		return first, nil
	}
	return stream.ExternalProcessor_ProcessServer.Recv()
}

func managedRequestHeaders(request *ext_proc.ProcessingRequest) (*core.HeaderMap, error) {
	if request == nil || request.GetRequestHeaders() == nil || request.GetRequestHeaders().GetHeaders() == nil {
		return nil, errors.New("managed ExtProc stream must start with request headers")
	}
	return request.GetRequestHeaders().GetHeaders(), nil
}

func consumeBearerCredential(headerMap *core.HeaderMap) (string, bool) {
	if headerMap == nil {
		return "", false
	}
	values := make([]string, 0, 1)
	filtered := headerMap.Headers[:0]
	for _, header := range headerMap.Headers {
		if header == nil || !strings.EqualFold(strings.TrimSpace(header.Key), "authorization") {
			filtered = append(filtered, header)
			continue
		}
		values = append(values, strings.TrimSpace(extractHeaderValue(header)))
		header.Value = ""
		header.RawValue = nil
	}
	headerMap.Headers = filtered
	if len(values) != 1 {
		return "", false
	}
	parts := strings.Fields(values[0])
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") || parts[1] == "" {
		return "", false
	}
	return parts[1], true
}

func managedInternalGeneration(headerMap *core.HeaderMap) (routingcontext.Generation, bool) {
	if !managedInternalAuthenticated(headerMap) {
		return routingcontext.Generation{}, false
	}
	epoch, err := strconv.ParseUint(strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingRuntimeEpoch)), 10, 64)
	if err != nil {
		return routingcontext.Generation{}, false
	}
	revision, err := strconv.ParseInt(strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingSnapshotRevision)), 10, 64)
	if err != nil {
		return routingcontext.Generation{}, false
	}
	generation := routingcontext.Generation{
		NamespaceID:    strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingNamespace)),
		QuotaPartition: strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingQuotaPartition)),
		PublicationID:  strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingPublication)),
		RuntimeEpoch:   epoch, SnapshotRevision: revision,
		RoutingDigest: strings.TrimSpace(headerMapValue(headerMap, headers.VSRRoutingDigest)),
	}
	return generation, generation.Validate() == nil
}

func managedInternalAuthenticated(headerMap *core.HeaderMap) bool {
	return headerMap != nil &&
		strings.EqualFold(strings.TrimSpace(headerMapValue(headerMap, headers.VSRLooperRequest)), "true") &&
		internalauth.Authenticate(headerMapValue(headerMap, headers.VSRInternalAuth))
}

func headerMapValue(headerMap *core.HeaderMap, name string) string {
	if headerMap == nil {
		return ""
	}
	value := ""
	for _, header := range headerMap.Headers {
		if header != nil && strings.EqualFold(strings.TrimSpace(header.Key), name) {
			if value != "" {
				return ""
			}
			value = extractHeaderValue(header)
		}
	}
	return value
}
