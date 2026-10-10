package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const envoyUpstreamAddressAttribute = "upstream.address"

func captureUpstreamEndpointAddress(
	ctx *RequestContext,
	responseHeaders *core.HeaderMap,
	attributes map[string]*structpb.Struct,
) {
	if ctx == nil {
		return
	}
	if addr := upstreamAddressFromAttributes(attributes); addr != "" {
		ctx.UpstreamBackendAddress = addr
		return
	}
	if addr := headerValueFromMap(responseHeaders, headers.VSRUpstreamHost); addr != "" {
		ctx.UpstreamBackendAddress = addr
	}
}

func upstreamAddressFromAttributes(attributes map[string]*structpb.Struct) string {
	if len(attributes) == 0 {
		return ""
	}
	return strings.TrimSpace(structAttributeString(attributes, envoyUpstreamAddressAttribute))
}

func structAttributeString(attributes map[string]*structpb.Struct, key string) string {
	st := attributes[key]
	if st == nil {
		return ""
	}
	if v, ok := st.Fields["value"]; ok {
		if s := strings.TrimSpace(v.GetStringValue()); s != "" {
			return s
		}
	}
	for _, v := range st.Fields {
		if s := strings.TrimSpace(v.GetStringValue()); s != "" {
			return s
		}
	}
	return ""
}

func headerValueFromMap(headerMap *core.HeaderMap, name string) string {
	if headerMap == nil || name == "" {
		return ""
	}
	for _, header := range headerMap.GetHeaders() {
		if header == nil || !strings.EqualFold(header.GetKey(), name) {
			continue
		}
		if value := strings.TrimSpace(extractHeaderValue(header)); value != "" {
			return value
		}
	}
	return ""
}
