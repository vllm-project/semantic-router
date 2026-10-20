package extproc

import (
	"bytes"
	"compress/gzip"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// maxDecodedUpstreamBodyBytes bounds a decoded reply by the limit the protocol
// codec enforces on a provider body, so a reply that inflates past it fails
// as an oversized identity reply does.
var maxDecodedUpstreamBodyBytes = int64(llmprotocol.DefaultPolicy().Limits.BodyBytes)

// isGzipContentEncoding reports whether a reply's Content-Encoding is exactly
// one gzip coding. Identity entries are ignored; any other or stacked coding
// is left as it arrived.
func isGzipContentEncoding(headerMap *core.HeaderMap) bool {
	codings, gzipped := 0, false
	for _, header := range headerMap.GetHeaders() {
		if !strings.EqualFold(header.GetKey(), "content-encoding") {
			continue
		}
		for _, coding := range strings.Split(extractHeaderValue(header), ",") {
			switch strings.ToLower(strings.TrimSpace(coding)) {
			case "", "identity":
			case "gzip", "x-gzip":
				codings++
				gzipped = true
			default:
				codings++
			}
		}
	}
	return gzipped && codings == 1
}

// decodeUpstreamResponseBody decodes a buffered gzip reply before any response
// phase reads it; decoded reports that the client must receive the returned
// body in place of the provider's bytes. A success that does not decode is
// answered with failure, as any unreadable reply is. An error reply that does
// not decode continues with an empty body, so the transport-error contract
// keeps the provider's status.
func (r *OpenAIRouter) decodeUpstreamResponseBody(
	body []byte,
	endOfStream bool,
	ctx *RequestContext,
) (out []byte, decoded bool, failure *ext_proc.ProcessingResponse) {
	if ctx == nil || !ctx.UpstreamResponseGzip || ctx.IsStreamingResponse || !endOfStream || len(body) == 0 {
		return body, false, nil
	}
	out, err := gunzipBounded(body, maxDecodedUpstreamBodyBytes)
	if err == nil {
		return out, true, nil
	}
	logging.ComponentWarnEvent("extproc", "upstream_response_decode_failed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"status":     ctx.UpstreamStatusCode,
		"error":      err.Error(),
	})
	if isUpstreamTransportError(ctx) {
		return nil, true, nil
	}
	metrics.RecordRequestError(ctx.RequestModel, "parse_error")
	return nil, false, r.createErrorResponse(502, "The selected model returned an invalid response")
}

func gunzipBounded(body []byte, limit int64) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()
	return httputil.ReadLimitedBody(reader, limit)
}

// deliverDecodedResponseBody sends the client the decoded reply unless a phase
// already replaced the body, and drops the provider's encoding and compressed
// length. An immediate response carries its own headers and is left alone.
func deliverDecodedResponseBody(response *ext_proc.ProcessingResponse, decoded []byte) {
	if response.GetResponseBody() == nil {
		return
	}
	if response.GetResponseBody().GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, decoded)
	}
	common := response.GetResponseBody().GetResponse()
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	ensureHeaderRemoved(common.HeaderMutation, "content-encoding")
	ensureHeaderRemoved(common.HeaderMutation, "content-length")
}
