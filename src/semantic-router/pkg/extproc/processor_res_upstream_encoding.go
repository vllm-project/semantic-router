package extproc

import (
	"bytes"
	"compress/gzip"
	"context"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// maxDecodedUpstreamBodyBytes bounds a decoded reply by the limit the protocol
// codec enforces on a provider body, so a reply that inflates past it fails
// as an oversized identity reply does.
var maxDecodedUpstreamBodyBytes = int64(llmprotocol.DefaultPolicy().Limits.BodyBytes)

// upstreamDecodeSlots bounds the replies being decompressed at once: a few
// kilobytes on the wire may inflate to maxDecodedUpstreamBodyBytes, so, as
// pkg/systemone bounds the request bodies it decodes, only a few inflate
// together and the rest wait.
var upstreamDecodeSlots = make(chan struct{}, 4)

// maxFailedReplyExcerptBytes bounds the decoded excerpt of a failed reply the
// native fallback records, as pkg/upstream bounds the bytes it hands over.
const maxFailedReplyExcerptBytes = 64 << 10

// isGzipContentEncoding reports whether a reply's Content-Encoding is exactly
// one gzip coding. Identity entries are ignored; any other or stacked coding
// is left as it arrived.
func isGzipContentEncoding(headerMap *core.HeaderMap) bool {
	var values []string
	for _, header := range headerMap.GetHeaders() {
		if strings.EqualFold(header.GetKey(), "content-encoding") {
			values = append(values, extractHeaderValue(header))
		}
	}
	return isGzipCoding(values)
}

func isGzipCoding(contentEncoding []string) bool {
	coding, single := httputil.ContentCoding(contentEncoding)
	return single && coding == "gzip"
}

// failedReplyBody returns the body excerpt of a failed reply as the fallback
// records it. pkg/upstream hands over the provider's bytes, bounded, so a gzip
// excerpt is compressed and may be cut short; it is decoded as far as it
// reads, matching the decoded reply the response-phase fallback records. A
// gzip excerpt that does not decode is recorded without a body, as the
// response phase continues an undecodable error reply with none.
func failedReplyBody(outcome routing.Outcome) []byte {
	if len(outcome.Body) == 0 || !isGzipCoding(outcome.Header.Values("content-encoding")) {
		return outcome.Body
	}
	reader, err := gzip.NewReader(bytes.NewReader(outcome.Body))
	if err != nil {
		return nil
	}
	defer func() { _ = reader.Close() }()
	body, _ := httputil.ReadTruncatedBody(reader, maxFailedReplyExcerptBytes)
	return body
}

// decodeUpstreamResponseBody decodes a buffered gzip reply before any response
// phase reads it; decoded reports that the client must receive the returned
// body in place of the provider's bytes. A success that does not decode is
// answered with failure, as any unreadable reply is. An error reply that does
// not decode continues with an empty body, so the transport-error contract
// keeps the provider's status.
//
// A buffered reply arrives as one body message, whose end_of_stream is false
// when trailers follow; the non-streaming phases read that message as the
// whole reply, so it is decoded whatever the flag says.
func (r *OpenAIRouter) decodeUpstreamResponseBody(
	body []byte,
	ctx *RequestContext,
) (out []byte, decoded bool, failure *ext_proc.ProcessingResponse) {
	if ctx == nil || !ctx.UpstreamResponseGzip || ctx.IsStreamingResponse || len(body) == 0 {
		return body, false, nil
	}
	out, err := gunzipBounded(ctx.TraceContext, body, maxDecodedUpstreamBodyBytes)
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

// gunzipBounded decodes body into at most limit bytes once a decode slot is
// free, and gives up waiting when requestCtx, if any, is done.
func gunzipBounded(requestCtx context.Context, body []byte, limit int64) ([]byte, error) {
	var gone <-chan struct{}
	if requestCtx != nil {
		gone = requestCtx.Done()
	}
	select {
	case upstreamDecodeSlots <- struct{}{}:
	case <-gone:
		return nil, context.Cause(requestCtx)
	}
	defer func() { <-upstreamDecodeSlots }()
	reader, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()
	return httputil.ReadLimitedBody(reader, limit)
}

// deliverUpstreamResponseBody keeps a buffered gzip reply's headers true to
// the bytes the client receives. A decoded reply is sent decoded unless a
// phase already replaced the body. Whenever the client receives anything but
// the provider's own bytes, the decoded reply or a body a phase wrote, the
// provider's encoding and compressed length are dropped. An immediate response
// carries its own headers and is left alone.
func deliverUpstreamResponseBody(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
	body []byte,
	decoded bool,
) {
	if response.GetResponseBody() == nil || ctx == nil || !ctx.UpstreamResponseGzip || ctx.IsStreamingResponse {
		return
	}
	if decoded && response.GetResponseBody().GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, body)
	}
	common := response.GetResponseBody().GetResponse()
	if common.GetBodyMutation() == nil {
		return
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	ensureHeaderRemoved(common.HeaderMutation, "content-encoding")
	ensureHeaderRemoved(common.HeaderMutation, "content-length")
}
