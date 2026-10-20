package extproc

import (
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
)

// appendUpstreamEncodingHeader asks the provider for a gzip reply when the
// router reads the whole reply and its consumer accepts gzip: a client that
// asked for gzip, or a Looper hop, whose reply the router itself consumes. The
// response-body phase decodes the reply before anything reads it.
//
// Every other dispatch keeps the request-header phase's identity default.
// Streamed replies are parsed as they arrive. A client that states no
// preference is not taken to accept gzip, so a data plane that never hands the
// router the reply body (response_body_mode NONE) still sends it identity.
func appendUpstreamEncodingHeader(headersOut *[]*core.HeaderValueOption, ctx *RequestContext) {
	if ctx == nil || isClientStreamingRequested(ctx) {
		return
	}
	if !ctx.LooperRequest && !clientAcceptsGzip(ctx) {
		return
	}
	*headersOut = append(*headersOut, overwriteRequestHeader("accept-encoding", "gzip"))
}

// clientAcceptsGzip reports whether the client's Accept-Encoding allows gzip.
// A missing header states no preference and does not count as acceptance.
func clientAcceptsGzip(ctx *RequestContext) bool {
	value, ok := ctx.Headers["accept-encoding"]
	return ok && acceptsGzip(value)
}

// acceptsGzip applies RFC 9110 section 12.5.3 to an Accept-Encoding value:
// gzip, or its x-gzip alias, is acceptable when listed or covered by "*" with
// a non-zero weight, and an explicit entry takes precedence over "*". An empty
// value accepts no coding.
func acceptsGzip(value string) bool {
	explicit, wildcard := -1.0, -1.0
	for _, entry := range strings.Split(value, ",") {
		coding, params, _ := strings.Cut(entry, ";")
		switch strings.ToLower(strings.TrimSpace(coding)) {
		case "gzip", "x-gzip":
			explicit = max(explicit, codingWeight(params))
		case "*":
			wildcard = max(wildcard, codingWeight(params))
		}
	}
	if explicit >= 0 {
		return explicit > 0
	}
	return wildcard > 0
}

// codingWeight returns an Accept-Encoding entry's weight: 1 without a q
// parameter, and 0, not acceptable, for a q value outside RFC 9110's qvalue
// grammar.
func codingWeight(params string) float64 {
	for _, param := range strings.Split(params, ";") {
		name, value, found := strings.Cut(param, "=")
		if !found || !strings.EqualFold(strings.TrimSpace(name), "q") {
			continue
		}
		return parseQValue(strings.TrimSpace(value))
	}
	return 1
}

// parseQValue parses qvalue = ( "0" [ "." 0*3DIGIT ] ) / ( "1" [ "." 0*3("0") ] ).
func parseQValue(value string) float64 {
	if value == "" || len(value) > 5 || (value[0] != '0' && value[0] != '1') {
		return 0
	}
	if len(value) > 1 {
		if value[1] != '.' {
			return 0
		}
		for _, digit := range value[2:] {
			if digit < '0' || digit > '9' {
				return 0
			}
		}
	}
	weight, err := strconv.ParseFloat(value, 64)
	if err != nil || weight > 1 {
		return 0
	}
	return weight
}
