package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// debugHeadersRequested reports whether the request opted into verbose/debug
// response headers via the x-vsr-debug request header (value "true",
// case-insensitive).
//
// It is the shared trigger for the v0.4 "debug/replay mode" header surface
// (issue #2216): when on, headers that the contract otherwise omits or demotes
// to replay are emitted inline for that request — e.g. the same-protocol
// client/upstream markers in #2206, and the demoted intermediate signal headers
// in #2205, which both gate on this predicate.
func debugHeadersRequested(ctx *RequestContext) bool {
	return strings.EqualFold(strings.TrimSpace(headerValueCI(ctx, headers.VSRDebug)), "true")
}
