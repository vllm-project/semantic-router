package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// debugHeadersRequested reports whether the request opted into verbose/debug
// response headers via the x-vsr-debug request header (value "true",
// case-insensitive).
//
// It is the shared trigger for the v0.4 "debug/replay mode" header surface
// (issue #2216): when on, headers that the contract otherwise omits or demotes
// to replay are emitted inline for that request — e.g. the same-protocol
// client/upstream markers in #2206, and the demoted intermediate signal headers
// in #2205, which both gate on this predicate. gateDebugHeaders has already
// dropped the header from a caller that global.router.headers.debug does not
// allow.
func debugHeadersRequested(ctx *RequestContext) bool {
	return strings.EqualFold(strings.TrimSpace(headerValueCI(ctx, headers.VSRDebug)), "true")
}

// gateDebugHeaders drops a client's x-vsr-debug request header unless
// global.router.headers.debug lets this caller turn the debug headers on. The
// headers name the signal and guard rules a request matched, so a client can
// never turn them on by itself.
func (r *OpenAIRouter) gateDebugHeaders(ctx *RequestContext) {
	if ctx == nil || headerValueCI(ctx, headers.VSRDebug) == "" {
		return
	}
	var policy config.RouterHeadersConfig
	var identity config.IdentityConfig
	if r != nil && r.Config != nil {
		policy = r.Config.Headers
		identity = r.Config.Authz.Identity
	}
	userID := headerValueCI(ctx, identity.GetUserIDHeader())
	groups := classification.ParseUserGroups(headerValueCI(ctx, identity.GetUserGroupsHeader()))
	if policy.AllowsDebugHeaders(userID, groups) {
		return
	}
	removeHeaderValueCI(ctx, headers.VSRDebug)
	logging.ComponentDebugEvent("extproc", "debug_headers_refused", map[string]interface{}{
		"request_id": ctx.RequestID,
		"policy":     string(policy.Debug),
	})
}
