package extproc

// extractUserID returns only the authenticated authenticated User identity. Request
// headers and body metadata are untrusted and never participate in memory,
// cache, replay, or usage scope.
func extractUserID(ctx *RequestContext) string {
	return cacheScopeUserID(ctx)
}
