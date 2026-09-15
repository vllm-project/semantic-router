//go:build dev

package extproc

// extractUserID returns the same ingress-derived identity as production. A dev
// build must not turn client-controlled metadata into an authorization source.
func extractUserID(ctx *RequestContext) string {
	return trustedIdentityUserID(ctx)
}
