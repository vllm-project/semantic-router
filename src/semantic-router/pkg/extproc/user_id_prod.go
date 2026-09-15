//go:build !dev

package extproc

// extractUserID returns the ingress-derived identity. Production and dev
// builds intentionally share this source; build tags must not reintroduce
// body metadata or raw-header fallbacks.
func extractUserID(ctx *RequestContext) string {
	return trustedIdentityUserID(ctx)
}
