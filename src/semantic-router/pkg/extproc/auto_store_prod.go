//go:build !dev

package extproc

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.RequestAutoStore == nil {
		return false, false
	}

	return *ctx.RequestAutoStore, true
}
